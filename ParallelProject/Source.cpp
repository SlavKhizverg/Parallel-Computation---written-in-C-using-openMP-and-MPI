#include "Header.h"



int main(int argc, char *argv[])
{
	int N, K, LIMIT;
	double dT, T, QM;
	double *initValues = (double*)malloc(6 * sizeof(double));
	int *sendcounts, *displs, i;
	int pointsPerSlave,
		pointsInMaster;

	Point* points = 0, *procPoint;
	Cluster* clusters = 0;

	int  namelen, numprocs, myid;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	//init the mpi
	MPI_Init(&argc, &argv);
	//idantify what proces call
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	//what is the size of the MPI_COMM_WORLD
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	//get the proces name
	MPI_Get_processor_name(processor_name, &namelen);

	MPI_Status status;
	MPI_Datatype PointMPIType;
	MPI_Datatype ClusterMPIType;
	//init the structs into mpiType
	createPointMPIType(&PointMPIType);
	createClusterMPIType(&ClusterMPIType);

	if (myid == MASTER)
	{

		readPointsFromFile(&points, INPUT_FNAME, &N, &K, &T, &dT, &LIMIT, &QM);
		initValues[0] = N;
		initValues[1] = K;
		initValues[2] = T;
		initValues[3] = dT;
		initValues[4] = LIMIT;
		initValues[5] = QM;
		printf("N: %d, K: %d, T: %lf, dT: %.2lf, LIMIT: %d, QM: %.2lf \n", N, K, T, dT, LIMIT, QM);
		fflush(stdout);
	}
	//sent to all the initValues
	MPI_Bcast(initValues, 6, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

	if (myid != MASTER)
	{
		N = initValues[0];
		K = initValues[1];
		T = initValues[2];
		dT = initValues[3];
		LIMIT = initValues[4];
		QM = initValues[5];
	}

	pointsPerSlave = N / numprocs;
	pointsInMaster = N - (numprocs - 1) * pointsPerSlave;

	sendcounts = (int*)malloc(numprocs * sizeof(int));
	displs = (int*)malloc(numprocs * sizeof(int));

	//dispels where to start count points
	for (i = 0; i < numprocs; i++)
	{
		if (i == 0)
		{
			sendcounts[i] = pointsInMaster;
			displs[i] = 0;
		}
		else
		{
			sendcounts[i] = pointsPerSlave;
			displs[i] = pointsInMaster + (i - 1)*pointsPerSlave;
		}

	}

	procPoint = (Point*)malloc(sendcounts[myid] * sizeof(Point));
	//send points from master to slave
	MPI_Scatterv(points, sendcounts, displs, PointMPIType, procPoint, sendcounts[myid], PointMPIType, MASTER, MPI_COMM_WORLD);


	kMeans(clusters, K, points, N, T, dT, LIMIT, QM, myid, numprocs, procPoint, sendcounts, displs, PointMPIType, ClusterMPIType);


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	free(clusters);
	free(points);
	MPI_Finalize();
	return 0;
}

void kMeans(Cluster* clusters, int K, Point* points, int N, int T, double dT, int LIMIT, double QM, int rank, int numprocs, Point* procPoint, int* sizes, int* offsets, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType)
{
	const int pointsInProc = sizes[rank];
	int i, lim;
	double t, q;
	char flag, slaveFlag = 0;

	MPI_Status status;

	for (t = 0; t < T; t += dT)
	{
		if (t != 0)//move points only after first iteration
		{
			cudaRecalculatePoints(procPoint, pointsInProc, dT);
			free(clusters);

		}

		if (rank == MASTER)
			initClusters(&clusters, K, procPoint);
		else
			clusters = (Cluster*)malloc(K * sizeof(Cluster));

		MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);


		for (lim = 0; lim < LIMIT; lim++)
		{
			flag = 0;

			cudaGroupPoints(clusters, K, procPoint, pointsInProc, &flag);
			/*sendbuf
			starting address of send buffer(choice)
			sendcount
			number of elements in send buffer(integer)
			sendtype
			data type of send buffer elements(handle)
			recvcounts
			integer array(of length group size) containing the number of elements that are received from each process(significant only at root)
			displs
			integer array(of length group size).Entry i specifies the displacement relative to recvbuf at which to place the incoming data from process i(significant only at root)
			recvtype
			data type of recv buffer elements(significant only at root) (handle)
			root
			rank of receiving process(integer)
			comm
			communicator(handle)
			*/
			MPI_Gatherv(procPoint, pointsInProc, PointMPIType, points, sizes, offsets, PointMPIType, MASTER, MPI_COMM_WORLD);

			if (rank == MASTER)
			{
				for (i = 1; i < numprocs; i++)
				{
					MPI_Recv(&slaveFlag, 1, MPI_CHAR, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

					flag += slaveFlag;

				}
			}
			else
				MPI_Send(&flag, 1, MPI_CHAR, MASTER, 0, MPI_COMM_WORLD);

			if (rank == MASTER)
			{
				updateClusters(clusters, K, points, N);
				recalculateClusters(clusters, K);
			}
			//all the process get the new clusters
			MPI_Bcast(clusters, K, ClusterMPIType, MASTER, MPI_COMM_WORLD);

			MPI_Bcast(&flag, 1, MPI_CHAR, MASTER, MPI_COMM_WORLD);


			if (flag == 0) //check if any points switched clusters during the iteration
				lim = LIMIT;
		}

		if (rank == MASTER)
		{

			calcDiameters(clusters, K, points, N);
			q = QualityMasureResult(clusters, K);

			printf("t = %.2lf\n", t);
			printClusters(clusters, K);
		}

		MPI_Bcast(&q, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

		if (q <= QM)
		{
			if (rank == MASTER)
			{
				writeOutputFile(OUTPUT_FNAME, clusters, K, t, q);
				printf("\n Quality Measure Reached    \n");
			}
			t = T;
		}
	}

}

void initClusters(Cluster** clusters, int K, Point* points)
{
	int i;
	*clusters = (Cluster*)malloc(K * sizeof(Cluster));
#pragma omp for
	for (i = 0; i < K; i++)
	{
		(*clusters)[i].ID = i;
		(*clusters)[i].diameter = 0;
		(*clusters)[i].centerX = points[i].x;
		(*clusters)[i].centerY = points[i].y;
		(*clusters)[i].centerZ = points[i].z;
		(*clusters)[i].sumX = 0;
		(*clusters)[i].sumY = 0;
		(*clusters)[i].sumZ = 0;
		(*clusters)[i].counter = 0;
	}
}



void recalculateClusters(Cluster* clusters, int K)
{
	int i;
#pragma omp for
	for (i = 0; i < K; i++)
	{//caculate the average position of points in cluster to get the center
		clusters[i].centerX = clusters[i].sumX / clusters[i].counter;
		clusters[i].centerY = clusters[i].sumY / clusters[i].counter;
		clusters[i].centerZ = clusters[i].sumZ / clusters[i].counter;
	}
}

void updateClusters(Cluster* clusters, int K, Point* points, int N) {
	int i;

#pragma omp for
	for (i = 0; i < K; i++)
	{
		clusters[i].sumX = 0;
		clusters[i].sumY = 0;
		clusters[i].sumZ = 0;
		clusters[i].counter = 0;
	}

	for (i = 0; i < N; i++)
	{
		clusters[points[i].clusterID].sumX += points[i].x;
		clusters[points[i].clusterID].sumY += points[i].y;
		clusters[points[i].clusterID].sumZ += points[i].z;
		clusters[points[i].clusterID].counter++;
	}

}


void calcDiameters(Cluster* clusters, int K, Point* points, int N)
{
	int i, j;
	double distance, diameter;
#pragma omp for private (diameter)
	for (i = 0; i < N; i++)
	{
		diameter = clusters[points[i].clusterID].diameter;
		//use private that evrey thread copyies
#pragma omp parallel for private (distance)
		for (j = i + 1; j < N; j++) // looking for the distances of the farthest 2 points inside a cluster
			if (points[i].clusterID == points[j].clusterID)
			{
				distance = sqrt(pow(points[i].x - points[j].x, 2) + pow(points[i].y - points[j].y, 2) + pow(points[i].z - points[j].z, 2));
				if (diameter < distance)
					diameter = distance;
			}
		clusters[points[i].clusterID].diameter = diameter;
	}
}
//check avarage for each cluster each one check ratio to all others
double QualityMasureResult(Cluster* clusters, int K)
{
	int i, j;
	double devidedDistances = 0, q;
#pragma omp for
	for (i = 0; i < K; i++)

#pragma omp parallel for reduction(+ : devidedDistances)
		for (j = 0; j < K; j++)
			if (i != j)
			{
				devidedDistances += clusters[i].diameter / sqrt(pow(clusters[i].centerX - clusters[j].centerX, 2) + pow(clusters[i].centerY - clusters[j].centerY, 2) + pow(clusters[i].centerZ - clusters[j].centerZ, 2));

			}
	q = devidedDistances / (K * (K - 1));
	printf("\n Quality Measure result : %.2lf ;", q);
	return q;
}

void readPointsFromFile(Point** buffer, char* fileName, int* N, int* K, double* T, double* dT, int* LIMIT, double* QM)
{
	FILE *f = fopen(fileName, "r");
	int i;
	double x, y, z, Vx, Vy, Vz;
	int res;

	if (!f)
	{
		printf("File not found.");
		fflush(stdout);
		return;
	}

	res = fscanf_s(f, "%d", N); // number of points
	res = fscanf_s(f, "%d", K); //number of clusters
	res = fscanf_s(f, "%lf", T); //maximum time limit
	res = fscanf_s(f, "%lf", dT); //time interval
	res = fscanf_s(f, "%d", LIMIT); //maximum iterations limit
	res = fscanf_s(f, "%lf", QM); //quality mesurement for stopping condition

	fflush(stdout);

	*buffer = (Point*)malloc(*N * sizeof(Point));

	for (i = 0; i < *N; i++)
	{
		res = fscanf(f, "%lf", &x);
		res = fscanf(f, "%lf", &y);
		res = fscanf(f, "%lf", &z);
		res = fscanf(f, "%lf", &Vx);
		res = fscanf(f, "%lf", &Vy);
		res = fscanf(f, "%lf", &Vz);

		(*buffer)[i].x = x;
		(*buffer)[i].y = y;
		(*buffer)[i].z = z;
		(*buffer)[i].Vx = Vx;
		(*buffer)[i].Vy = Vy;
		(*buffer)[i].Vz = Vz;
		(*buffer)[i].clusterID = 0;
	}
	fclose(f);
}

void writeOutputFile(char* fileName, Cluster* clusters, int K, double t, double q)
{
	FILE* f = fopen(fileName, "w");
	int i;
	if (f == NULL)
	{
		printf("Failed opening the file. Exiting!\n");
		fflush(stdout);
		return;
	}

	fprintf(f, "First occurrence t = %.2lf with q = %.2lf \n", t, q);
	fprintf(f, "Centers of the clusters : \n");
	for (i = 0; i < K; i++)
	{
		fprintf(f, "%lf	%lf	%lf \n", clusters[i].centerX, clusters[i].centerY, clusters[i].centerZ);
	}


	fclose(f);
}


void printClusters(Cluster *clusters, int K)
{
	int i;
	for (i = 0; i < K; i++)
	{
		printf("\n ClusterId: %d  centerX %.2lf; centerY %.2lf; centerZ %.2lf; diameter: %.2lf; numOfPoints: %d;  \n", clusters[i].ID, clusters[i].centerX, clusters[i].centerY, clusters[i].centerZ, clusters[i].diameter, clusters[i].counter);

	}

	fflush(stdout);
}
//create new mpidatatype as PointMPIType
void createPointMPIType(MPI_Datatype* PointMPIType)
{
	//the typs in the struct point
	MPI_Datatype type[7] = { MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE,MPI_DOUBLE,  MPI_DOUBLE, MPI_INT };
	int blocklen[7] = { 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[7] = { offsetof(Point, x), offsetof(Point, y),offsetof(Point, z), offsetof(Point, Vx), offsetof(Point, Vy),offsetof(Point, Vz), offsetof(Point, clusterID) };

	MPI_Type_create_struct(7, blocklen, disp, type, PointMPIType);
	MPI_Type_commit(PointMPIType);
}
//create new mpidatatype as PointMPIType
void createClusterMPIType(MPI_Datatype* ClusterMPIType)
{
	//the typs in the struct cluster
	MPI_Datatype type[9] = { MPI_INT, MPI_DOUBLE,MPI_DOUBLE, MPI_DOUBLE,   MPI_DOUBLE,  MPI_DOUBLE,  MPI_DOUBLE, MPI_DOUBLE, MPI_INT };
	int blocklen[9] = { 1, 1, 1, 1, 1, 1, 1, 1, 1 };
	MPI_Aint disp[9] = { offsetof(Cluster, ID), offsetof(Cluster, centerX), offsetof(Cluster, centerY),offsetof(Cluster, centerZ), offsetof(Cluster, sumX), offsetof(Cluster, sumY),offsetof(Cluster, sumZ), offsetof(Cluster, counter) };

	MPI_Type_create_struct(9, blocklen, disp, type, ClusterMPIType);
	MPI_Type_commit(ClusterMPIType);
}

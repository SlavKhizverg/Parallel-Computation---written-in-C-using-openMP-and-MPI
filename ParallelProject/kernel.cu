#include "Header.h"


__global__ void moveKernel(Point* points, int N, double dT)
{
	//uonic index for each thrad
	//move the point on the fastest way and initlize the clasters
	const int i = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	if (i < N)
	{
		points[i].x += dT * points[i].Vx;
		points[i].y += dT * points[i].Vy;
		points[i].z += dT * points[i].Vz;
		points[i].clusterID = 0;
	}
}

__global__ void groupKernel(Cluster* clusters, int K, Point* points, int N, char* flag)
{
	//uonic index for each thrad
	const int Pi = blockIdx.x * MAX_NUM_OF_THREADS + threadIdx.x;
	int Ci;
	double newDistance, oldDistance;

	if (Pi < N)
	{
		for (Ci = 0; Ci < K; Ci++)
		{

			oldDistance = sqrt(pow(points[Pi].x - clusters[points[Pi].clusterID].centerX, 2) + pow(points[Pi].y - clusters[points[Pi].clusterID].centerY, 2) + pow(points[Pi].z - clusters[points[Pi].clusterID].centerZ, 2));
			newDistance = sqrt(pow(points[Pi].x - clusters[Ci].centerX, 2) + pow(points[Pi].y - clusters[Ci].centerY, 2) + pow(points[Pi].z - clusters[Ci].centerZ, 2));
			//calculate distances of points from cluster centers and make switch if necessery
			if (newDistance < oldDistance)
			{
				//if flag ==1 finish the lim for loop
				*flag = 1;
				points[Pi].clusterID = clusters[Ci].ID;
			}
		}

	}
}





// Helper function for using CUDA to add vectors in parallel.
cudaError_t cudaRecalculatePoints(Point* points, int N, double dT)
{
	Point *dev_points = 0;
	cudaError_t cudaStatus;
	int blocks = N / MAX_NUM_OF_THREADS + 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaRecalculatePoints - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_points);
		return cudaStatus;
	}

	// Alloc space for device copies
	cudaStatus = cudaMalloc((void**)&dev_points, N * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaRecalculatePoints - cudaMalloc failed!");
		cudaFree(dev_points);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaRecalculatePoints - cudaMemcpy failed!");
		cudaFree(dev_points);
		return cudaStatus;
	}


	// Launch a kernel on the GPU with one thread for each element.
	moveKernel << <blocks, MAX_NUM_OF_THREADS >> > (dev_points, N, dT);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "moveKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_points);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaRecalculatePoints - cudaDeviceSynchronize returned error code %d after launching moveKernel!\n", cudaStatus);
		cudaFree(dev_points);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaRecalculatePoints - cudaMemcpy failed!");
		cudaFree(dev_points);
		return cudaStatus;
	}


	cudaFree(dev_points);
	return cudaStatus;
}

cudaError_t cudaGroupPoints(Cluster* clusters, int K, Point* points, int N, char* flag)
{
	Cluster* dev_clusters = 0;
	Point *dev_points = 0;
	char* dev_flag = 0;
	cudaError_t cudaStatus;
	int blocks = N / MAX_NUM_OF_THREADS + 1;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}


	cudaStatus = cudaMalloc((void**)&dev_points, N * sizeof(Point));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc points failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_clusters, K * sizeof(Cluster));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc clusters failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&dev_flag, sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMalloc flag failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_points, points, N * sizeof(Point), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input points failed!");
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_clusters, clusters, K * sizeof(Cluster), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input clusters failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_flag, flag, sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy input flag failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Launch a kernel on the GPU with one thread for each element.
	groupKernel << <blocks, MAX_NUM_OF_THREADS >> > (dev_clusters, K, dev_points, N, dev_flag);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "groupKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching groupKernel!\n", cudaStatus);
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(points, dev_points, N * sizeof(Point), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy output points failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(flag, dev_flag, sizeof(char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGroupPoints - cudaMemcpy output flag failed!");
		cudaFree(dev_flag);
		cudaFree(dev_points);
		cudaFree(dev_clusters);
		return cudaStatus;
	}

	cudaFree(dev_flag);
	cudaFree(dev_points);
	cudaFree(dev_clusters);
	return cudaStatus;
}


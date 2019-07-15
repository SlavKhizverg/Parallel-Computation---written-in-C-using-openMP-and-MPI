#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>

#define INPUT_FNAME "input.txt"
#define OUTPUT_FNAME "output.txt"
#define MAX_NUM_OF_THREADS 512
#define MASTER 0
#define SLAVE1 1
#define SLAVE2 2

struct Cluster
{
	int ID;
	double diameter;
	double centerX;
	double centerY;
	double centerZ;
	double sumX;
	double sumY;
	double sumZ;
	int counter;
}typedef Cluster;

struct Point
{
	double x;
	double y;
	double z;
	double Vx;
	double Vy;
	double Vz;
	int clusterID;
}typedef Point;

void createClusterMPIType(MPI_Datatype* ClusterMPIType);

void createPointMPIType(MPI_Datatype* PointMPIType);

void calcDiameters(Cluster* clusters, int K, Point* points, int N);

void initClusters(Cluster** clusters, int K, Point* points);

void recalculateClusters(Cluster* clusters, int K);

void kMeans(Cluster* clusters, int K, Point* points, int N, int T, double dT, int LIMIT, double QM, int rank, int numprocs, Point* procPoint, int* sizes, int* offsets, MPI_Datatype PointMPIType, MPI_Datatype ClusterMPIType);
double QualityMasureResult(Cluster* clusters, int K);

void readPointsFromFile(Point** buffer, char* fileName, int* N, int* K, double* T, double* dT, int* LIMIT, double* QM);


void updateClusters(Cluster* clusters, int K, Point* points, int N);

void printClusters(Cluster *clusters, int K);
cudaError_t cudaGroupPoints(Cluster* clusters, int K, Point* points, int N, char* flag);

cudaError_t cudaRecalculatePoints(Point* points, int N, double dT);

void writeOutputFile(char* fileName, Cluster* clusters, int K, double t, double q);
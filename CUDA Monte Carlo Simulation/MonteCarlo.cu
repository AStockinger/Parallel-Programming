// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE				// run 16, 32, 64
#define BLOCKSIZE		32		// number of threads per block 
#endif

#ifndef SIZE
#define SIZE			16000	// 16k, 32k, 64k, 128k, 256k and 512k
#endif

//#ifndef NUMTRIALS	
//#define NUMTRIALS		10		// to make the timing more accurate
//#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN =	 0.0;
const float XCMAX =	 2.0;
const float YCMIN =	 0.0;
const float YCMAX =	 2.0;
const float RMIN  =	 0.5;
const float RMAX  =	 2.0;

float Ranf( float low, float high ){
    float r = (float)rand();              // 0 - RAND_MAX
    float t = r / (float) RAND_MAX;       // 0. - 1.
    return low + t * ( high - low );
}

void TimeOfDaySeed( ){
	struct tm y2k = { 0 };
	y2k.tm_hour = 0; 
    y2k.tm_min = 0; 
    y2k.tm_sec = 0;
	y2k.tm_year = 100; 
    y2k.tm_mon = 0; 
    y2k.tm_mday = 1;
	time_t  timer;
	time( &timer );
	double seconds = difftime( timer, mktime(&y2k) );
	unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
	srand( seed );
}


// Monte Carlo Simulation (CUDA Kernel)
// int array C holds the total number of 'hits' per block in each index
__global__  void MonteCarlo( float *X, float *Y, float *R, int *C ){
	__shared__ int hits[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	hits[tnum] = 1;

	// solve for the intersection using the quadratic formula:	
	float a = 2.;
	float b = -2.*( X[gid] + Y[gid] );
	float c = X[gid]*X[gid] + Y[gid]*Y[gid] - R[gid]*R[gid];
	float d = b*b - 4.*a*c;

	// If d is less than 0., then the circle was completely missed. (Case A) 
	if( d < 0.){
		hits[tnum] = 0;
	}
	else{
		// else it hits the circle...
		// get the first intersection:
		d = sqrt( d );
		float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
		float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

		// If tmin is less than 0., then the circle completely engulfs the laser pointer. (Case B)
		if ( tmin < 0.){
			hits[tnum] = 0;
		}
		else{
			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - X[gid];
			float ny = ycir - Y[gid];
			float n = sqrt( nx * nx + ny * ny );
			nx /= n;	// unit vector
			ny /= n;	// unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt( inx * inx + iny * iny );
			inx /= in;	// unit vector
			iny /= in;	// unit vector

			// get the outgoing (bounced) vector:
			float dot = inx*nx + iny*ny;
			//float outx = inx - 2.*nx*dot;	// angle of reflection = angle of incidence`
			float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = ( 0. - ycir ) / outy;

			// If t is less than 0., then the reflected beam went up instead of down
			if( t < 0.){
				hits[tnum] = 0;
			}
		}
	}

	// add up all hits
	for (int offset = 1; offset < numItems; offset *= 2){
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0){
			hits[tnum] += hits[tnum + offset];
		}
	}

	__syncthreads();
	// record results to array of hits per block
	if (tnum == 0)
		C[wgNum] = hits[0];
}


// main program:
int main( int argc, char* argv[ ] ){
	int dev = findCudaDevice(argc, (const char **)argv);

	TimeOfDaySeed();

	// allocate host memory:
	float * hxcs = new float [ SIZE ];
	float * hycs = new float [ SIZE ];
	float * hrs = new float [ SIZE ];
	int * hC = new int [ SIZE/BLOCKSIZE ];

	// fill in arrays with random values in the given ranges:
	for( int i = 0; i < SIZE; i++ ){
		hxcs[i] = Ranf(XCMIN, XCMAX);
		hycs[i] = Ranf(YCMIN, YCMAX);
		hrs[i] = Ranf(RMIN, RMAX);
	}

	// allocate device memory:
	float *dxcs, *dycs, *drs;
	int *dC;
	dim3 dimsX( SIZE, 1, 1 );
	dim3 dimsY( SIZE, 1, 1 );
	dim3 dimsR( SIZE, 1, 1 );
	dim3 dimsC( SIZE/BLOCKSIZE, 1, 1 );


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dxcs), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dycs), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&drs), SIZE*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dC), (SIZE/BLOCKSIZE)*sizeof(int) );
		checkCudaErrors( status );


	// copy host memory to the device:
	status = cudaMemcpy( dxcs, hxcs, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dycs, hycs, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( drs, hrs, SIZE*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );

	// setup the execution parameters:
	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( SIZE / threads.x, 1, 1 );

	// Create and start timer
	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:
	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:
	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:
	MonteCarlo<<< grid, threads >>>( dxcs, dycs, drs, dC);

	// record the stop event:
	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:
	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance
	double secondsTotal = 0.001 * (double)msecTotal;
	double TrialsPerSecond = (float)SIZE / secondsTotal;
	double megaTrialsPerSecond = TrialsPerSecond / 1000000.;
	fprintf( stderr, "Blocksize = %d, NumTrials = %d, MegaTrials/Second = %10.6lf\n", BLOCKSIZE, SIZE, megaTrialsPerSecond );


	// copy result from the device to the host:
	status = cudaMemcpy( hC, dC, (SIZE/BLOCKSIZE)*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum of all recordings in C:
	int sumHits = 0;
	for(int i = 0; i < SIZE/BLOCKSIZE; i++ ){
		sumHits += hC[i];
	}
	
	// probability around 42
	fprintf( stderr, "probability = %4.6lf\n", (float)sumHits/(float)SIZE);


	// clean up memory:
	delete [ ] hxcs;
	delete [ ] hycs;
	delete [ ] hrs;
	delete [ ] hC;

	status = cudaFree( dxcs );
		checkCudaErrors( status );
	status = cudaFree( dycs );
		checkCudaErrors( status );
	status = cudaFree( drs );
		checkCudaErrors( status );
	status = cudaFree( dC );
		checkCudaErrors( status );

	return 0;
}
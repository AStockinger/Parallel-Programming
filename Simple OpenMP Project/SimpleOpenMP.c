#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define ARRAYSIZE       100000
#define NUMTRIES        20

float A[ARRAYSIZE];
float B[ARRAYSIZE];
float C[ARRAYSIZE];

double mega_mults(float *A, float *B, int threads);

int main()
{
	#ifndef _OPENMP
			fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
			return 1;
	#endif

	// random seed
	time_t t;
	srand((unsigned)time(&t));

	// fill arrays
	for (int i = 0; i < ARRAYSIZE; i++) {
		A[i] = ((float)rand()/(float)(RAND_MAX));
		B[i] = ((float)rand()/(float)(RAND_MAX));
	}

	// calculate execution times
	double threads1 = mega_mults(A, B, 1);
	double threads4 = mega_mults(A, B, 4);

	// calculate and print speedup and parallel fraction
	double S = threads1 / threads4;
	double f = (4. / 3.) * (1. - (1. / S));
	printf("Speedup: %8.2lf\n", S);
	printf("Parallel fraction: %8.2lf\n", f);

	return 0;
}

double mega_mults(float *A, float *B, int threads) {
	omp_set_num_threads(threads);
	fprintf(stderr, "Using %d thread(s)\n", threads);

	double maxMegaMults = 0.;
	double summults = 0.;
	double totalTime = 0.;

	for (int t = 0; t < NUMTRIES; t++)
	{
		double time0 = omp_get_wtime();

		// C does not allow loop variable declaration in loop body, and it must be private
		int i;
		#pragma omp parallel for
		for (i = 0; i < ARRAYSIZE; i++)
		{
			C[i] = A[i] * B[i];
		}

		double time1 = omp_get_wtime();
		totalTime = (time1 - time0);
		double megaMults = (double)ARRAYSIZE / (time1 - time0) / 1000000.;
		summults += megaMults;
		if (megaMults > maxMegaMults)
			maxMegaMults = megaMults;
	}

	printf("Execution time for %d threads: %10.2lf microseconds\n", threads, 1000000. * totalTime);
	// these numbers should be close to be trustworthy
	printf("Peak Performance = %8.2lf MegaMults/Sec\n", maxMegaMults);
	printf("Average Performance = %8.2lf MegaMults/Sec\n", summults / (double)NUMTRIES);

	// note: %lf stands for "long float", which is how printf prints a "double"
	//        %d stands for "decimal integer", not "double"

	return totalTime;
}
#include <omp.h>
#include <time.h>
#include "p4simd.h"

#ifndef LEN
#define LEN 1000
#endif

#ifndef NUMTRIES
#define NUMTRIES 100
#endif

float A[LEN];
float B[LEN];
float C[LEN];

float Randf(unsigned int *seedp, float low, float high) {
  float r = (float)rand_r(seedp);
  return (low + r * (high - low) / (float)RAND_MAX);
}

void Mul(float *A, float *B, float *C, int len) {
    for (int i = 0; i < len; i++) {
        C[i] = A[i] * B[i];
    }
}

float MulSum(float *A, float *B, int len) {
    float sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += A[i] * B[i];
    }
    return sum;
}

int main() {
  #ifndef _OPENMP
        fprintf( stderr, "OpenMP is not supported here -- sorry.\n" );
        return 1;
  #endif

  unsigned int seed = time(NULL);

  double mulAvg = 0.0, mulPeak = 1000.;
  double simdAvg = 0.0, simdPeak = 1000.;
  double redAvg = 0.0, redPeak = 1000.;
  double simdRedAvg = 0.0, simdRedPeak = 1000.;

  // fill arrays
  for (int i = 0; i < LEN; i++) {
    A[i] = Randf(&seed, 0.1f, 1.f);
    B[i] = Randf(&seed, 0.1f, 1.f);
  }

  // non-SIMD mult loop
  for (int i = 0; i < NUMTRIES; i++) {
    double t1 = omp_get_wtime();
    Mul(A, B, C, LEN);
    double t2 = omp_get_wtime();
    double currentMul = (t2 - t1);
    if (currentMul < mulPeak){
      mulPeak = currentMul;
    }
    mulAvg += currentMul;
  }
  mulAvg /= NUMTRIES;

  // SIMD mult loop
  for(int i = 0; i < NUMTRIES; i++){
    double t1 = omp_get_wtime();
    SimdMul(A, B, C, LEN);
    double t2 = omp_get_wtime();
    double currentSIMD = (t2 - t1);
    if(currentSIMD < simdPeak){
        simdPeak = currentSIMD;
    }
    simdAvg += currentSIMD;
  }
  simdAvg /= NUMTRIES;

  // non-SIMD mult/reduction loop
  for(int i = 0; i < NUMTRIES; i++){
    double t1 = omp_get_wtime();
    float sum = MulSum(A, B, LEN);
    double t2 = omp_get_wtime();
    double currentRed = (t2 - t1);
    if (currentRed < redPeak){
      redPeak = currentRed;
    }
    redAvg += currentRed;
  }
  redAvg /= NUMTRIES;

  // SIMD mult/reduction loop
  for(int i = 0; i < NUMTRIES; i++){
    double t1 = omp_get_wtime();
    float sum = SimdMulSum(A, B, LEN);
    double t2 = omp_get_wtime();
    double currentRedSimd = (t2 - t1);
    if(currentRedSimd < simdRedPeak){
        simdRedPeak = currentRedSimd;
    }
    simdRedAvg += currentRedSimd;
  }
  simdRedAvg /= NUMTRIES;

  printf("Array size = %d\n", LEN);
  printf("Mul Peak Performance: %.8lf ms\n", mulPeak * 1000);
  printf("Mul Average Performance: %.8lf ms\n", mulAvg * 1000); 

  printf("SIMD Mul Peak Performance: %.8lf ms\n", simdPeak * 1000);
  printf("SIMD Mul Average Performance: %.8lf ms\n", simdAvg * 1000); 

  printf("Mul Sum Peak Performance: %.8lf ms\n", redPeak * 1000);
  printf("Mul Sum Average Performance: %.8lf ms\n", redAvg * 1000);
  
  printf("SIMD Mul Sum Peak Performance: %.8lf ms\n", simdRedPeak * 1000);
  printf("SIMD Mul Sum Average Performance: %.8lf ms\n\n", simdRedAvg * 1000); 
}
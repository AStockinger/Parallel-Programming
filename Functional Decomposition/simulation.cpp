#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>

// globals for system state
int	NowYear = 2019;		    // 2019 - 2024
int	NowMonth = 0;		    // 0 - 11
float NowPrecip;		    // inches of rain per month
float NowTemp;	    	    // temperature this month
float NowHeight = 1.;       // grain height in inches
int	NowNumDeer = 3;		    // number of deer in the current population
int NowNumPredators = 0;    // number of deer predators in current population

const float GRAIN_GROWS_PER_MONTH =	  8.0;      // grain growth in inches
const float ONE_DEER_EATS_PER_MONTH = 0.5;
const float ONE_PREDATOR_EATS_PER_MONTH = 0.5;

// precipitation in inches
const float AVG_PRECIP_PER_MONTH =	6.0;	// average
const float AMP_PRECIP_PER_MONTH =	6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

// temp in fahrenheit
const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

float Ranf( unsigned int *seedp,  float low, float high ){
    float r = (float) rand_r(seedp);              // 0 - RAND_MAX
    return(low + r * (high - low) / (float)RAND_MAX);
}

int Ranf( unsigned int *seedp, int ilow, int ihigh ){
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;
    return (int)( Ranf(seedp, low, high));
}

float SQR(float x){
    return x*x;
}

void GrainDeer(){
    while( NowYear < 2025 ){
        int nextDeer = NowNumDeer;
        nextDeer -= (int)((float)NowNumPredators * ONE_PREDATOR_EATS_PER_MONTH);
        float capacity = (float)(NowNumDeer * ONE_DEER_EATS_PER_MONTH);
        if(NowHeight > capacity){
            nextDeer++;
        }
        else if(NowHeight < capacity){
            nextDeer--;
        }

        if(nextDeer < 0){
            nextDeer = 0;
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowNumDeer = nextDeer;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void Grain(){
    while( NowYear < 2025 ){
        float nextHeight = NowHeight;
        float tempFactor = exp( -SQR(( NowTemp - MIDTEMP ) / 10. ));
        float precipFactor = exp( -SQR(( NowPrecip - MIDPRECIP ) / 10. ));
        nextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;     // grain grows
        nextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;           // deer eat grain

        if(nextHeight < 0.0){
            nextHeight = 0.0;    // clamp against zero
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowHeight = nextHeight; // copy next state into now state

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void printState(){
    if(NowYear == 2019 && NowMonth == 0){
        printf("Month\tYear\tTemp(C)\tPrecip(cm)\tHeight(cm)\tDeer\tPredators\n");
    }
    printf("%d\t%d\t%8.2lf\t%8.2lf\t%8.2lf\t%d\t%d\n", NowMonth, NowYear, (5.0/9.0 * (NowTemp-32)), (2.54 * NowPrecip), (2.54 * NowHeight), NowNumDeer, NowNumPredators);
}

void Watcher(unsigned int* seed){
    while( NowYear < 2025 ){
        // DoneComputing barrier:
        #pragma omp barrier

        // DoneAssigning barrier:
        #pragma omp barrier

        // print
        printState();

        // increment month
        if(NowMonth == 11){
            NowYear++;
            NowMonth = 0;
        }
        else{
            NowMonth++;
        }

        // calculate new environment
        float ang = ( 30.*(float)NowMonth + 15. ) * ( M_PI / 180. );
        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. ){
            NowPrecip = 0.;
        }

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void Predator(){
    while( NowYear < 2025 ){
        int nextPredator = NowNumPredators;
        float base = (float)(NowNumPredators * ONE_PREDATOR_EATS_PER_MONTH);
        if(base >= NowNumDeer){
            nextPredator -= 2;
        }
        else if(base < NowNumDeer && NowPrecip > 8.0){
            nextPredator++;
        }

        if(nextPredator < 0){
            nextPredator = 0;
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowNumPredators = nextPredator;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

int main(){
    #ifndef _OPENMP
		fprintf( stderr, "No OpenMP support!\n" );
		return 1;
	#endif

    unsigned int seed = time(NULL);

    float ang = ( 30.*(float)NowMonth + 15. ) * ( M_PI / 180. );
    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. ){
        NowPrecip = 0.;
    }

    omp_set_num_threads( 4 );	// same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GrainDeer();
        }

        #pragma omp section
        {
            Grain();
        }

        #pragma omp section
        {
            Watcher(&seed);
        }

        #pragma omp section
        {
            Predator();
        }
    }   // implied barrier -- all functions must return in order to allow any of them to get past here

    return 0;
}
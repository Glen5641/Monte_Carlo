#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <time.h>

/*
 * Monte Carlo Pi Estimation Algorithm in CUDA
 *
 * This Project uses Cuda and thread
 * topology to estimate Pi.
 *
 * Author: Clayton Glenn
 */

#define MAX_THREAD 16
#define MIN_THREAD 8
#define MAX_N 20
#define MIN_N 8
#define BLOCK_SIZE 256
#define DEBUG 0

/** Kernel Function
  * First finds the Thread ID within the block of GPU Threads
  * and if the Thread is Correct, it Encrypts the corresponding
  * Character in the String.
 **/
__global__
void monte(int *flags, float *x_vals, float *y_vals, int t, int n) {

  //Get Thread id
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Loop N/Threads times plus one
  for(int i = 0; i < (n/t + 1); i++){

    // If looped id count is less than n, grab rand x
    // and y and check within unit. Increment if so
    if((i*t+tid) < n){
      if((pow(x_vals[(i*t+tid)], 2) + pow(y_vals[(i*t+tid)],2)) <= 1) flags[(tid)]++;
    }
  }
}

/**
  * Helper Function
  * Prints an string to standard error showing help
  * for valid arguments in the executable
 **/
void printerror(){
  fprintf(stderr, "Invalid Arguments\n");
  fprintf(stderr, "Correct Form: ./monte [# threads] [# points]\n");
  exit(0);
}
/**
  * Main Program
  * This Program is for Homework 6 to encrypt some text or show
  * the encryption method of text that is 2 to the power of N
  * characters long all initialized to zero.
 **/
int main(int argc, char **argv) {

  // Declare a buffer of max size to start
  int N       = MIN_THREAD;
  int THREADS = MIN_THREAD;
  int BLOCKS  = 256;

  // Check for immediate errors in args
  if (argc < 3 || argc > 3) printerror();

  // Get Thread Count Per Block
  THREADS = strtol(argv[1], NULL, 10);
  THREADS = ((int)pow(2, THREADS));
  if(THREADS < BLOCKS) BLOCKS = 1;
  else THREADS = THREADS / BLOCKS;

  // Get N Coordinates
  N = strtol(argv[2], NULL, 10);
  N = (int)pow(2, N);

  // Print N and Threads for distinguish
  printf("(Threads: %d) (N: %d)\n", THREADS * BLOCKS, N);

  //Set Array of Size Thread
  int flags[BLOCKS*THREADS];
  float randx[N];
  float randy[N];
  srand( time( NULL ) );
  for(int i = 0; i < N; i++){
    if(i < BLOCKS*THREADS)flags[i] = 0;
    randx[i] = ( float )rand()/RAND_MAX;
    randy[i] = ( float )rand()/RAND_MAX;
  }

  // Init all other variables
  int *dev_flags;
  float *dev_randx;
  float *dev_randy;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float final_time = 0.0;

  // Allocate memory in the GPU for the int array
  cudaMalloc(&dev_randx, N*sizeof(float));
  cudaMalloc(&dev_randy, N*sizeof(float));
  cudaMalloc(&dev_flags, BLOCKS*THREADS*sizeof(int));

  // Copy the Memory from the array to the array pointers
  cudaMemcpy(dev_flags, flags, BLOCKS*THREADS*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_randx, randx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_randy, randy, N*sizeof(float), cudaMemcpyHostToDevice);

  // Total Time Record
  cudaEventRecord(start);
  monte<<<BLOCKS, THREADS>>>(dev_flags, dev_randx, dev_randy, BLOCKS*THREADS, N);
  cudaEventRecord(stop);

  // Copy the results from GPU to the CPU
  cudaMemcpy(flags, dev_flags, BLOCKS*THREADS*sizeof(int), cudaMemcpyDeviceToHost);

  // Count total successes for each thread
  int success = 0;
  for(int i = 0; i < BLOCKS*THREADS; i++){
    if(flags[i] > 0) success += flags[i];
  }

  // Print Successes, failures, and estimation
  //printf("Success: %d\n", success);
  //printf("Failure: %d\n", (N - success));
  printf("Estimation of Pi: %1.6f\n", ((float)success/N)*4);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&final_time, start, stop);
  printf("Time in Kernel: %1.10f\n\n", final_time/1000);

  cudaFree(dev_flags);
  cudaFree(dev_randx);
  cudaFree(dev_randy);
}

/** 
* Program for facilitating Matrix Multiplication
*/


/**
 * Header files that are used
 */
#include <stdio.h>
#include <stdlib.h>

/**
 * Preprocessor directives
 */
#define EXEC_CPU     0
#define VECTOR_SIZE  1000000000
#define ROWS         3000
#define K            4000
#define COLS         5000
#define BLOCK_SIZE   32 
#define INTSIZE sizeof(unsigned int)

/**
 * Kernel for matrix multiplication (GPU)
 */
__global__ void matMult(int* a, int* b, int* res, unsigned  int rows, unsigned int k, unsigned int cols)
{
    /**
     * Getiing the x and Y dimension iterators
     */
    unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;

    /**
     * Sum being set to 0
     */
    unsigned int sum = 0;

    if(r< rows && c< cols)
    {
        for(int x=0; x<k; x++)
        {
            sum += a[r*k +x] + b[x*cols + c]; 
        }
            /**
              * Storing the result
              */
        res[r*cols + c] = sum;
    }
}

int main( void ) 
{ 

    /** 
     * Set device that we will use for our cuda code
     * It will be either 0 or 1
     */
    cudaSetDevice(1);

    /**
     * Seeding the randomness
     */
    srand(time(NULL));
    
    /** 
     * Time Variables
     */
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    /**
     * Express matrix elements as 1 dimension
     */
    unsigned int aSize =  ROWS * K * INTSIZE;
    unsigned int bSize =  K * COLS* INTSIZE;
    unsigned int cSize =  ROWS * COLS * INTSIZE;

    int *a, *b, *c_cpu, *c_gpu;

    /**
     * Allocating memory on the Host (CPU)
     */
    cudaMallocHost((void**)&a,aSize);
    cudaMallocHost((void**)&b,bSize);
    cudaMallocHost((void**)&c_cpu,cSize);
    cudaMallocHost((void**)&c_gpu,cSize);
 
    /** 
     * Pointers in GPU memory
     */
    int *dev_a;
    int *dev_b;
    int *dev_c;

    /** 
     * Fill the arrays 'a' and 'b' on the CPU
     */
    for(int r=0; r<ROWS; r++)
    {
        for(int c=0; c<K; c++)
        {
            a[ r*K + c] = rand()%10;
        }
    }

    for(int r=0; r<K; r++)
    {
        for(int c=0; c<COLS; c++)
        {
            b[ r*COLS + c ] = rand()%10;
        }
    }

    /** 
     * CPU Calculation
     */
    printf("Running sequential job.\n");

    /**
     * Starting the timer
     */
    cudaEventRecord(start,0);

    if(EXEC_CPU)
    {
        /** 
         * Calculate C in the CPU
         */
        for(unsigned int r=0; r<ROWS; r++)
        {
            for(unsigned int c=0; c<COLS; c++)
            {

                int sum = 0; 
                for(int k=0; k<K;k++)
                {
                    sum +=  a[r*K + k] + b[k*COLS + c];
                }

                c_cpu[r*COLS + c] = sum;
            }
        }
    }

    /**
     * Stopping the timer
     */
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    /**
     * Time elpased
     */
    cudaEventElapsedTime(&time, start, stop);
    printf("\tSequential Job Time: %.2f ms\n", time);

    /** 
     * Allocate the memory on the GPU
     */
    cudaMalloc( (void**)&dev_a,  aSize);
    cudaMalloc( (void**)&dev_b,  bSize);
    cudaMalloc( (void**)&dev_c,  cSize);

    /** 
     * Copy the arrays 'a' and 'b' to the GPU
     */
    cudaMemcpy( dev_a, a, aSize, cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, bSize, cudaMemcpyHostToDevice );

    /** 
     * GPU Calculation
     */
    printf("Running parallel job.\n");

    unsigned int gridRows =  (ROWS + BLOCK_SIZE - 1)/ BLOCK_SIZE; 
    unsigned int gridCols =  (COLS+ BLOCK_SIZE - 1)/ BLOCK_SIZE; 

    /**
     * Grid(s) and Block(s) division
     */
    dim3 grids(gridCols, gridRows);
    dim3 blocks(BLOCK_SIZE, BLOCK_SIZE);

    /**
     * Starting the timer
     */
    cudaEventRecord(start,0);
    
    /**
     * Summon the Kernel
     */
    matMult<<<grids, blocks>>>(dev_a, dev_b, dev_c, ROWS, K, COLS);

    /**
     * Ending the timer
     */
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    /**
     * Time taken
     */
    cudaEventElapsedTime(&time, start, stop);
    printf("\tParallel Job Time: %.2f ms\n", time);

    /**
     * Getting the result(s) back
     */
    cudaMemcpy( c_gpu, dev_c, cSize, cudaMemcpyDeviceToHost);

    if(EXEC_CPU){
        /** 
         * Compare the results
         */
        int error = 0;
        for(unsigned int r=0; r<ROWS; r++)
        {
            for(unsigned int c=0; c<COLS; c++)
            {
                if (c_cpu[r*COLS + c] != c_gpu[r*COLS + c]){
                    error = 1;
                    break;
                }
            }
        }

        if (error == 0)
        {
            printf ("Correct result. No errors were found.\n");

        }
    }

    /**
     * Freeing the memory
     */
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c_cpu);
    cudaFreeHost(c_gpu);

    return 0;
}
/**
 * Header files that are used
 */
#include <stdio.h>
#include <stdlib.h>
#include "primedisk.h"

/**
 * Preprocessor directives
 */
#define EXEC_CPU   0
#define LIMIT      10000
#define BLOCK_SIZE 32

typedef unsigned long long int uint64_cu;
#define INT_SIZE sizeof(uint64_cu)

/**
 * Count the number of primes
 */
uint64_cu countPrime(uint64_cu* arr, uint64_cu len)
{
    uint64_cu pcount = 0;

    for(uint64_cu x=0; x<len; x++)
    {
        if(arr[x]!=0)
            pcount++;
    }
    return pcount;
}

/**
 * Adding primes generated to the array for further processing
 */
void addPrimes(uint64_cu* target, uint64_cu* source,uint64_cu sourcelen)
{
    uint64_cu pindex = 0;

    for(uint64_cu val=0; val<sourcelen; val++)
    {
        if(source[val]!=0)
        {
            target[pindex] = source[val];
            pindex++;
        }
    }
}

/**
 * Kernel for calculating the primes (GPU)
 */
__global__ void calcPrime(uint64_cu* primelist, uint64_cu* inputlist,uint64_cu p_len, uint64_cu i_len)
{
    uint64_cu ind_1 = blockIdx.x * blockDim.x + threadIdx.x;
    // uint64_cu num = primelist[ind_1-1];
    // uint64_cu lastno = inputlist[i_len-1];

    // printf("\n threadId %llu , i_len %llu, p_len %llu",ind_1,i_len, p_len);

    if(ind_1<p_len)
    {
        uint64_cu num = primelist[ind_1];
        // printf("\ncore num %llu\n",num);
        // uint64_cu lastno = inputlist[i_len-1];
        for(uint64_cu start = 0; start< i_len; start++)
        {
            if(inputlist[start] == num) 
                continue;

            if(inputlist[start] % num == 0)
            {
                // printf("CROSSING %llu --- %llu \n",num, inputlist[start]);
                inputlist[start] = 0;
            }
        }
    }
}

int main( void ) 
{
    /** 
     * Getting the GPU
     */
    cudaSetDevice(1);
    srand(time(NULL));
    
    /** 
     * Time Variables
     */
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate (&start);
    cudaEventCreate (&stop);

    /** 
     * Set device that we will use for our cuda code
     * It will be either 0 or 1
     */
    PrimeHeader ret = readPrimes();
    uint64_cu firstLimit = ret.lastMaxNo;
    uint64_cu p_len = ret.length;
    uint64_cu* primelist = ret.primelist;
    
    printf("\n\n ret lastMaxNo-> %llu ",ret.lastMaxNo);
    printf("\tlength -> %llu ",ret.length);
    // printList(ret.primelist ,ret.length);

    printf(" \n\n>>>>>>>>>>>>>> POST FILE READ\n");
    if(ret.length == 0 )
    {
        /** 
         * Start point of the algorithm (CPU seeding)
         */
        firstLimit = 10;
        printf("firstLimit %llu \n", firstLimit);

        uint64_cu firstLimitLen = firstLimit-1;
        printf("firstLimitLen %llu \n", firstLimitLen);

        uint64_cu* firstLimitArray = (uint64_cu*) malloc(firstLimitLen*INT_SIZE);

        for(uint64_cu x=2; x<= firstLimit; x++)
        {
            // printf(" %d %d \t",x-2,x);
            firstLimitArray[x-2] = x;
        }
        // printList(firstLimitArray, firstLimitLen);

        /** 
          * Start the Timer
          */
        cudaEventRecord(start,0);

        for(uint64_cu val = 0; val < firstLimitLen/2; val++)
        {
            uint64_cu num = firstLimitArray[val];
            if(num==0) continue;
            // printf("\n fixing prime %llu ", num);
            for(uint64_cu index=val+1; index< firstLimitLen; index++)
            {
                // printf(" %llu, %llu ", num, firstLimitArray[index]);
                if(firstLimitArray[index]%num== 0 && firstLimitArray[index]!=0)
                    firstLimitArray[index] = 0;
            }
        }

        /** 
          * Stopping the Timer
          */
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        /** 
          * Get the elapsed time
          */
        cudaEventElapsedTime(&time, start, stop);
        // printList(firstLimitArray, firstLimitLen);
        printf("\nSerial Job Time: %.2f ms\n", time);
        // printList(firstLimitArray, firstLimitLen);
        uint64_cu pcount = countPrime(firstLimitArray, firstLimitLen);
        // printf("first round primes %llu",pcount);

        p_len = pcount;
        primelist = (uint64_cu*) malloc(pcount*INT_SIZE);

        /** 
          * Add the primes to the list and Write them to the file
          */
        addPrimes(primelist, firstLimitArray, firstLimitLen);
        writePrimes(primelist,p_len,firstLimit);
    } 

    while(firstLimit < LIMIT*LIMIT)
    {
        printf("\nfirstLimit %llu",firstLimit);
        uint64_cu CUR_MAX = firstLimit;

        uint64_cu startNo = CUR_MAX+1;
        uint64_cu endNo = CUR_MAX * CUR_MAX; 

        uint64_cu range = endNo - CUR_MAX;
        printf("\n######################## startNo %llu , endNo %llu  ########################", startNo, endNo);
        // printf("\nrange %llu\n",range);
        uint64_cu* inputlist = (uint64_cu*) malloc(range*INT_SIZE);

        for(uint64_cu index = 0; index < range; index++)
        {
            inputlist[index] = index + startNo;
        }
        // printList(inputlist,range);

        /** 
         * Pointers in GPU memory
         */
        uint64_cu* dev_ilist;
        uint64_cu* dev_plist;

        // 
        /** 
         * Allocate the memory on the GPU
         */
        cudaMalloc( (void**)&dev_plist,  p_len*INT_SIZE);
        cudaMalloc( (void**)&dev_ilist,  range*INT_SIZE);

        /** 
          * Copy contents to the GPU
          */
        cudaMemcpy( dev_plist, primelist, p_len*INT_SIZE, cudaMemcpyHostToDevice );
        cudaMemcpy( dev_ilist, inputlist, range*INT_SIZE, cudaMemcpyHostToDevice );

        /** 
         * GPU Calculation
         */
        uint64_cu gridSize =  ((p_len + BLOCK_SIZE - 1)/ BLOCK_SIZE) + 1;

        /** 
          * Start the Timer
          */
        cudaEventRecord(start,0);

        /** 
          * Summon the Kernel
          */
        calcPrime<<<gridSize, BLOCK_SIZE>>>(dev_plist, dev_ilist, p_len, range);

        /** 
          * Start the Timer
          */
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        /** 
          * Elapsed time
          */
        cudaEventElapsedTime(&time, start, stop);

        /** 
          * Get the result back to CPU from GPU
          */
        cudaMemcpy( inputlist, dev_ilist, range*INT_SIZE, cudaMemcpyDeviceToHost);
        printf("\n\nUpto %llu , Parallel Job Time: %.2f ms\n",endNo ,time);
        // printList(inputlist,range);

        /** 
         * WRITE primes from INPUTLIST
         */
        uint64_cu ilistPrimeCount = countPrime(inputlist,range);

        printf("ilistPrimeCount %llu \n",ilistPrimeCount);

        uint64_cu* ilistprimes = (uint64_cu*) malloc(ilistPrimeCount*INT_SIZE);

        /** 
          * Add the primes to the list and Write them to the file
          */
        addPrimes(ilistprimes, inputlist, range);
        writePrimes(ilistprimes,ilistPrimeCount,endNo);
        // printList(ilistprimes,ilistPrimeCount);

        /** 
         * APPEND LOGIC
         */
        uint64_cu totalPrimes = p_len + ilistPrimeCount;

        printf("\n%llu totalPrimes for Upto %llu",totalPrimes,endNo);

        uint64_cu* primeNewArray = (uint64_cu*) malloc(totalPrimes*INT_SIZE);
        
        memcpy(primeNewArray,primelist,p_len*INT_SIZE);
        memcpy(primeNewArray+p_len, ilistprimes, ilistPrimeCount*INT_SIZE);
        // printList(primeNewArray, totalPrimes);

        primelist = primeNewArray;
        p_len = totalPrimes;
        firstLimit = endNo;
        fflush(stdout);
    }

    printf("\n**** MAIN END ***\n");
    return 0;
}

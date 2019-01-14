/**
 * Header files that are used
 */
 
#include <stdio.h>
#include <stdlib.h>
#include "primedisk.h"

/**
 * Preprocessor directives
 */
typedef unsigned long long int uint64_cu;
#define INTSIZE sizeof(uint64_cu)

/**
 * main
 */
int main(void)
{ 
	/**
      * Read the primes
      */
    PrimeHeader ret = readPrimes();
    printf("\n\n ret lastMaxNo-> %llu ",ret.lastMaxNo);
    printf("\tlength -> %llu ",ret.length);
    // printList(ret.primelist ,ret.length); 
}

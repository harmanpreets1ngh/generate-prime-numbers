/**
 * Header files that are used
 */
#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <pthread.h>
#include <string>

/**
 * Preprocessor directives
 */
#define ulli unsigned long long int

using namespace std;
using namespace std::chrono;

/**
 * This function takes the lower number and higher number.
 * It opens a file name where the prime numbers are printed to.
 * It uses a brute force algorithm where every number upto the root of the given number is used to divide the 
 * given number. If the number divides, then it is not a prime and will not be added to the file, else it will be added.
 * The execution time is monitered and is printed out at the end of the execution.
 */
void PrimeGen(ulli ln, ulli rn)
{
	/**
	 * Start time being set
	 */
	const steady_clock::time_point begin_time = steady_clock::now();
	
	/**
	 * Initialize fd to open file.
	 */
	ofstream primetxt;
	
	/**
	 * Set filename
	 */
	string filename = "primes_cpu.txt";
	
	/**
	 * Open the file
	 */
	primetxt.open(filename);
	
	/**
	 * Loop over the given lower and higher numbers starts here.
	 * This tests the primality and then write the prime numbers to the 
	 * opened file
	 */
	for (ulli p=ln; p<=rn; ++p)
	{
		bool prime = true;
		ulli n = p;
		for(ulli ip = 2; ip<= sqrt(n); ++ip)
		{
			if(n==2 || n==3)
			{
				primetxt<<endl<<n;
				continue;
			}
			if(n%ip == 0)
			{
				prime = false;
				break;
			}	
		}
		if(prime == false)
		{
			continue;
		}
		primetxt<<endl<<n;
	}
	
	/**
	 * Set end time here.
	 */
	const steady_clock::time_point end_time = steady_clock::now();

	/**
	 * Convert the difference in time to nanoseconds
	 */
	const auto duration_ns = duration_cast<nanoseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to microseconds
	 */
	const auto duration_mis = duration_cast<microseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to milliseconds
	 */
	const auto duration_ms = duration_cast<milliseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to seconds
	 */
	const auto duration_s = duration_cast<seconds>(end_time - begin_time);
	
	/**
	 * Print out the times to screen
	 */
	cout << endl << "EXECUTION TIME "<< endl;
	cout << duration_ns.count() << " nanoseconds\n";
	cout << duration_mis.count() << " microseconds\n";
	cout << duration_ms.count() << " milliseconds\n";
	cout << duration_s.count() << " seconds\n" << endl;
	
	/**
	 * Close the file.
	 */
	primetxt.close();
}

/**
 * The main file takes input of the range of number between which we want to find prime numbers, 
 * from 2 to 18446744073709551615 (maximum range of unsigned long long int)
 * We then call the PrimeGen function to generate the prime numbers.
 */
int main()
{
	/**
	 * Initialize the variables to hold lower number, higher number and number to be run per thread
	 */
	ulli ln, rn;

	/**
	 * Take input of lower and higher numbers
	 */
	cout << "Enter the range from which to generate prime numbers (lower range minimum - 2): ";
	cin >> ln >> rn;
	
	/**
	 * Call the function to generate prime numbers
 	 */
	PrimeGen( ln, rn);

	cout << "Prime number generation completed.\n";
	return 0;
}

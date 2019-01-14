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
 * This function takes the lower number, higher number and the thread number which called the function
 * It opens a file name where the thread_num is appended to separate all the files that have been created.
 * It uses a brute force algorithm where every number upto the root of the given number is used to divide the 
 * given number. If the number divides, then it is not a prime and will not be added to the file, else it will be added.
 * The execution time is monitered and is printed out at the end of the execution.
 */
void PrimeGen(ulli ln, ulli rn, int thread_num)
{
	cout << "Thread " << thread_num << " initialized." << endl;

	/**
	 * Start time being set
	 */
	const thread_local steady_clock::time_point begin_time = steady_clock::now();
	
	/**
	 * Initialize fd to open file.
	 */
	ofstream primetxt;

	/**
	 * Setting filename as per the thread number.
	 */
	string filename = "prime_cpu_thread_", extension = ".txt";
	string newfilename = filename + to_string(thread_num) + extension;
	
	/**
	 * Open the file
	 */
	primetxt.open(newfilename);

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
	const thread_local steady_clock::time_point end_time = steady_clock::now();

	/**
	 * Convert the difference in time to nanoseconds
	 */
	const thread_local auto duration_ns = duration_cast<nanoseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to microseconds
	 */
	const thread_local auto duration_mis = duration_cast<microseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to milliseconds
	 */
	const thread_local auto duration_ms = duration_cast<milliseconds>(end_time - begin_time);
	
	/**
	 * Convert the difference in time to seconds
	 */
	const thread_local auto duration_s = duration_cast<seconds>(end_time - begin_time);
	
	/**
	 * Print out the times to screen
	 */
	cout << endl << "EXECUTION TIME of " << thread_num << endl;
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
 * The main file takes input of the range of number between which we want to find prime numbers, from 2 to 18446744073709551615 (maximum range of unsigned long long int)
 * We then enter the number of threads over which we want to run the program, and a loop runs for that many threads which equally distributing the load among each thread.
 * The threads are then joined at the end.
 */
int main()
{
	/**
	 * Initialize the variables to hold lower number, higher number and number to be run per thread
	 */
	ulli ln, rn, per_thread;

	/**
	 * Initialize the variable to hold number of threads
	 */
	int thread_n;
	
	/**
	 * Take input of lower and higher numbers
	 */
	cout << "Enter the range from which to generate prime numbers (lower range minimum - 2): ";
	cin >> ln >> rn;

	/**
	 * Take input of number of threads
	 */
	cout <<endl<<"Enter the number of threads you want to run for : ";
	cin >> thread_n;

	/**
	 * Create a vector that will hold all the threads that are created.
	 */
	std::vector<thread> t;

	/**
	 * Calculate the ratio of numbers that each thread will test for primality.
	 */
	int total_numbers = rn - ln + 1;
	cout<<"total numbers "<<total_numbers<<endl;
	int ratio_per_thread = ((thread_n+1)*thread_n) / 2;
	cout <<"ratio_per_thread "<< ratio_per_thread<<endl;
	per_thread = total_numbers/ratio_per_thread + 1;
	cout<<"per_thread "<<per_thread<<endl;
	ulli nln = ln, nrn =ln;

	/**
	 * Run a loop that will create threads where each thread has a certain ratio of numbers depending on the number of the thread. The last thread will have the least as it would have the largest numbers
	 * and push the thread into the vector
	 */
	for(int i = 1; i <= thread_n; ++i)
	{
		nln = nrn;
		nrn = (nln + per_thread * (thread_n - i + 1)) < rn ? (nln + per_thread * (thread_n - i + 1)) : rn; 
//		cout <<"new nln and rln "<<nln <<" " <<nrn<<endl;
	t.push_back(thread(PrimeGen, nln, nrn, i));
		++nrn;
	}

	/** 
	 * For each thread in the vector, call the join function
	 */
	for(auto& threads : t)
	{
		threads.join();
	}

	cout << "Prime number generation completed.\n\n";
	return 0;
}

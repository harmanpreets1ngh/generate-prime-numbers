## Connecting to CCIS Server and Running the codes

These codes were run and tested on the CCIS servers.

1. To connect to CCIS Servers,

 `ssh <CCIS_USERNAME>@login.ccs.neu.edu`

2. You will be prompted to enter your password after which you will get access of the machine on the server.

3. Git clone this repository using

 `git clone https://github.com/Summer18CS5600/finalproject-the_last_byte.git`

4. Move into the 'CPU Implementation' folder inside the 'finalproject-the_last_byte.git' folder

 `cd CPU\ Implementation`

5. To create the executables of the given three files, you can use the below commands
  * **cpu_prime_gen_threads_v2.cpp** - This code accepts a range of numbers and the number of threads that you want and unequally distributes the workload among the threads.
  
    The first thread gets the largest ratio and its reduces by the factor of the number of remaining threads to spawn. 
    This approach was done because we saw that the last thread usually took the longest time since it used to have the largest numbers to test for primality in its range.
    
    This compilation can be done using the below command
    
    `g++ -pthread -std=c++11 -o threadedv2 cpu_prime_gen_threads_v2.cpp`
    
    And we can run this using `./threadedv2`
    
    
  * **cpu_prime_gen_threads.cpp** - This code accepts a range of numbers and the number of threads that you want to spawn and equally distirbutes the load among all the threads.
  
    This compilation can be done using the below code
    
    `g++ -pthread -std=c++11 -o threaded cpu_prime_gen_threads.cpp`
    
    And we can run this using `./threaded`
    
  * **cpu_prime_gen.cpp** - This code is the direct method of testing of primality. We just input the range and a single CPU takes on the entire load on its own.
  
    We can compile this by using the below command
  
    `g++ -std=c++11 -o cpu cpu_prime_gen.cpp`
  
    And we can run this using `./cpu`
    
6. There will be some *.txt files that were created on running the above codes. Those files are replaced when the code is run again. For the threaded programs, there will be multiple files created which are appended with the thread number at the end of the file name.

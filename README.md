# Final Project

Youtube Video Link: https://www.youtube.com/watch?v=Wg8UzDGd_WI&feature=youtu.be
___


## Proposal

  * Teammate 1: Aashish Thyagarajan
  * Teammate 2: Harman Preet Singh
  * Teammate 3: Vipul Sharma

### In one sentence, describe your project.

Generating higher order prime numbers utilizing GPU cluster(s). 

### What is the minimum deliverable for this project?

* A working kernel implementation to calculate prime numbers over multiple GPUs with focus on accuracy and correctness (we would need the access to the Discovery Lab for the GPU cluster).

* We will be using the Sieve of Eratosthenes for the prime number calculation.

### What is the maximum deliverable for this project?

* Our focus will be on performance optimization of generating prime numbers.

* We will try to push the upper limit on calculating higher order prime numbers.

### What tools or environment will you be using?

We will need to use:
 * C
 * CUDA API
 * Discovery Lab cluster (multiple GPUs)

### What are your first step(s)?

* Go through the cited publication below to get a better understanding of the project.
* Learn about parallel programming with GPUs and CUDA.
* Try and generate small prime numbers in single GPU implementations on our laptops/desktops.

### Find at least one related publication or tool and reference it here.

[A New High Performance GPU-based Approach to Prime
Numbers Generation](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.683.5203&rep=rep1&type=pdf)

### What is the biggest problem you foresee or question you need to answer to get started?

* How to implement parallel programming using CUDA.
* Ensuring system stability in a multiple GPU environment.
* Effectively avoiding hardware/kernel implementation bottlenecks.
* Getting Discovery Lab access and utilizing it effectively for our project.

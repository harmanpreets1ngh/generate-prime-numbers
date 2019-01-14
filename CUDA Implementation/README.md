## Connecting to Discovery Cluster

1. To connect to Discovery:
`ssh -X <USERNAME>@discovery.neu.edu`

2. After typing the password, this will take you to a gateway node. **YOU SHOULD NOT** execute any work here. Instead you should allocate a node for yourself. To do so, you need to load the modules first.

3.  ~/.bashrc file (in your home directory).

	a. Open file using an editor `vim ~/.bashrc`

        b. Go to the end of the file and copy these lines.

	     # Manually load the required modules
	     module load slurm-14.11.8
	     module load gnu-4.8.1-compilers
	     module load fftw-3.3.3
	     module load openmpi-1.8.3
	     module load cuda-7.0

    c. Reload the file. `source ~/.bashrc`

4. Allocate the resources:

	`salloc -N 1 --exclusive -p par-gpu` # Reserve the node
	
	`squeue -u $USER` # Print the nodes that #USER has reserved

5. This will print the nodes allocated for the user. Copy the value under "NODELIST" and run:

	     JOBID	PARTITION NAME USER ST TIME NODES NODELIST 
	     1135257 par-gpu bash bohmagos R 2:21 1 compute-2-132

6. And run the following command to access your node:
	`ssh -X compute-2-132` # your compute node may be different
7. From now on, you can run your work.

8. To test if everything is working fine, Run “nvidia-smi” command. It should print the information on the GPU’s available on the machine

**IMPORTANT:** When you are done with everything, type the following commands to make the node available to other users.

`exit` # this will take you to the gateway computer again

`squeue -u $USER` # Print the nodes that USER has reserved, you should copy the value under "JOBID"

`scancel 1135257` # free the node so another person can use it 

`squeue -u $USER` # Just to make sure it is gone, check it again

___

## Executing the code

After connecting to the discovery cluster, in order to execute the code:

  type `make clean` to remove all the object files, shared libraries and output files.

type `make` to execute the project.
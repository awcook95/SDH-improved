/* ==================================================================
	Programmer: Yicheng Tu (ytu@cse.usf.edu)
	The basic SDH algorithm implementation for 3D data
	To compile: nvcc SDH.c -o SDH in the C4 lab machines
   ==================================================================
*/

/* USF Fall 2019 CIS4930 Programming on Massively Parallel Systems
   Project Description: Write a CUDA program to implement the same
   functionality as the CPU only code

   Student: Alexander Cook
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>


#define BOX_SIZE	23000 /* size of the data box on one dimension            */

/* descriptors for single atom in the tree */
typedef struct atomdesc {
	double x_pos;
	double y_pos;
	double z_pos;
} atom;

typedef struct hist_entry{
	//float min;
	//float max;
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */
} bucket;

//Global variables
bucket * h_histogram;	/* list of all buckets in the histogram     */
long long	PDH_acnt;	/* total number of data points              */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                               */
atom * atom_list;		/* list of all data points					*/

//Device helper function: Distance of two points in the atom_list
//Device helper function which now takes a pointer as an argument instead of using a global pointer
__device__ double d_p2p_distance(atom *atom_list, int ind1, int ind2) {
	
	double x1 = atom_list[ind1].x_pos;
	double x2 = atom_list[ind2].x_pos;
	double y1 = atom_list[ind1].y_pos;
	double y2 = atom_list[ind2].y_pos;
	double z1 = atom_list[ind1].z_pos;
	double z2 = atom_list[ind2].z_pos;
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__global__ void PDH_kernel(atom *d_atom_list, bucket *d_histogram, int PDH_acnt, int PDH_res, int num_buckets){
	int T = blockIdx.x * blockDim.x + threadIdx.x;
	int t = threadIdx.x;

	extern __shared__ int s_histogram[];

	for(int i = t; i < num_buckets; i += blockDim.x){ //initialize array to 0 in block sized chunks
		s_histogram[i] = 0;
	}

	__syncthreads();
 
	int h_pos;
	double dist;

	for(int i = T + 1; i < PDH_acnt; i++){
		dist = d_p2p_distance(d_atom_list, T, i);
			h_pos = (int) (dist / PDH_res);
			atomicAdd(&(s_histogram[h_pos]), 1);
			 
	}

	__syncthreads();

	//reduce shared output into global output
	for(int i = t; i < num_buckets; i += blockDim.x){ //output to global memory in block sized chunks
		atomicAdd(&(d_histogram[i].d_cnt), s_histogram[i]);
	}

}


/* 
	Print the counts in all buckets of the histogram 
*/
void output_histogram(bucket *histogram){
	int i; 
	long long total_cnt = 0;
	for(i=0; i< num_buckets; i++) {
		if(i%5 == 0) /* we print 5 buckets in a row */
			printf("\n%02d: ", i);
		printf("%15lld ", histogram[i].d_cnt);
		total_cnt += histogram[i].d_cnt;
	  	/* we also want to make sure the total distance count is correct */
		if(i == num_buckets - 1)	
			printf("\n T:%lld \n", total_cnt);
		else printf("| ");
	}
}


int main(int argc, char **argv)
{
	int i;

	PDH_acnt = atoi(argv[1]);
	PDH_res	 = atof(argv[2]);

	//Allocate host memory
	num_buckets = (int)(BOX_SIZE * 1.732 / PDH_res) + 1;
	h_histogram = (bucket *)malloc(sizeof(bucket) * num_buckets);
	atom_list = (atom *)malloc(sizeof(atom)*PDH_acnt);

	//initialize histogram to zero
	memset(h_histogram, 0, sizeof(bucket)*num_buckets);

	//Allocate device memory
	bucket *d_histogram; //pointer to array of buckets
	atom *d_atom_list; //pointer to array of atoms

	cudaMalloc((void**)&d_histogram, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&d_atom_list, sizeof(atom)*PDH_acnt);
	
	srand(1);
	/* Generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		atom_list[i].x_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].y_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		atom_list[i].z_pos = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	//Copy host data to device memory
	cudaMemcpy(d_histogram, h_histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_list, atom_list, sizeof(atom)*PDH_acnt, cudaMemcpyHostToDevice);

	//Define block and grid size
	int num_threads = 256; //number of threads in one dimension of a block
	int num_blocks = (PDH_acnt + num_threads - 1)/num_threads; //calculate number of blocks needed
	
	//Start counting time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch kernel
	PDH_kernel<<<num_blocks,num_threads, sizeof(bucket)*num_buckets>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res, num_buckets);
	//PDH_kernelST<<<1,1>>>(d_atom_list, d_histogram, PDH_acnt, PDH_res);

	//stop counting time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	

	//Copy data from gpu memory to host memory
	cudaMemcpy(h_histogram, d_histogram, sizeof(bucket)*num_buckets, cudaMemcpyDeviceToHost);
	
	/* Print out the histogram again for gpu version */
	output_histogram(h_histogram);

	//report running time
	printf("******** Total Running Time of Kernel = %0.5f ms *******\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(h_histogram);
	cudaFree(d_histogram);
	
	

	return 0;
}



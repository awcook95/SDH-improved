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
	double *x_pos;
	double *y_pos;
	double *z_pos;
} atom;

typedef struct hist_entry{
	unsigned long long d_cnt;   /* need a long long type as the count might be huge */

} bucket;

//Global variables
bucket * h_histogram;	/* list of all buckets in the histogram     */
long long	PDH_acnt;	/* total number of data points              */
int num_buckets;		/* total number of buckets in the histogram */
double   PDH_res;		/* value of w                               */

/* 3 arrays for x, y, and z positions of each atom                  */
double *h_x_arr;
double *h_y_arr;
double *h_z_arr;



//Device helper function: Distance of two points in the atom_list
__device__ double d_p2p_distance(double *x_arr, double *y_arr, double *z_arr, int ind1, int ind2) {
	
	double x1 = x_arr[ind1];
	double x2 = x_arr[ind2];
	double y1 = y_arr[ind1];
	double y2 = y_arr[ind2];
	double z1 = z_arr[ind1];
	double z2 = z_arr[ind2];
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}


__global__ void PDH_kernel(double *x_arr, double *y_arr, double *z_arr, bucket *d_histogram, int PDH_acnt, int PDH_res){
	int t = blockIdx.x * blockDim.x + threadIdx.x;
 
	int h_pos;
	double dist;

	for(int i = t + 1; i < PDH_acnt; i++){
		dist = d_p2p_distance(x_arr, y_arr, z_arr, t, i);
			h_pos = (int) (dist / PDH_res);
			atomicAdd(&(d_histogram[h_pos].d_cnt), 1);
			 
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
	h_x_arr = (double *)malloc(sizeof(double) * PDH_acnt); //3 arrays for atom positions
	h_y_arr = (double *)malloc(sizeof(double) * PDH_acnt);
	h_z_arr = (double *)malloc(sizeof(double) * PDH_acnt);

	
	
	//initialize histogram to zero
	memset(h_histogram, 0, sizeof(bucket)*num_buckets);

	//Allocate device memory
	bucket *d_histogram; //pointer to array of buckets
	double *d_x_arr;
	double *d_y_arr;
	double *d_z_arr;

	cudaMalloc((void**)&d_histogram, sizeof(bucket)*num_buckets);
	cudaMalloc((void**)&d_x_arr, sizeof(double)*PDH_acnt);
	cudaMalloc((void**)&d_y_arr, sizeof(double)*PDH_acnt);
	cudaMalloc((void**)&d_z_arr, sizeof(double)*PDH_acnt);

	
	srand(1);
	/* Generate data following a uniform distribution */
	for(i = 0;  i < PDH_acnt; i++) {
		h_x_arr[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		h_y_arr[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
		h_z_arr[i] = ((double)(rand()) / RAND_MAX) * BOX_SIZE;
	}

	//Copy host data to device memory
	cudaMemcpy(d_histogram, h_histogram, sizeof(bucket)*num_buckets, cudaMemcpyHostToDevice);
	cudaMemcpy(d_x_arr, h_x_arr, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y_arr, h_y_arr, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);
	cudaMemcpy(d_z_arr, h_z_arr, sizeof(double) * PDH_acnt, cudaMemcpyHostToDevice);

	//Define block and grid size
	int num_threads = 256; //number of threads in one dimension of a block
	int num_blocks = (PDH_acnt + num_threads - 1)/num_threads; //calculate number of blocks needed
	
	//Start counting time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Launch kernel
	PDH_kernel<<<num_blocks,num_threads>>>(d_x_arr, d_y_arr, d_z_arr, d_histogram, PDH_acnt, PDH_res);
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



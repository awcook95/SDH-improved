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
__device__ double d_p2p_distance(atom atom_struct1, atom atom_struct2, int ind1, int ind2) {
	
	double x1 = atom_struct1.x_pos[ind1];
	double x2 = atom_struct2.x_pos[ind2];
	double y1 = atom_struct1.y_pos[ind1];
	double y2 = atom_struct2.y_pos[ind2];
	double z1 = atom_struct1.z_pos[ind1];
	double z2 = atom_struct2.z_pos[ind2];
		
	return sqrt((x1 - x2)*(x1-x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
}

/*Algrithm 2: Block-based computation
This algorithm uses the inter-block/intra-block method to allow use of
shared memory for 2-BS computation.
*/
__global__ void PDH_kernel(double *x_arr, double *y_arr, double *z_arr, bucket *d_histogram, int PDH_acnt, int PDH_res){

	//variables
	int B = blockDim.x; //block size
	int M = gridDim.x; //total number of blocks
	int T = blockIdx.x * blockDim.x + threadIdx.x; //thread id in grid (global)
	int t = threadIdx.x; //thread id within current block (local)
	int b = blockIdx.x; //block id

	//declare shared memory for block L
	__shared__ atom L; //struct of arrays
	__shared__ double Lx[256];
	__shared__ double Ly[256];
	__shared__ double Lz[256];

	L.x_pos = Lx; //make struct data members point to the created shared arrays
	L.y_pos = Ly;
	L.z_pos = Lz;

	//each thread loads [0-255] of the shared arrays for block L
	if(T < PDH_acnt){
	Lx[t] = x_arr[T]; //load from global thread index to local thread index
	Ly[t] = y_arr[T];
	Lz[t] = z_arr[T];
	}

	__syncthreads(); //ensure all threads are finished loading shared memory

	//declare shared memory for block R
		__shared__ atom R; //struct of arrays
		__shared__ double Rx[256];
		__shared__ double Ry[256];
		__shared__ double Rz[256];

		R.x_pos = Rx; //make struct data members point to the created shared arrays
		R.y_pos = Ry;
		R.z_pos = Rz;

	int h_pos; //position in the histogram to increment
	double dist; //stores distance computed between 2 atoms

	//nested loop for inter-block computation
	for(int i = b + 1; i < M; i++){ //b+1, because intra-block distance is a later step

		//each thread loads [0-255] of the shared arrays for block R
		T = i * blockDim.x + threadIdx.x; //need to load "i-th block" away from the current block
		if(T < PDH_acnt){
		Rx[t] = x_arr[T]; //load from global thread index to local thread index
		Ry[t] = y_arr[T];
		Rz[t] = z_arr[T];
		}

		__syncthreads();

		for(int j = 0; j < B && (i * blockDim.x + j) < PDH_acnt; j++){ //each block, 'L' computes distances between its own atoms and every other block's 'R'
			dist = d_p2p_distance(L, R, t, j);
			h_pos = (int) (dist / PDH_res);
			atomicAdd(&(d_histogram[h_pos].d_cnt), 1);
		}
		__syncthreads(); //second sync necessary because the outer loop will start loading before the inner loop is done
	}

	//loop for intra-block computation
	for(int i = t + 1; i < B && (blockIdx.x * blockDim.x + i) < PDH_acnt; i++){ //each block computes distances between its own atoms
		dist = d_p2p_distance(L, L, t, i);
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
	int num_threads = 128; //number of threads in one dimension of a block
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



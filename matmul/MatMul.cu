#include <iostream> 
#include <fstream> 
#include <vector> 
#include <cuda_runtime.h> 
#include <cmath>

//kernal function 
__global__ void matmul(float *m0, float *m1, float *mres, int r0, int c0, int r1, int c1, int rres, int cres){ 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if((row<rres) && (col<cres)){
		float P_val = 0;
		for(int k=0;k<c0;++k){
			P_val += m0[row*c0+k] * m1[k*c1+col];
		}
		mres[row*cres+col] = P_val;
	}
}

//function to read matrix values from the file 
void read_Matrix(const char* filename, std::vector<float>& matrix, int& rows, int& cols) { 
	std::ifstream file(filename);
	file >> rows >> cols; 
	matrix.resize(rows*cols);
	for(int i=0;i<rows*cols;++i){
		file >> matrix[i];
	} 
	file.close();
} 

int main(){ 
	std::vector<float> matrix0, matrix1; 
	int rows0, cols0, rows1, cols1; 
	read_Matrix("input0.raw", matrix0, rows0, cols0);
       	read_Matrix("input1.raw", matrix1, rows1, cols1);
	
	if(cols0 != rows1) {
		std::cerr << "dimensions not compatible for the multiplication" << std::endl;
		return 1;
	}
	
	int res_rows = rows0; 
	int res_cols = cols1;
	std::vector<float> result(res_rows * res_cols);

	float* d_m0;
	float* d_m1;
	float* d_res;	
	cudaMalloc(&d_m0, rows0*cols0*sizeof(float));
       	cudaMalloc(&d_m1, rows1*cols1*sizeof(float));
	cudaMalloc(&d_res, res_rows*res_cols*sizeof(float));

	cudaMemcpy(d_m0, matrix0.data(), rows0*cols0*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_m1, matrix1.data(), rows1*cols1*sizeof(float), cudaMemcpyHostToDevice);

	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid(ceil(1.0*res_cols/16), ceil(1.0*res_rows/16), 1);

	matmul<<<DimGrid, DimBlock>>>(d_m0, d_m1, d_res, rows0, cols0, rows1, cols1, res_rows, res_cols);

	cudaMemcpy(result.data(), d_res, res_rows*res_cols*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_m0);
	cudaFree(d_m1);
	cudaFree(d_res);
}
	

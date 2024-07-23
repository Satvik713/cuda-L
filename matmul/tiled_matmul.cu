#include <iostream> 
#include <fstream> 
#include <vector> 
#include <cmath>

#define TILE_WIDTH 16
__global__ void tiled_mat_mul(float *m0, float *m1, float *mres, int r0, int c0, int r1, int c1, int r_res, int c_res) {
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by*TILE_WIDTH+ty;
	int col = bx*TILE_WIDTH+tx;
	float P_val = 0;

	for(int ph=0;ph<ceil(c0/(float)TILE_WIDTH);++ph) {
		if(row<r0 && ph*TILE_WIDTH+tx<c0) {
			Mds[ty][tx] = m0[row*c0+ph*TILE_WIDTH+tx];
		}
		else {
			Mds[ty][tx] = 0.0;
		}
		if(col<c1 && ph*TILE_WIDTH+ty<r1) { 
			Nds[ty][tx] = m1[(ph*TILE_WIDTH+ty)*c1+col];
		}
		else {
			Nds[ty][tx] = 0.0;
		}
		__syncthreads();
		for(int k=0;k<TILE_WIDTH;++k) {
			P_val += Mds[ty][k]*Nds[k][tx];
		}
		__syncthreads();
	}
	if((row<r_res) && (col<c_res)) {
		mres[row*c_res+col] = P_val;
	}
}


void read_Matrix(const char* filename, std::vector<float>& matrix, int& rows, int& cols) { 
	std::ifstream file(filename); 
	file >> rows >> cols; 
	matrix.resize(rows*cols); 
	for(int i=0;i<rows*cols;++i){
		file >> matrix[i]; 
	} 
	file.close();
} 

int main() { 
	std::vector<float> m0, m1;
	int r0, c0, r1, c1;
	read_Matrix("input0.raw", m0, r0, c0);
	read_Matrix("input1.raw", m1, r1, c1);
	
	if(c0 != r1) { 
		std::cerr << "dimension mismatch" << std::endl; 
		return 1;
	} 

	int res_r = r0;
	int res_c = c1;
	std::vector<float> result(res_r * res_c);
	float* d_m0;
	float* d_m1; 
	float* d_res;
	cudaMalloc(&d_m0, r0*c0*sizeof(float));
	cudaMalloc(&d_m1, r1*c1*sizeof(float));
	cudaMalloc(&d_res, res_r*res_c*sizeof(float)); 
	cudaMemcpy(d_m0, m0.data(), r0*c0*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_m1, m1.data(), r1*c1*sizeof(float), cudaMemcpyHostToDevice);
	dim3 DimBlock(16, 16, 1);
	dim3 DimGrid(ceil(1.0*res_c/16), ceil(1.0*res_r/16), 1);
	tiled_mat_mul<<<DimGrid, DimBlock>>>(d_m0, d_m1, d_res, r0, c0, r1, c1, res_r, res_c);
	cudaMemcpy(result.data(), d_res, res_r*res_c*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_m0);
	cudaFree(d_m1);
	cudaFree(d_res);

	for(int i=0;i<res_r;++i) {
		for(int j=0;j<res_c;++j){
			std::cout << result[i*res_c+j] << " ";
		}
		std::cout << std::endl;
	}
	return 0;
}

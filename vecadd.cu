#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

__global__ void vecAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char* argv[]){
    std::string file_name1 = argv[1];
    std::string file_name2 = argv[2];

    std::ifstream file1(file_name1);
    std::ifstream file2(file_name2);

    int len1;
    int len2; 

    file1 >> len1;
    file2 >> len2;

    std::vector<float> hostInput1(len1); 
    for(int i=0;i<len1;i++){
        file1 >> hostInput1[i];
    }
    file1.close();
    std::vector<float> hostInput2(len2);
    for(int i=0;i<len2;i++){
        file2 >> hostInput2[i];
    }
    file2.close();

    float *hostOutput = (float *)malloc(len1 * sizeof(float));
    
    // now we will allocate GPU memory 
    float *deviceInput1, *deviceInput2, *deviceOutput; 
    size_t size = len1 * sizeof(float);
    cudaMalloc((void**)&deviceInput1, size);
    cudaMalloc((void**)&deviceInput2, size);
    cudaMalloc((void**)&deviceOutput, size);

    // now copy data from host to gpu 
    cudaMemcpy(deviceInput1, hostInput1.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2.data(), size, cudaMemcpyHostToDevice);

    // initialization of grid and block dim 
    dim3 DimGrid(ceil(len1 / 256.0), 1, 1);
    dim3 DimBlock(256, 1, 1);

    vecAdd<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, len1); // calling the kernel 

    cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);

    // after copying the output data from device to host memory we clear the data from the cuda memory 

    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    std::cout << len1 << "\n";
    for (int i = 0; i < len1; ++i) {
        std::cout << hostOutput[i] << "\n";
    }

    free(hostOutput);

    return 0;
}   

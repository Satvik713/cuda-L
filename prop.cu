#include <iostream> 
#include <cuda_runtime.h> 

int main(){ 
	int dev_count;
	cudaGetDeviceCount(&dev_count);

	if(dev_count==0){ 
		std::cout <<"No cuda devices found!" << std::endl;
		return 1;
	} 

	cudaDeviceProp dev_prop;
	for(int i=0;i<dev_count;i++){
		cudaGetDeviceProperties(&dev_prop, i);
		std::cout << "device" << i << ":" << dev_prop.name << std::endl;
		std::cout << "total global memory:" << dev_prop.totalGlobalMem << "bytes" << std::endl;
		std::cout << "shared memory per  block:" << dev_prop.sharedMemPerBlock << "bytes" << std::endl;
		std::cout << "registers per block:" << dev_prop.regsPerBlock << std::endl;
		std::cout << "Warp size:" << dev_prop.warpSize << std::endl;
		std::cout << "max threads per block:" << dev_prop.maxThreadsPerBlock << std::endl;
		std::cout << "max threads dim: {" << dev_prop.maxThreadsDim[0] << "," 
						<< dev_prop.maxThreadsDim[1] << "," 
						<< dev_prop.maxThreadsDim[2] << "}" << std::endl; 
		std::cout << " max grid size: {" << dev_prop.maxGridSize[0] << "," 
						<< dev_prop.maxGridSize[1] << "," 
						<< dev_prop.maxGridSize[2] << "}" << std::endl;	
	}
	return 0; 

}

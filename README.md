## Learning Cuda

matmul folder contains three files- cpu_matmul.cpp, MatMul.cu, tiled_matmul.cu

As the name suggests, cpu_matmul.cpp is the cpu implementation of matrix multiplication. 
MatMul.cu is a simple cuda kernel for matrix multiplication whereas tiled_matmul.cu is tiled version of the MatMul.cu 

+------------------------------+<br>
| cpu_matmul.cpp&nbsp;&nbsp;-2700&nbsp;us&nbsp;&nbsp;&nbsp;|<br>
| MatMul.cu&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-10.30&nbsp;us&nbsp;&nbsp;&nbsp;|<br>
| tiled_matmul.cu&nbsp;&nbsp;&nbsp;-9.18&nbsp;&nbsp;us&nbsp;&nbsp;&nbsp;|<br>
+------------------------------+<br>


The difference between the runtime for MatMul.cu and tiled_matmul.cu is much less than what is theoretically expected. 

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>  

void readMatrix(const char* filename, std::vector<float>& matrix, int& rows, int& cols) {
    std::ifstream file(filename);

    file >> rows >> cols;
    matrix.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        file >> matrix[i];
    }
    file.close();
}

void matrixMultiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; ++i) {
        for (int j = 0; j < B_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
}

int main() {
    std::vector<float> matrix0, matrix1;
    int rows0, cols0, rows1, cols1;
    readMatrix("input0.raw", matrix0, rows0, cols0);
    readMatrix("input1.raw", matrix1, rows1, cols1);

    int result_rows = rows0;
    int result_cols = cols1;
    std::vector<float> result(result_rows * result_cols);
    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiply(matrix0, matrix1, result, rows0, cols0, cols1);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Time taken for matrix multiplication: " << elapsed.count() << " seconds" << std::endl;
}

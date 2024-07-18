#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

std::vector<float> read_raw_file(const std::string& file_name) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_name);
    }

    int num_values;
    file >> num_values;

    std::vector<float> values(num_values);
    for (int i = 0; i < num_values; ++i) {
        file >> values[i];
    }

    file.close();
    return values;
}

void write_raw_file(const std::string& file_name, const std::vector<float>& values) {
    std::ofstream file(file_name);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + file_name);
    }

    file << values.size() << "\n";
    for (const float& value : values) {
        file << value << "\n";
    }

    file.close();
}

std::vector<float> add_vectors(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        throw std::invalid_argument("Input vectors must have the same length");
    }

    std::vector<float> result(vec1.size());
    for (size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] + vec2[i];
    }

    return result;
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 3) {
            std::cerr << "Usage: " << argv[0] << " <input_file1> <input_file2>" << std::endl;
            return 1;
        }

        std::string file1 = argv[1];
        std::string file2 = argv[2];

        std::vector<float> vec1 = read_raw_file(file1);
        std::vector<float> vec2 = read_raw_file(file2);

        std::vector<float> result = add_vectors(vec1, vec2);

        std::string output_file = "output.raw";
        write_raw_file(output_file, result);

        std::cout << "Result written to " << output_file << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}

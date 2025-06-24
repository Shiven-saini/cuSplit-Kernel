#include <iostream>
#include <string>
#include "../headers/image_processor.h"

int main(int argc, char* argv[]) {
    std::string inputDir = "data";
    std::string outputDir = "processed";

    if (argc >= 2) {
        inputDir = argv[1];
    }
    if (argc >= 3) {
        outputDir = argv[2];
    }

    std::cout << "Input Directory: " << inputDir << std::endl;
    std::cout << "Output Directory: " << outputDir << std::endl;

    ImageProcessor processor;
    processor.processImages(inputDir, outputDir);

    return 0;
}

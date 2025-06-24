#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>

class ImageProcessor {
public:
    ImageProcessor();
    ~ImageProcessor();

    // Process all images in inputDir and write separated channels to outputDir
    void processImages(const std::string& inputDir, const std::string& outputDir);

private:
    // Process a single image file
    void processSingleImage(const std::string& filePath, const std::string& outputDir);
};

#endif // IMAGE_PROCESSOR_H

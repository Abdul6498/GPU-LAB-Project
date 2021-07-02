#include <iostream>
#include <fstream>
#include <Core/Image.hpp>
#include "cl_disparity.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s left right out_sad out_ncc\n", argv[0]);
        exit(1);
    }
    std::string input_l_path = argv[1];
    std::string input_r_path = argv[2];
    std::string output_sad_path = argv[3];
    std::string output_ncc_path = argv[4];

    std::vector<float> input_l;
    std::vector<float> input_r;
    size_t width, height;
    Core::readImagePGM(input_l_path, input_l, width, height);
    Core::readImagePGM(input_r_path, input_r, width, height);

    std::vector<short> output;
    output.resize(width * height);
    std::vector<float> output_img;
    output_img.resize(width * height);

    short max_disparity = 48;
    short window_size = 10;

    CLDisparityPerfData perf;

    auto disparity = CLDisparity();
    disparity.compute(input_l.data(), input_r.data(), output.data(), width, height, CLDisparityTypeSAD, max_disparity, window_size, &perf);

    std::cout << "Took " << perf.compute << " (" << perf.overhead << " overhead)" << std::endl;

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) max_disparity;
    Core::writeImagePGM(output_sad_path, output_img, width, height);

    disparity.compute(input_l.data(), input_r.data(), output.data(), width, height, CLDisparityTypeNCC, max_disparity, window_size, &perf);

    std::cout << "Took " << perf.compute << " (" << perf.overhead << " overhead)" << std::endl;

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) max_disparity;
    Core::writeImagePGM(output_ncc_path, output_img, width, height);
}

#include <iostream>
#include <Core/Image.hpp>
#include "cl_disparity.h"
#include "cpu_disparity.h"

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
    short window_size = 3;
    short denoise_radius = 2;

    cpu_disparity();

    CLDisparityPerfData perf;

    auto disparity = CLDisparity();
    disparity.compute({
        input_l.data(),
        input_r.data(),
        output.data(),
        width,
        height,
        CLDisparityTypeSAD,
        max_disparity,
        window_size,
        false,
        denoise_radius,
        &perf
    });

    std::cout << "upload " << perf.upload
        << " equalize " << perf.equalize
        << " disparity " << perf.disparity
        << " denoise " << perf.denoise
        << " download " << perf.download
        << " -- total " << (perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download)
        << std::endl;

    for (auto i = 0; i < output.size(); i++) output_img[i] = input_l[i];
    Core::writeImagePGM(output_sad_path + "_DBG_hist.pgm", output_img, width, height);

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) max_disparity;
    Core::writeImagePGM(output_sad_path, output_img, width, height);

    disparity.compute({
        input_l.data(),
        input_r.data(),
        output.data(),
        width,
        height,
        CLDisparityTypeNCC,
        max_disparity,
        window_size,
        false,
        denoise_radius,
        &perf
    });

    std::cout << "upload " << perf.upload
              << " equalize " << perf.equalize
              << " disparity " << perf.disparity
              << " denoise " << perf.denoise
              << " download " << perf.download
              << " -- total " << (perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download)
              << std::endl;


    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) max_disparity;
    Core::writeImagePGM(output_ncc_path, output_img, width, height);
}

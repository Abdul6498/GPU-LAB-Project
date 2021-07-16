#include <iostream>
#include <Core/Image.hpp>
#include "cl_disparity.h"
#ifdef __UNIX__
#include <pthread.h>
#endif
#include "vec_operation.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path to folder with image files>" << std::endl;
        exit(1);
    }

    std::string input_path = argv[1];
    int idx = -1;	//golobal index for total number of files is directory
    int idx_L = -1;	//Get user index for left image
    int idx_R = -1;	//Get user index for Right image

    stereo stereo;	// Class for stereo functions
    stereo.window = 3;	//window size
    stereo.dmax = 50;	//Max diparity
    auto median_radius = 2; // median filter radius (gpu has max value 4)
    auto median_size = median_radius + 1 + median_radius;
    std::vector < std::string > filename;	//Variable for file name

    //std::string path = "../../../../Opencl-ex1/images/";	//Path for image directory ends with/
    idx = stereo.load_files(input_path, filename);	//Load files

    line("Select Left image index: ");
    GetValue(idx_L);
    line("Select Right image index: ");
    GetValue(idx_R);

    if (idx_L > idx || idx_R > idx)
    {
        std::cerr << "Error! Input indexes are out of Range \n";
        return -1;
    }
    else if (idx_L == idx_R)
    {
        std::cerr << "Warning! You are comparing the same image \n";
    }

    filter filter;

    std::vector<float> inputData_L, inputData_R;	//Variable for image data
    std::size_t inputWidth, inputHeight;	//Variables for width and height of image

    Core::readImagePGM(filename[idx_L], inputData_L, inputWidth, inputHeight);	//Read image dat from the specific path
    Core::readImagePGM(filename[idx_R], inputData_R, inputWidth, inputHeight);

    std::size_t countX = inputWidth;
    std::size_t countY = inputHeight;

    static std::vector<float> imageL(inputData_L.size());	//Left image
    static std::vector<float> imageR(inputData_L.size());	//Right image
    static std::vector<float> image_eq_l(inputData_L.size());	// Histogram-equalized left image
    static std::vector<float> image_eq_r(inputData_L.size());	// Histogram-equalized right image
    static std::vector<float> image_sad(inputData_L.size());	// disparity map (SAD)
    static std::vector<float> image_ncc(inputData_L.size());	// disparity map (NCC)
    static std::vector<float> imageM_sad(inputData_L.size());	// median-filtered disparity (SAD)
    static std::vector<float> imageM_ncc(inputData_L.size());	// median-filtered disparity (NCC)

    std::string cpu_med_sad_out_path = input_path + "/Median_SAD_out_cpu.pgm";
    std::string cpu_med_ncc_out_path = input_path + "/Median_NCC_out_cpu.pgm";
    std::string cpu_sad_out_path = input_path + "/SAD_out_cpu.pgm";
    std::string cpu_ncc_out_path = input_path + "/NCC_out_cpu.pgm";
    std::string gpu_med_sad_out_path = input_path + "/Median_SAD_out_gpu.pgm";
    std::string gpu_med_ncc_out_path = input_path + "/Median_NCC_out_gpu.pgm";
    std::string gpu_sad_out_path = input_path + "/SAD_out_gpu.pgm";
    std::string gpu_ncc_out_path = input_path + "/NCC_out_gpu.pgm";

    std::vector<std::string> performance;

    auto hist_start = std::chrono::high_resolution_clock::now();
    std::thread t1(&filter::histogram, &filter, std::ref(inputData_L), std::ref(image_eq_l));
    std::thread t2(&filter::histogram, &filter, std::ref(inputData_R), std::ref(image_eq_r));
    t1.join();
    t2.join();
    auto hist_end = std::chrono::high_resolution_clock::now();
    auto hist_time = std::chrono::duration_cast<std::chrono::milliseconds>(hist_end - hist_start).count();

    std::thread t3(&stereo::load_images, &stereo, std::ref(image_eq_l), std::ref(imageL), inputWidth, inputHeight);
    std::thread t4(&stereo::load_images, &stereo, std::ref(image_eq_r), std::ref(imageR), inputWidth, inputHeight);
    t3.join();
    t4.join();

    auto sad_start = std::chrono::high_resolution_clock::now();
    stereo.SAD_disparity<std::vector<float>>(imageL, imageR, image_sad, countX, countY);	//SAD Disparity Calculation
    auto sad_end = std::chrono::high_resolution_clock::now();
    auto sad_time = std::chrono::duration_cast<std::chrono::milliseconds>(sad_end - sad_start).count();

    std::thread t5(&stereo::NCC_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_ncc), countX, countY);

    std::thread t6(&filter::median_fltr, &filter, std::ref(image_sad), std::ref(imageM_sad), median_size, countX, countY);

    auto ncc_start = std::chrono::high_resolution_clock::now();
    t5.join();
    auto ncc_end = std::chrono::high_resolution_clock::now();
    auto ncc_time = std::chrono::duration_cast<std::chrono::milliseconds>(ncc_end - ncc_start).count();

    std::thread t7(&filter::median_fltr, &filter, std::ref(image_ncc), std::ref(imageM_ncc), median_size, countX, countY);

    auto med_start = std::chrono::high_resolution_clock::now();
    t6.join();
    t7.join();
    auto med_end = std::chrono::high_resolution_clock::now();
    auto med_time = std::chrono::duration_cast<std::chrono::milliseconds>(med_end - med_start).count();

    Core::writeImagePGM(cpu_sad_out_path, image_sad, countX, countY);
    Core::writeImagePGM(cpu_ncc_out_path, image_ncc, countX, countY);
    Core::writeImagePGM(cpu_med_sad_out_path, imageM_sad, countX, countY);
    Core::writeImagePGM(cpu_med_ncc_out_path, imageM_ncc, countX, countY);

    std::vector<short> output;
    output.resize(countX * countY);
    std::vector<float> output_img;
    output_img.resize(countX * countY);

    CLDisparityPerfData perf;

    auto disparity = CLDisparity();
    CLDisparityComputeInput dci = {
        imageL.data(),
        imageR.data(),
        output.data(),
        countX,
        countY,
        CLDisparityTypeSAD,
        (short) stereo.dmax,
        (short) stereo.window,
        false, // broken :(
        (short) median_radius,
        &perf
    };
    disparity.compute(dci);

    std::cout << "upload " << perf.upload
        << " equalize " << perf.equalize
        << " disparity " << perf.disparity
        << " denoise " << perf.denoise
        << " download " << perf.download
        << " -- total " << (perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download)
        << std::endl;
    performance.push_back("[GPU] [SAD] Upload Time: " + std::to_string(perf.upload) + "ms");
    performance.push_back("[GPU] [SAD] Equlization Execution Time: " + std::to_string(perf.equalize) + "ms");
    performance.push_back("[GPU] [SAD] Disparity Execution Time: " + std::to_string(perf.disparity) + "ms");
    performance.push_back("[GPU] [SAD] Denosing Time: " + std::to_string(perf.denoise) + "ms");
    performance.push_back("[GPU] [SAD] Download Time: " + std::to_string(perf.download) + "ms");
    performance.push_back("[GPU] [SAD] Total Time: " + std::to_string(perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download) + "ms");

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) stereo.dmax;
    Core::writeImagePGM(gpu_med_sad_out_path, output_img, countX, countY);

    dci.type = CLDisparityTypeNCC;
    disparity.compute(dci);

    std::cout << "upload " << perf.upload
              << " equalize " << perf.equalize
              << " disparity " << perf.disparity
              << " denoise " << perf.denoise
              << " download " << perf.download
              << " -- total " << (perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download)
              << std::endl;

    performance.push_back("[GPU] [NCC] Upload Time: " + std::to_string(perf.upload) + "ms");
    performance.push_back("[GPU] [NCC] Equlization Execution Time: " + std::to_string(perf.equalize) + "ms");
    performance.push_back("[GPU] [NCC] Disparity Execution Time: " + std::to_string(perf.disparity) + "ms");
    performance.push_back("[GPU] [NCC] Denosing Time: " + std::to_string(perf.denoise) + "ms");
    performance.push_back("[GPU] [NCC] Download Time: " + std::to_string(perf.download) + "ms");
    performance.push_back("[GPU] [NCC] Total Time: " + std::to_string(perf.upload + perf.equalize + perf.disparity + perf.denoise + perf.download) + "ms");

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) stereo.dmax;
    Core::writeImagePGM(gpu_med_ncc_out_path, output_img, countX, countY);

    // TODO: print performance info
    performance.push_back("[CPU] Histogram Execution Time: " + std::to_string(hist_time) + "ms");
    performance.push_back("[CPU] SAD Execution Time: " + std::to_string(sad_time) + "ms");
    performance.push_back("[CPU] NCC Execution Time: " + std::to_string(ncc_time) + "ms");
    performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");

    vec::print_performance(performance);
}

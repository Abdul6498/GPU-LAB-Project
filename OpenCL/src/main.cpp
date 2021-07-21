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

    PerfTime hist_l, hist_r;
    std::thread t1(&filter::histogram, &filter, std::ref(inputData_L), std::ref(image_eq_l), &hist_l);
    std::thread t2(&filter::histogram, &filter, std::ref(inputData_R), std::ref(image_eq_r), &hist_r);
    t1.join();
    t2.join();
    auto hist_time = hist_l + hist_r;

    std::thread t3(&stereo::load_images, &stereo, std::ref(image_eq_l), std::ref(imageL), inputWidth, inputHeight);
    std::thread t4(&stereo::load_images, &stereo, std::ref(image_eq_r), std::ref(imageR), inputWidth, inputHeight);
    t3.join();
    t4.join();

    PerfTime sad_time, ncc_time;
    std::thread t_sad(&stereo::SAD_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_sad), countX, countY, &sad_time);
    std::thread t_ncc(&stereo::NCC_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_ncc), countX, countY, &ncc_time);
    t_sad.join();
    t_ncc.join();

    PerfTime sad_med_time, ncc_med_time;
    std::thread t_msad(&filter::median_fltr, &filter, std::ref(image_sad), std::ref(imageM_sad), median_size, countX, countY, &sad_med_time);
    std::thread t_mncc(&filter::median_fltr, &filter, std::ref(image_ncc), std::ref(imageM_ncc), median_size, countX, countY, &ncc_med_time);
    t_msad.join();
    t_mncc.join();
    auto med_time = sad_med_time + ncc_med_time;

    Core::writeImagePGM(cpu_sad_out_path, image_sad, countX, countY);
    Core::writeImagePGM(cpu_ncc_out_path, image_ncc, countX, countY);
    Core::writeImagePGM(cpu_med_sad_out_path, imageM_sad, countX, countY);
    Core::writeImagePGM(cpu_med_ncc_out_path, imageM_ncc, countX, countY);

    performance.push_back("[CPU] Histogram Execution Time: " + std::to_string(hist_time) + "ms");
    performance.push_back("[CPU] SAD Execution Time: " + std::to_string(sad_time) + "ms");
    performance.push_back("[CPU] NCC Execution Time: " + std::to_string(ncc_time) + "ms");
    performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");

    performance.push_back(" --- ");

    std::vector<short> output;
    output.resize(countX * countY);
    std::vector<float> output_img;
    output_img.resize(countX * countY);

    CLDisparityPerfData perf_sad, perf_ncc;

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
        // GPU equalization is broken :(
        // - so we just use the equalized image from the cpu
        false,
        (short) median_radius,
        &perf_sad
    };
    disparity.compute(dci);

    performance.push_back("[GPU] [SAD] Upload Time: " + perf_sad.upload.toString(true));
    performance.push_back("[GPU] [SAD] Equalization Execution Time: " + perf_sad.equalize.toString(true));
    performance.push_back("[GPU] [SAD] Disparity Execution Time: " + perf_sad.disparity.toString(true));
    performance.push_back("[GPU] [SAD] Denoising Time: " + perf_sad.denoise.toString(true));
    performance.push_back("[GPU] [SAD] Download Time: " + perf_sad.download.toString(true));
    auto time_gpu_total_sad = perf_sad.upload + perf_sad.equalize + perf_sad.disparity + perf_sad.denoise + perf_sad.download;
    performance.push_back("[GPU] [SAD] Total Time: " + time_gpu_total_sad.toString(true));

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) stereo.dmax;
    Core::writeImagePGM(gpu_med_sad_out_path, output_img, countX, countY);

    dci.type = CLDisparityTypeNCC;
    dci.perf_out = &perf_ncc;
    disparity.compute(dci);

    performance.push_back("[GPU] [NCC] Upload Time: " + perf_ncc.upload.toString(true));
    performance.push_back("[GPU] [NCC] Equalization Execution Time: " + perf_ncc.equalize.toString(true));
    performance.push_back("[GPU] [NCC] Disparity Execution Time: " + perf_ncc.disparity.toString(true));
    performance.push_back("[GPU] [NCC] Denoising Time: " + perf_ncc.denoise.toString(true));
    performance.push_back("[GPU] [NCC] Download Time: " + perf_ncc.download.toString(true));
    auto time_gpu_total_ncc = perf_ncc.upload + perf_ncc.equalize + perf_ncc.disparity + perf_ncc.denoise + perf_ncc.download;
    performance.push_back("[GPU] [NCC] Total Time: " + time_gpu_total_ncc.toString(true));

    for (auto i = 0; i < output.size(); i++) output_img[i] = ((float) output[i]) / (float) stereo.dmax;
    Core::writeImagePGM(gpu_med_ncc_out_path, output_img, countX, countY);

    performance.push_back(" --- ");

    // because the GPU doesn't perform histogram equalization, we won't consider it for this total
    auto sad_time_total = sad_time + sad_med_time;
    auto ncc_time_total = ncc_time + ncc_med_time;

    auto speed_fac_sad = double(sad_time) / perf_sad.disparity.getMilliseconds();
    auto speed_fac_ncc = double(ncc_time) / perf_ncc.disparity.getMilliseconds();
    auto speed_fac_o_sad = double(sad_time_total) / time_gpu_total_sad.getMilliseconds();
    auto speed_fac_o_ncc = double(ncc_time_total) / time_gpu_total_ncc.getMilliseconds();
    performance.push_back("[SAD] GPU speedup: " + std::to_string(speed_fac_sad));
    performance.push_back("[SAD] GPU speedup (+overhead): " + std::to_string(speed_fac_o_sad));
    performance.push_back("[NCC] GPU speedup: " + std::to_string(speed_fac_ncc));
    performance.push_back("[NCC] GPU speedup (+overhead): " + std::to_string(speed_fac_o_ncc));

    vec::print_performance(performance);
}

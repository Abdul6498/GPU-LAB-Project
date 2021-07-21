#ifndef CL_DISPARITY_H
#define CL_DISPARITY_H

#include <iostream>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

enum CLDisparityType {
    CLDisparityTypeSAD,
    CLDisparityTypeNCC
};

typedef struct CLDisparityPerfData {
    Core::TimeSpan upload;
    Core::TimeSpan equalize;
    Core::TimeSpan disparity;
    Core::TimeSpan denoise;
    Core::TimeSpan download;
} CLDisparityPerfData;

typedef struct CLDisparityComputeInput {
    float* input_l;
    float* input_r;
    short* output;
    size_t width, height;
    CLDisparityType type;
    short max_disparity;
    short window_size;
    bool equalize_input;
    short denoise_radius;
    CLDisparityPerfData* perf_out;
} CLDisparityComputeInput;

class CLDisparity {
private:
    cl::Context ctx;
    cl::Device device;
    cl::Kernel kern_sad;
    cl::Kernel kern_ncc;
    cl::Kernel kern_hist_chunk;
    cl::Kernel kern_hist_comb;
    cl::Kernel kern_hist_eq;
    cl::Kernel kern_med;

public:
    CLDisparity();

    /**
     * Computes disparity
     * @param input_l left input image
     * @param input_r right input image
     * @param output output buffer of the same size as the input images
     * @param width image width
     * @param height image height
     * @param type disparity method
     * @param max_disparity
     * @param window_size
     * @param perf_data if not null, will output performance data
     */
    void compute(CLDisparityComputeInput i);
};

#endif //CL_DISPARITY_H

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
    Core::TimeSpan compute;
    Core::TimeSpan overhead;
} CLDisparityPerfData;

class CLDisparity {
private:
    cl::Context ctx;
    cl::Device device;
    cl::Kernel kern_sad;
    cl::Kernel kern_ncc;

public:
    CLDisparity();

    void compute(
        float* input_l,
        float* input_r,
        short* output,
        size_t width,
        size_t height,
        CLDisparityType type,
        short max_disparity,
        short window_size,
        CLDisparityPerfData* perf_data = nullptr
    );
};

#endif //CL_DISPARITY_H

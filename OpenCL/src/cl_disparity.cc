#include "cl_disparity.h"

CLDisparity::CLDisparity() {
    ctx = cl::Context(CL_DEVICE_TYPE_GPU);
    auto available_devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
    int highestCU = 0;
    for (const auto& d : available_devices) {
        int cu = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        if (cu > highestCU) {
            device = d;
            highestCU = cu;
        }
    }
    if (!highestCU) throw "no CL devices found!";
    std::vector<cl::Device> devices;
    devices.push_back(device);
    OpenCL::printDeviceInfo(std::cerr, device);

    std::string source =
#include "gen/disparity.cl"
    ;
    std::vector<std::pair<const char*, size_t>> sources;
    sources.push_back(std::make_pair(source.data(), source.length()));
    auto prog = cl::Program(ctx, sources);
    OpenCL::buildProgram(prog, devices);

    kern_sad = cl::Kernel(prog, "disparity_sad");
    kern_ncc = cl::Kernel(prog, "disparity_ncc");
}

void CLDisparity::compute(
    float* input_l,
    float* input_r,
    short* output,
    size_t width,
    size_t height,
    CLDisparityType type,
    short max_disparity,
    short window_size,
    CLDisparityPerfData* perf_data
) {
    cl::CommandQueue queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

    auto output_size = width * height * sizeof(short);
    auto tmp_size = width * height * sizeof(float);

    auto d_input_l = cl::Image2D(ctx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), width, height);
    auto d_input_r = cl::Image2D(ctx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), width, height);
    auto d_output = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, output_size);

    cl::Event evt_overhead1;
    cl::Event evt_overhead2;
    cl::Event evt_overhead3;
    cl::Event evt_compute;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;
    queue.enqueueWriteImage(d_input_l, false, {}, region, width * sizeof(float), 0, input_l, nullptr, &evt_overhead1);
    queue.enqueueWriteImage(d_input_r, false, {}, region, width * sizeof(float), 0, input_r, nullptr, &evt_overhead2);

    if (type == CLDisparityTypeSAD) {
        kern_sad.setArg(0, d_input_l);
        kern_sad.setArg(1, d_input_r);
        kern_sad.setArg(2, d_output);
        kern_sad.setArg(3, max_disparity);
        kern_sad.setArg(4, window_size);
        queue.enqueueNDRangeKernel(kern_sad, 0, { width, height }, cl::NullRange, nullptr, &evt_compute);
    } else if (type == CLDisparityTypeNCC) {
        auto d_tmp = cl::Buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, tmp_size);

        kern_ncc.setArg(0, d_input_l);
        kern_ncc.setArg(1, d_input_r);
        kern_ncc.setArg(2, d_output);
        kern_ncc.setArg(3, max_disparity);
        kern_ncc.setArg(4, window_size);
        kern_ncc.setArg(5, d_tmp);
        queue.enqueueNDRangeKernel(kern_ncc, 0, { width, height }, cl::NullRange, nullptr, &evt_compute);
    }

    queue.enqueueReadBuffer(d_output, false, 0, output_size, output, nullptr, &evt_overhead3);
    queue.finish();

    if (perf_data != nullptr) {
        perf_data->compute = OpenCL::getElapsedTime(evt_compute);
        perf_data->overhead = OpenCL::getElapsedTime(evt_overhead1)
            + OpenCL::getElapsedTime(evt_overhead2)
            + OpenCL::getElapsedTime(evt_overhead3);
    }
}

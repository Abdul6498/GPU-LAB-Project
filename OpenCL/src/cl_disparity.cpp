#include <cmath>
#include "cl_disparity.h"

CLDisparity::CLDisparity() {
    ctx = cl::Context(CL_DEVICE_TYPE_GPU);
    auto available_devices = ctx.getInfo<CL_CONTEXT_DEVICES>();
    int highestCU = 0;
    for (const auto& d : available_devices) {
        int cu = d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
        if (cu > highestCU) {
            auto device_name = d.getInfo<CL_DEVICE_NAME>();
            if (device_name.find("AMD Radeon") != std::string::npos && device_name.find("Compute Engine") != std::string::npos) {
                std::cerr << "Ignoring AMD Radeon Compute Engine because the compute kernel crashes macOS" << std::endl;
                continue;
            }
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
    kern_hist_chunk = cl::Kernel(prog, "histogram_chunks");
    kern_hist_comb = cl::Kernel(prog, "histogram_combine");
    kern_hist_eq = cl::Kernel(prog, "histogram_equalize");
    kern_med = cl::Kernel(prog, "median_filter");
}

void CLDisparity::compute(CLDisparityComputeInput i) {
    cl::CommandQueue queue(ctx, device, CL_QUEUE_PROFILING_ENABLE);

    auto output_size = i.width * i.height * sizeof(short);
    auto tmp_size = i.width * i.height * sizeof(float);

    auto d_input_l = cl::Image2D(ctx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), i.width, i.height);
    auto d_input_r = cl::Image2D(ctx, CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), i.width, i.height);
    auto d_output = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE, output_size);

    cl::Event evt_upload_l, evt_upload_r;
    cl::Event evt_hist_cl, evt_hist_cr, evt_hist_hl, evt_hist_hr, evt_hist_el, evt_hist_er;
    cl::Event evt_disparity;
    cl::Event evt_median;
    cl::Event evt_download;

    cl::size_t<3> region;
    region[0] = i.width;
    region[1] = i.height;
    region[2] = 1;
    queue.enqueueWriteImage(d_input_l, false, {}, region, i.width * sizeof(float), 0, i.input_l, nullptr, &evt_upload_l);
    queue.enqueueWriteImage(d_input_r, false, {}, region, i.width * sizeof(float), 0, i.input_r, nullptr, &evt_upload_r);

    if (i.equalize_input) {
        size_t hist_chunk_size = 16;
        size_t chunk_count_x = ceil(float(i.width) / float(hist_chunk_size));
        size_t chunk_count_y = ceil(float(i.width) / float(hist_chunk_size));
        size_t hist_chunks_size = chunk_count_x * chunk_count_y * sizeof(ushort);
        size_t hist_size = 256 * sizeof(float);
        auto d_hist_chunks_l = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE, hist_chunks_size);
        auto d_hist_chunks_r = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE, hist_chunks_size);
        auto d_cdf_l = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE, hist_size);
        auto d_cdf_r = cl::Buffer(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_READ_WRITE, hist_size);

        kern_hist_chunk.setArg(0, d_input_l);
        kern_hist_chunk.setArg(1, d_hist_chunks_l);
        kern_hist_chunk.setArg(2, i.width);
        kern_hist_chunk.setArg(3, i.height);
        queue.enqueueNDRangeKernel(kern_hist_chunk, 0, { chunk_count_x, chunk_count_y }, cl::NullRange, nullptr, &evt_hist_cl);

        kern_hist_chunk.setArg(0, d_input_r);
        kern_hist_chunk.setArg(1, d_hist_chunks_r);
        kern_hist_chunk.setArg(2, i.width);
        kern_hist_chunk.setArg(3, i.height);
        queue.enqueueNDRangeKernel(kern_hist_chunk, 0, { chunk_count_x, chunk_count_y }, cl::NullRange, nullptr, &evt_hist_cr);

        kern_hist_comb.setArg(0, d_hist_chunks_l);
        kern_hist_comb.setArg(1, d_cdf_l);
        kern_hist_comb.setArg(2, chunk_count_x * chunk_count_y);
        kern_hist_comb.setArg(3, i.width * i.height);
        queue.enqueueNDRangeKernel(kern_hist_comb, 0, { 256 }, { 256 }, nullptr, &evt_hist_hl);

        kern_hist_comb.setArg(0, d_hist_chunks_r);
        kern_hist_comb.setArg(1, d_cdf_r);
        kern_hist_comb.setArg(2, chunk_count_x * chunk_count_y);
        kern_hist_comb.setArg(3, i.width * i.height);
        queue.enqueueNDRangeKernel(kern_hist_comb, 0, { 256 }, { 256 }, nullptr, &evt_hist_hr);

        kern_hist_eq.setArg(0, d_input_l);
        kern_hist_eq.setArg(1, d_cdf_l);
        kern_hist_eq.setArg(2, d_input_l);
        queue.enqueueNDRangeKernel(kern_hist_eq, 0, { i.width, i.height }, cl::NullRange, nullptr, &evt_hist_el);

        kern_hist_eq.setArg(0, d_input_r);
        kern_hist_eq.setArg(1, d_cdf_r);
        kern_hist_eq.setArg(2, d_input_r);
        queue.enqueueNDRangeKernel(kern_hist_eq, 0, { i.width, i.height }, cl::NullRange, nullptr, &evt_hist_er);

        auto dbg_img = cl::Image2D(ctx, CL_MEM_HOST_READ_ONLY | CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_R, CL_FLOAT), i.width, i.height);
        queue.enqueueCopyImage(d_input_l, dbg_img, {}, {}, region);
        queue.enqueueReadImage(dbg_img, false, {}, region, i.width * sizeof(float), 0, i.input_l);
        queue.finish();
    }

    if (i.type == CLDisparityTypeSAD) {
        kern_sad.setArg(0, d_input_l);
        kern_sad.setArg(1, d_input_r);
        kern_sad.setArg(2, d_output);
        kern_sad.setArg(3, i.max_disparity);
        kern_sad.setArg(4, i.window_size);
        queue.enqueueNDRangeKernel(kern_sad, 0, { i.width, i.height }, cl::NullRange, nullptr, &evt_disparity);
    } else if (i.type == CLDisparityTypeNCC) {
        auto d_tmp = cl::Buffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, tmp_size);

        kern_ncc.setArg(0, d_input_l);
        kern_ncc.setArg(1, d_input_r);
        kern_ncc.setArg(2, d_output);
        kern_ncc.setArg(3, i.max_disparity);
        kern_ncc.setArg(4, i.window_size);
        kern_ncc.setArg(5, d_tmp);
        queue.enqueueNDRangeKernel(kern_ncc, 0, { i.width, i.height }, cl::NullRange, nullptr, &evt_disparity);
    }

    if (i.denoise_radius) {
        kern_med.setArg(0, d_output);
        kern_med.setArg(1, d_output);
        kern_med.setArg(2, i.width);
        kern_med.setArg(3, std::min(4, (int) i.denoise_radius));
        queue.enqueueNDRangeKernel(kern_med, 0, { i.width, i.height }, cl::NullRange, nullptr, &evt_median);
    }

    queue.enqueueReadBuffer(d_output, false, 0, output_size, i.output, nullptr, &evt_download);
    queue.finish();

    if (i.perf_out != nullptr) {
        i.perf_out->upload = OpenCL::getElapsedTime(evt_upload_l)
            + OpenCL::getElapsedTime(evt_upload_r);
        if (i.equalize_input) {
            i.perf_out->equalize = OpenCL::getElapsedTime(evt_hist_cl)
                                   + OpenCL::getElapsedTime(evt_hist_hl) + OpenCL::getElapsedTime(evt_hist_el)
                                   + OpenCL::getElapsedTime(evt_hist_cr) + OpenCL::getElapsedTime(evt_hist_hr)
                                   + OpenCL::getElapsedTime(evt_hist_er);
        }
        i.perf_out->disparity = OpenCL::getElapsedTime(evt_disparity);
        if (i.denoise_radius) {
            i.perf_out->denoise = OpenCL::getElapsedTime(evt_median);
        }
        i.perf_out->download = OpenCL::getElapsedTime(evt_download);
    }
}

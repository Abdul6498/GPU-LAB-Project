#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp>
#endif

__kernel void disparity_sad(
    __read_only image2d_t input_l,
    __read_only image2d_t input_r,
    __write_only __global short* output,
    __read_only short sample_offset,
    __read_only short window_size
) {
    size_t count_x = get_global_size(0);
    int x = get_global_id(0);
    int y = get_global_id(1);

    short disparity;
    float min_diff = INFINITY;

    int cx = x - window_size / 2;
    int cy = y - window_size / 2;

    for (short dx = -1; dx >= -sample_offset; dx--) {
        int window_x = cx + dx;
        float diff = 0.;

        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, (int2) { window_x + ddx, cy + ddy }).r;
                diff += fabs(r - l);
            }
        }

        if (diff < min_diff) {
            disparity = -dx;
            min_diff = diff;
        }
    }

    output[count_x * y + x] = disparity;
}

__kernel void disparity_ncc(
    __read_only image2d_t input_l,
    __read_only image2d_t input_r,
    __write_only __global short* output,
    __read_only short sample_offset,
    __read_only short window_size,
    __read_write __global float* mean_img
) {
    size_t count_x = get_global_size(0);
    int x = get_global_id(0);
    int y = get_global_id(1);

    short disparity = 0;
    float max_corr = -INFINITY;
    float n = window_size * window_size;

    int cx = x - window_size / 2;
    int cy = y - window_size / 2;

    float mean_l;
    {
        float sum_l = 0.;
        float sum_r = 0.;
        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, (int2) { cx + ddx, cy + ddy }).r;
                sum_l += l;
                sum_r += r;
            }
        }
        mean_l = sum_l / n;
        float mean_r = sum_r / n;

        mean_img[count_x * y + x] = mean_r;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    for (short dx = -1; dx >= -sample_offset; dx--) {
        int window_x = cx + dx;

        float mean_r = mean_img[count_x * y + (x + dx)];

        float cov_sum = 0.;
        float var_sum_l = 0.;
        float var_sum_r = 0.;

        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, (int2) { window_x + ddx, cy + ddy }).r;

                cov_sum += (l - mean_l) * (r - mean_r);
                var_sum_l += (l - mean_l) * (l - mean_l);
                var_sum_r += (r - mean_r) * (r - mean_r);
            }
        }

        float r = cov_sum / sqrt(var_sum_l * var_sum_r);

        if (r > max_corr) {
            disparity = -dx;
            max_corr = r;
        }
    }

    output[count_x * y + x] = disparity;
}

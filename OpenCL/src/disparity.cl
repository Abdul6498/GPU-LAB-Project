#ifndef __OPENCL_VERSION__
#include <OpenCL/OpenCLKernel.hpp>
#endif

const sampler_t image_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
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
    float max_possible_diff = 255 * window_size * window_size;
    float diffs[3] = { max_possible_diff, max_possible_diff, max_possible_diff }; // moving window of 3 past diffs
    float min_diff = INFINITY;

    // cx/cy is the window's top left corner
    // the window is centered on (x, y)
    int cx = x - window_size / 2;
    int cy = y - window_size / 2;

    for (short dx = 0; dx >= -sample_offset - 1; dx--) {
        // window x position
        int window_x = cx + dx;
        // absolute difference
        float diff = 0.;

        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, image_sampler, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, image_sampler, (int2) { window_x + ddx, cy + ddy }).r;
                diff += fabs(r - l);
            }
        }

        diffs[0] = diffs[1];
        diffs[1] = diffs[2];
        diffs[2] = diff;

        if (diffs[1] < min_diff) {
            short cdx = dx + 1; // diffs[1] is the previous offset!
            //disparity = -cdx;
            // subpixel localization
            disparity = -cdx - 0.5 * ((diffs[2] - diffs[0]) / (diffs[0] - (2 * diffs[1]) + diffs[2]));
            min_diff = diffs[1];
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

    float n = window_size * window_size;

    // cx/cy is the window's top left corner
    // the window is centered on (x, y)
    int cx = x - window_size / 2;
    int cy = y - window_size / 2;

    float mean_l;
    {
        // compute mean of window around current pixel
        float sum_l = 0.;
        float sum_r = 0.;
        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, image_sampler, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, image_sampler, (int2) { cx + ddx, cy + ddy }).r;
                sum_l += l;
                sum_r += r;
            }
        }
        mean_l = sum_l / n;
        float mean_r = sum_r / n;

        // ...and store it in the global mean image
        mean_img[count_x * y + x] = mean_r;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    short disparity = 0;
    float correlations[3] = { 0, 0, 0 }; // moving window of 3 past correlations
    float max_corr = -INFINITY;

    for (short dx = 0; dx >= -sample_offset - 1; dx--) {
        int window_x = cx + dx;

        float mean_r = mean_img[count_x * y + (x + dx)];

        float cov_sum = 0.;
        float var_sum_l = 0.;
        float var_sum_r = 0.;

        for (int ddy = 0; ddy < window_size; ddy++) {
            for (int ddx = 0; ddx < window_size; ddx++) {
                float l = read_imagef(input_l, image_sampler, (int2) { cx + ddx, cy + ddy }).r;
                float r = read_imagef(input_r, image_sampler, (int2) { window_x + ddx, cy + ddy }).r;

                cov_sum += (l - mean_l) * (r - mean_r);
                var_sum_l += (l - mean_l) * (l - mean_l);
                var_sum_r += (r - mean_r) * (r - mean_r);
            }
        }

        float r = cov_sum / sqrt(var_sum_l * var_sum_r);

        correlations[0] = correlations[1];
        correlations[1] = correlations[2];
        correlations[2] = r;

        if (correlations[1] > max_corr) {
            short cdx = dx + 1; // correlations[1] is the previous offset!
            // disparity = -cdx;
            // subpixel localization
            disparity = -cdx - 0.5 * ((correlations[2] - correlations[0])
                / (correlations[0] - (2 * correlations[1]) + correlations[2]));
            max_corr = correlations[1];
        }
    }

    output[count_x * y + x] = disparity;
}

// additional pre/post-processing filters

__kernel void histogram_chunks(
    __read_only image2d_t input,
    __write_only __global ushort* chunks,
    __read_only int width,
    __read_only int height
) {
    const int chunk_size = 16;
    int chunk_x = get_global_id(0);
    int chunk_y = get_global_id(1);

    int chunks_width = get_global_size(0);
    int chunk_offset = (chunk_y * chunks_width + chunk_x) * 256;

    ushort hist[256];
    for (size_t i = 0; i < 256; i++) {
        hist[i] = 0;
    }

    int start_x = chunk_x * chunk_size;
    int start_y = chunk_y * chunk_size;
    for (int y = start_y; y < min(start_y + chunk_size, height); y++) {
        for (int x = start_x; x < min(start_x + chunk_size, width); x++) {
            int hist_value = floor(read_imagef(input, image_sampler, (int2) { x, y }).r * 255.f);
            hist_value = clamp(hist_value, 0, 255);
            hist[hist_value]++;
        }
    }

    for (size_t i = 0; i < 256; i++) {
        chunks[chunk_offset + i] = hist[i];
    }
}
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void histogram_combine(
    __read_only __global ushort* chunks,
    __write_only __global float* cdf,
    __read_only size_t chunk_count,
    __read_only size_t total_pixel_count
) {
    int gray_value = get_global_id(0);

    ulong val_count = 0;
    for (size_t i = 0; i < chunk_count; i++) {
        val_count += chunks[i * 256 + gray_value];
    }

    __local double pdf[256];
    pdf[gray_value] = double(val_count) / total_pixel_count * 41;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gray_value == 0) {
        // turn it into a CDF
        // (a parallel prefix sum seems overkill for this)
        double accum = 0.;
        for (size_t i = 0; i < 256; i++) {
            accum += pdf[i];
            printf("[%d] %f, %f\n", i, pdf[i], accum);
            cdf[i] = accum;
        }
    }
}
__kernel void histogram_equalize(
    __read_only image2d_t input,
    __read_only __global float* cdf,
    __write_only image2d_t output
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    float value = read_imagef(input, (int2) { x, y }).r;
    value = cdf[(size_t) clamp((value * 255.f), 0.f, 255.f)];
    value = y < 32 ? ((x % 256) / 255.f) : cdf[x % 256]; // debug histogram output
    write_imagef(output, (int2) { x, y }, (float4) { value, 0, 0, 0 });
}

__inline void swap(short* a, short* b) {
    int t = *b;
    *b = *a;
    *a = t;
}
void sort(short* arr, size_t l, size_t h);
void sort(short* arr, size_t l, size_t h) {
    bool f = true;
    while (f) {
        f = false;
        for (size_t i = 0; i < h; i++) {
            if (arr[i] > arr[i + 1]) {
                swap(&arr[i], &arr[i + 1]);
                f = true;
            }
        }
        h -= 1;
    }
}

__kernel void median_filter(
    __read_only __global short* input,
    __read_only __global short* output,
    __read_only int width,
    __read_only int radius
) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    const int val_count = (radius + 1 + radius) * (radius + 1 + radius);
    short values[81];

    int i = 0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            float v = input[(y + dy) * width + (x + dx)];
            values[i++] = v;
        }
    }

    sort(values, 0, val_count - 1);
    output[y * width + x] = values[val_count / 2];
}

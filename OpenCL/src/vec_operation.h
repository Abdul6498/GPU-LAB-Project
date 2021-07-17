#pragma once

#include <stdio.h>

#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/Image.hpp>
#include <OpenCL/cl-patched.hpp>
#include <OpenCL/Program.hpp>
#include <OpenCL/Event.hpp>
#include <OpenCL/Device.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <utility>
#include <filesystem>
#include <array>
#include <boost/lexical_cast.hpp>
#include <thread>

#define LOG(x) std::cout << x << std::endl;
#define LOG_w(x) std::cout << x << " , ";
#define newline() std::cout << std::endl;
#define GetValue(x) std::cin >> x ;
#define line(x) std::cout << x;

#define PERF_BEGIN auto P_start_time = std::chrono::high_resolution_clock::now();
#define PERF_END(x) auto P_end_time = std::chrono::high_resolution_clock::now(); \
    if (x != nullptr) *x = std::chrono::duration_cast<std::chrono::milliseconds>(P_end_time - P_start_time).count();

typedef long long PerfTime;

namespace vec {

	/*
	* Map Values from 0 - 50 to 0 to 1
	* src: Input pair of values
	* dst: Output Pair of Values
	* val: Value to be mapped
	*/
	template<typename tVal>
	tVal map_value(std::pair<tVal, tVal> src, std::pair<tVal, tVal> dst, tVal val);

	/*
	* Get global index of in a certain range
	* countX: Width of image
	* i: Column index(x)
	* j: Row Index(y)
	*/
	int getIndexGlobal(std::size_t countX, int i, int j);
	/*
	* Get Pixel intensities for certain pixels
	* img: Input image
	* countX: Width of image
	* countY: Height of Image
	* i: Column index(x)
	* j: Row Index(y)
	*/
	float getValueGlobal(const std::vector<float>& img, std::size_t countX, std::size_t countY, int i, int j);

	/*
	* Creates Rectangle or mask of a window size (window x window)
	* in_image: Input image
	* x: Column index(x)
	* y: Row Index(y)
	* countX: Width of image
	* countY: Height of Image
	* rect: output mask/kernel/rectangle
	*/
	template<typename T>
	T Rect(T& in_image, int x, int y, size_t window, int countX, int countY, T& rect);

	/*
	* Multiply 2 vectors elment by element
	* vec1: 1st vector input
	* vec2: 2nd vector input
	* vec_out: output vector
	* window: size of one dimension of the window
	*/
	template<typename T>
	T vec_multiply(T& vec1, T& vec2, T& vec_out, size_t window);

	/*
	* Add 2 vectors elment by element
	* vec1: 1st vector input
	* vec2: 2nd vector input
	* vec_out: output vector
	* window: size of one dimension of the window
	*/
	template<typename T>

	T vec_addition(T& vec1, T& vec2, T& vec_out, size_t window);
	/*
	* Absolute subtraction of 2 vectors elment by element
	* vec1: 1st vector input
	* vec2: 2nd vector input
	* vec_out: output vector
	* window: size of one dimension of the window
	*/
	template<typename T>
	T vec_subtract(T& vec1, T& vec2, T& vec_out, size_t window);

	/*
	* Sum of all elements of vector
	* vec1: Input vector
	* window: size of one dimension of the window
	* return sum.
	*/
	template<typename T>
	float vec_sum(T& vec1, size_t window);

	/*
	* Print Performance of algorithm for CPU and GPU
	*/
	void print_performance(std::vector<std::string>& performance);
	/*
	* Multiply Vector with a scaler number
	* vec: Input Vector.
	* num: Scaler number
	*/
	std::vector<uint8_t> scaler_multiply(std::vector<float>& vec, uint8_t num);

	/*
	* Find Median of vector
	* ve: Input Vector
	*/
	float find_median(std::vector<float>& vec);

	/*
	* Clone 2 images/vectors
	* src: Input image
	* dst: Output Image
	*/
	std::vector<float> clone(std::vector<float>& src, std::vector<float>& dst);

	template<typename T>
	T Rect(T& in_image, int x, int y, size_t window, int countX, int countY, T& rect)
	{
		size_t pi = 0;
		size_t pj = 0;
		for (int i = x; i < (x + window); i++) {
			for (int j = y; j < (y + window); j++) {
				rect[pj + window * pi] = getValueGlobal(in_image, countX, countY, i, j);
				pj++;
			}
			pj = 0;
			pi++;
		}
		return rect;
	}
	template<typename T>
	T vec_multiply(T& vec1, T& vec2, T& vec_out, size_t window) {
		for (int i = 0; i < (window * window); i++)
		{
			vec_out[i] = vec1[i] * vec2[i];
		}
		return vec_out;
	}

	template<typename T>
	T vec_addition(T& vec1, T& vec2, T& vec_out, size_t window) {
		for (int i = 0; i < (window * window); i++)
		{
			vec_out[i] = vec1[i] + vec2[i];
		}
		return vec_out;
	}

	template<typename T>
	T vec_subtract(T& vec1, T& vec2, T& vec_out, size_t window) {
		for (int i = 0; i < (window * window); i++)
		{
			vec_out[i] = std::abs(vec1[i] - vec2[i]);
		}
		return vec_out;
	}

	template<typename T>
	float vec_sum(T& vec1, size_t window) {
		float sum = 0;
		for (int i = 0; i < (window * window); i++)
		{
			sum = vec1[i] + sum;
		}
		return sum;
	}
	template<typename tVal>
	tVal map_value(std::pair<tVal, tVal> src, std::pair<tVal, tVal> dst, tVal val)
	{
		return(src.first + ((dst.second - dst.first) / (src.second - src.first)) * (val - src.first));
	}


}
class filter
{
private:
	float m_median = 0;
public:
	std::vector<float> histogram(std::vector<float>& image, std::vector<float>& imageE, PerfTime* perf_out = nullptr);
	std::vector<float> median_fltr(std::vector<float>& image, std::vector<float>& image_out, size_t size, size_t countX, size_t countY, PerfTime* perf_out = nullptr);
};

class stereo
{
private:
	float sum = 0;
	float disparity = 0;
	int pos = 0;
public:
	size_t window = 10;
	float disp_est = 0;
	int dmax = 50;
	/*
	* Find Indexes:
	* str: Input String
	* s: Compared string
	*/
	int find_idx(std::string& str, std::string& s);

	/*
	* Load Files:
	* Path: Directory where images are located
	* Filename: Filenames with complete path are returned in this varibale
	*/
	int load_files(std::string path, std::vector < std::string >& filename);

	std::vector<float> load_images(std::vector<float>& image_in, std::vector<float>& image_out, std::size_t inputWidth, std::size_t inputHeight);
	/*
	* Compute Normalized Cross Correlation disparity
	* imageL: Left Camera Image
	* imageR: Right Camera Image
	* disp_image: Output Disparity Map image
	* CountX: Width  or number of columns in image
	* countY: Height  or number of rows in image
	*/
	template<typename tt>
	tt NCC_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out = nullptr);

	/*
	* Compute Sum of Absolute difference disparity
	* imageL: Left Camera Image
	* imageR: Right Camera Image
	* disp_image: Output Disparity Map image
	* CountX: Width  or number of columns in image
	* countY: Height  or number of rows in image
	*/
	template<typename tt>
	tt SAD_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out = nullptr);
};

//Compute Sum of Absolute Difference and Disparity map
template<typename tt>
tt stereo::SAD_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out)
{
    PERF_BEGIN
	LOG("SAD Processing started...........");
	std::pair<float, float> src(0, dmax), dst(0, 1);
	std::size_t count = window * window;
	std::vector<float> L(count);
	std::vector<float> R(count);
	std::vector<float> diff(count);
	float* sad_val = new float[countX];
	int best_match = 0;

	for (int i = dmax; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float min = 10000.01;
			vec::Rect<std::vector<float>>(imageL, i, j, window, countX, countY, L);	//Reference window in the left image
			for (int t = i - dmax; t < i; t++) {
				vec::Rect<std::vector<float>>(imageR, t, j, window, countX, countY, R);	//Comapring window in the right image
				vec::vec_subtract<std::vector<float>>(L, R, diff, window);
				sum = vec::vec_sum<std::vector<float>>(diff, window);
				sad_val[t] = sum;
				if (min > sum) {
					min = sum;
					disparity = i - t;
					best_match = t;
				}
			}
			//Find Sub pixel for better localization
			disp_est = disparity - 0.5 * ( (sad_val[best_match + 1] - sad_val[best_match - 1]) /
										(sad_val[best_match - 1] - (2 * sad_val[best_match]) + sad_val[best_match + 1]));
			auto disp_est1 = vec::map_value(src, dst, disp_est);
			disp_img[vec::getIndexGlobal(countX, i, j)] = disp_est1;

		}

	}
	LOG("SAD Processing Completed");
	PERF_END(perf_out)
	return disp_img;
}

//Compute Normalized cross Correlation and Disparity map
template<typename tt>
tt stereo::NCC_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out)
{
    PERF_BEGIN
	LOG("NCC Processing started...........");
	std::pair<float, float> src(0, dmax), dst(0, 1);
	std::size_t count = window * window;
	std::vector<float> L(count);
	std::vector<float> R(count);
	std::vector<float> diff(count);
	std::vector<float> R_sq(count);
	std::vector<float> L_sq(count);
	std::vector<float> prod(count);
	float* ncc_val = new float[countX];
	int best_match = 0;
	for (int i = dmax; i < (int)countX; i++)
	{
		for (int j = 0; j < (int)countY; j++)
		{
			float max = 0.000001;
			vec::Rect<std::vector<float>>(imageL, i, j, window, countX, countY, L);
			for (int t = i - dmax; t < i; t++)
			{
				vec::Rect<std::vector<float>>(imageR, t, j, window, countX, countY, R);
				vec::vec_multiply(L, R, prod, window);
				auto summ = vec::vec_sum<std::vector<float>>(prod, window);
				vec::vec_multiply(L, L, L_sq, window);
				vec::vec_multiply(R, R, R_sq, window);
				auto denom = std::sqrt(vec::vec_sum<std::vector<float>>(L_sq, window) * vec::vec_sum<std::vector<float>>(R_sq, window));
				auto norm = summ / denom;
				ncc_val[t] = norm;
				if (norm > max) {
					max = norm;
					disparity = i - t;
					best_match = t;
				}
			}
			disp_est = disparity - 0.5 * ((ncc_val[best_match + 1] - ncc_val[best_match - 1]) /
				(ncc_val[best_match - 1] - (2 * ncc_val[best_match]) + ncc_val[best_match + 1]));
			auto disp_est1 = vec::map_value(src, dst, disp_est);
			disp_img[vec::getIndexGlobal(countX, i, j)] = disp_est1;
		}
	}
	LOG("NCC Processing Completed");
	PERF_END(perf_out)
	return disp_img;
}

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

#include <boost/lexical_cast.hpp>

namespace vec {
	template<typename tVal>
	tVal map_value(std::pair<tVal, tVal> src, std::pair<tVal, tVal> dst, tVal val);
	int getIndexGlobal(std::size_t countX, int i, int j);
	float getValueGlobal(const std::vector<float>& img, std::size_t countX, std::size_t countY, int i, int j);
	template<typename T>
	T Rect(T& in_image, int x, int y, size_t window, int countX, int countY, T& rect);
	template<typename T>
	T vec_multiply(T& vec1, T& vec2, T& vec_out, size_t window);
	template<typename T>
	T vec_addition(T& vec1, T& vec2, T& vec_out, size_t window);
	template<typename T>
	T vec_subtract(T& vec1, T& vec2, T& vec_out, size_t window);
	template<typename T>
	float vec_sum(T& vec1, size_t window);
	void printperformance_data(std::string cpu_time, std::string sendtime, std::string gpu_time, std::string rec_time, float speedup, float speedup_w_m);

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
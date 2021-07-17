#include "vec_operation.h"

namespace vec {

	void show_options() {
		//Write Options
	}

	int getIndexGlobal(std::size_t countX, int i, int j) {
		return j * countX + i;
	}

	float getValueGlobal(const std::vector<float>& img, std::size_t countX, std::size_t countY, int i, int j) {
		if (i < 0 || (size_t)i >= countX || j < 0 || (size_t)j >= countY)
			return 0;
		else
			return img[getIndexGlobal(countX, i, j)];
	}

	void print_performance(std::vector<std::string> &performance)
	{

		for (int i = 0; i < performance.size(); i++) {
			LOG(performance[i]);
		}
	}

	std::vector<uint8_t> scaler_multiply(std::vector<float>& vec, uint8_t num) {
		std::vector<uint8_t> vec_out(vec.size());
		for (int i = 0; i < vec.size(); i++) {
			vec_out[i] = uint8_t(vec[i] * num);
		}
		return vec_out;
	}

	float find_median(std::vector<float>& vec) {
		float temp = 0;
		float m_median = 0;
		for (int i = 0; i < vec.size(); i++) {
			for (int j = i + 1; j < vec.size(); j++) {
				if (vec[i] >= vec[j]) {
					temp = vec[i];
					vec[i] = vec[j];
					vec[j] = temp;
				}
			}
		}
		vec.size() % 2 ? m_median = vec[vec.size() / 2] : m_median = vec[(vec.size() - 1) / 2 + 1];

		return m_median;
	}
	std::vector<float> clone(std::vector<float>& src, std::vector<float>& dst) {
		for (int i = 0; i < src.size(); i++) {
			dst[i] = src[i];
		}
		return dst;
	}

}
int stereo::find_idx(std::string& str, std::string& s)
{
    int pos = 0;
	bool flag = false;
	for (int i = 0; i < str.length(); i++) {
		if (str.substr(i, s.length()) == s) {
			pos = i;
			flag = true;
		}
	}
	if (flag == false) {
		LOG("NONE");
	}
	return pos;
}

int stereo::load_files(std::string path, std::vector < std::string >& filename) {
    int pos = 0;
	int i = 0;
	for (auto& entry : std::filesystem::directory_iterator(path))
	{
		std::string path_string{ entry.path().u8string() };
		filename.push_back(path_string);
		std::string str1 = "/";
		pos = find_idx(filename[i], str1) + 1;
		std::string str2 = filename[i].substr(pos, filename[i].length());
		line("Index# "); line(i); line(" ");  LOG(str2);
		i++;

	}
	return i-1;
}

std::vector<float> stereo::load_images(std::vector<float>& image_in, std::vector<float>& image_out, std::size_t inputWidth, std::size_t inputHeight) {
	for (size_t j = 0; j < inputHeight; j++) {
		for (size_t i = 0; i < inputWidth; i++) {
			image_out[i + inputWidth * j] = image_in[(i % inputWidth) + inputWidth * (j % inputHeight)];
		}
	}
	return image_out;
}

std::vector<float> filter::histogram(std::vector<float>& image, std::vector<float>& imageE, PerfTime* perf_out) {
    PERF_BEGIN

	std::vector<uint8_t> grey_image = vec::scaler_multiply(image, 255);
	int grey_values[256] = { 0 };
	float PDF[256] = { 0 };
	float CDF[256] = { 0.00001 };
	uint8_t Sk[256] = { 0 };

	for (int i = 0; i < grey_image.size(); i++)
		grey_values[grey_image[i]]++;

	PDF[0] = float(grey_values[0]) / float(grey_image.size());
	CDF[0] = PDF[0];
	for (int g = 1; g <= 255; g++) {
		PDF[g] = float(grey_values[g]) / float(grey_image.size());
		CDF[g] = float(PDF[g] + CDF[g - 1]);
	}

	for (int i = 0; i < grey_image.size(); i++)
		imageE[i] = float(CDF[grey_image[i]]);
	Core::writeImagePGM("hist.pgm", imageE, 450, 375);

	PERF_END(perf_out)
	return imageE;
}


std::vector<float> filter::median_fltr(std::vector<float>& image, std::vector<float>& image_out, size_t size, size_t countX, size_t countY, PerfTime* perf_out) {
    PERF_BEGIN
	LOG("Median");
	vec::clone(image, image_out);
	std::size_t count = size * size;
	std::vector<float> rect(count);
	float m_median = 0;
	for (int i = 0; i < countX-1; i++) {
		for (int j = 0; j < countY-1; j++) {
			vec::Rect<std::vector<float>>(image, i, j, size, countX, countY, rect);
			m_median = vec::find_median(rect);
			image_out[vec::getIndexGlobal(countX, i+1, j+1)] = m_median;
		}
	}

	PERF_END(perf_out)
	return image_out;
}

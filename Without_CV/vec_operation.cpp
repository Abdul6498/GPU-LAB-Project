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

	void printperformance_data(std::string cpu_time, std::string sendtime, std::string gpu_time, std::string rec_time, float speedup, float speedup_w_m)
	{
		std::cout << "CPU Time : " << cpu_time << std::endl;
		std::cout << "Copy Time : " << sendtime << std::endl;
		std::cout << "GPU Time : " << gpu_time << std::endl;
		std::cout << "Download Time : " << rec_time << std::endl;
		std::cout << "Speed Up : " << speedup << std::endl;
		std::cout << "Speed Up with memory copy : " << speedup_w_m << std::endl;
	}

}
int stereo::find_idx(std::string& str, std::string& s)
{

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
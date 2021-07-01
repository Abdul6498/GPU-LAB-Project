#include "vec_operation.h"

namespace vec {

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

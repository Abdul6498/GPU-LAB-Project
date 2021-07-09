
#include "vec_operation.h"

std::pair<float, float> src(0, 50), dst(0, 1);

enum disp_algo {
	SAD_Disprity,
	NCC_Disparity
};

int main(int argc, char** argv) {

	int idx = -1;	//golobal index for total number of files is directory
	int idx_L = -1;	//Get user index for left image
	int idx_R = -1;	//Get user index for Right image

	stereo stereo;	// Class for stereo functions
	stereo.window = 10;	//window size
	stereo.dmax = 50;	//Max diparity
	std::vector < std::string > filename;	//Variable for file name

	std::string path = "../../../../Opencl-ex1/images/";	//Path for image directory ends with/
	idx = stereo.load_files(path, filename);	//Load files


	line("Select Left image index: ");
	GetValue(idx_L);
	line("Select Right image index: ");
	GetValue(idx_R);

	if (idx_L > idx || idx_R > idx)
	{
		std::cerr << "Error! Input indexes are out of Range \n";
		return -1;
	}
	else if (idx_L == idx_R)
	{
		std::cerr << "Warning! You are comparing the same image \n";
	}


	
	std::vector<float> inputData;	//Variable for image data
	std::size_t inputWidth, inputHeight;	//Variables for width and height of image
	Core::readImagePGM(filename[idx_L], inputData, inputWidth, inputHeight);	//Read image dat from the specific path
	std::size_t countX = inputWidth;	
	std::size_t countY = inputHeight;

	std::vector<float> imageL(inputData.size());	//Left image
	std::vector<float> imageR(inputData.size());	//Right image
	std::vector<float> imageD(inputData.size());	//Disparity map

	{
		//Read Left image Data
		for (size_t j = 0; j < inputHeight; j++) {
			for (size_t i = 0; i < inputWidth; i++) {
				imageL[i + inputWidth * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
		Core::readImagePGM(filename[idx_R], inputData, inputWidth, inputHeight);

		//Read Right image Data
		for (size_t j = 0; j < inputHeight; j++) {
			for (size_t i = 0; i < inputWidth; i++) {
				imageR[i + inputWidth * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}
		auto sad_start = std::chrono::high_resolution_clock::now();

		stereo.SAD_disparity<std::vector<float>>(imageL, imageR, imageD, countX, countY);	//SAD Disparity Calculation

		auto sad_end = std::chrono::high_resolution_clock::now();
		auto sad_time = std::chrono::duration_cast<std::chrono::seconds>(sad_end - sad_start).count();

		Core::writeImagePGM("SAD_out.pgm", imageD, countX, countY);

		auto ncc_start = std::chrono::high_resolution_clock::now();
		stereo.NCC_disparity<std::vector<float>>(imageL, imageR, imageD, countX, countY);	//NCC Disparity Calculation
		auto ncc_end = std::chrono::high_resolution_clock::now();
		auto ncc_time = std::chrono::duration_cast<std::chrono::seconds>(ncc_end - ncc_start).count();

		Core::writeImagePGM("NCC_out.pgm", imageD, countX, countY);	

	LOG("Completed");
	return 0;
}
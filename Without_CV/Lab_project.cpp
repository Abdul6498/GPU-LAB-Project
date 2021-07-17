
#include "vec_operation.h"
#include <windows.h>
#pragma comment(lib, "user32.lib")

int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <path to folder with image files>" << std::endl;
	//	exit(1);
	}

	SYSTEM_INFO siSysInfo;

	// Copy the hardware information to the SYSTEM_INFO structure. 

	GetSystemInfo(&siSysInfo);

	std::string input_path = "../../../images/";	//Path for image directory ends with/ // argv[1];
	int idx = -1;	//golobal index for total number of files is directory
	int idx_L = -1;	//Get user index for left image
	int idx_R = -1;	//Get user index for Right image

	stereo stereo;	// Class for stereo functions
	stereo.window = 3;	//window size
	stereo.dmax = 50;	//Max diparity
	auto median_mask = 5; //Median Mask size
	std::vector < std::string > filename;	//Variable for file name

	//std::string path = "../../../images/";	//Path for image directory ends with/
	idx = stereo.load_files(input_path, filename);	//Load files

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

	filter filter;

	std::vector<float> inputData_L, inputData_R;	//Variable for image data
	std::size_t inputWidth, inputHeight;	//Variables for width and height of image

	Core::readImagePGM(filename[idx_L], inputData_L, inputWidth, inputHeight);	//Read image dat from the specific path
	Core::readImagePGM(filename[idx_R], inputData_R, inputWidth, inputHeight);

	std::size_t countX = inputWidth;
	std::size_t countY = inputHeight;

	static std::vector<float> imageL(inputData_L.size());	//Left image
	static std::vector<float> imageR(inputData_L.size());	//Right image
	static std::vector<float> image_sad(inputData_L.size());	//Disparity map
	static std::vector<float> imageE(inputData_L.size());	//Hist
	static std::vector<float> imageM_sad(inputData_L.size());	//Median
	static std::vector<float> imageM_ncc(inputData_L.size());	//Median
	static std::vector<float> imageH(inputData_L.size());	//Hist
	static std::vector<float> image_ncc(inputData_L.size());	//Disparity map

	std::string med_sad_out_path = input_path + "Median_SAD_out.pgm";
	std::string med_ncc_out_path = input_path + "Median_NCC_out.pgm";
	std::string sad_out_path = input_path + "SAD_out.pgm";
	std::string ncc_out_path = input_path + "NCC_out.pgm";

	std::vector<std::string> performance;

	PerfTime hist_l, hist_r;


	std::thread thistL(&filter::histogram, &filter, std::ref(inputData_L), std::ref(imageE), &hist_l);
	std::thread thistR(&filter::histogram, &filter, std::ref(inputData_R), std::ref(imageH), &hist_r);
	thistL.join();
	thistR.join();

	auto hist_time = hist_l + hist_r;

	std::thread tLdL(&stereo::load_images, &stereo, std::ref(imageE), std::ref(imageL), inputWidth, inputHeight);
	std::thread tLdR(&stereo::load_images, &stereo, std::ref(imageH), std::ref(imageR), inputWidth, inputHeight);
	tLdL.join();
	tLdR.join();

	PerfTime sad_time, ncc_time;

	std::thread tsad(&stereo::SAD_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_sad), countX, countY, &sad_time);
	std::thread tncc(&stereo::NCC_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_ncc), countX, countY, &ncc_time);
	tsad.join();
	tncc.join();

	PerfTime sad_med_time, ncc_med_time;
	std::thread tmsad(&filter::median_fltr, &filter, std::ref(image_sad), std::ref(imageM_sad), median_mask, countX, countY, &sad_med_time);
	std::thread tmncc(&filter::median_fltr, &filter, std::ref(image_ncc), std::ref(imageM_ncc), median_mask, countX, countY, &ncc_med_time);
	tmsad.join();
	tmncc.join();
	auto med_time = sad_med_time + ncc_med_time;


	Core::writeImagePGM(sad_out_path, image_sad, countX, countY);
	Core::writeImagePGM(ncc_out_path, image_ncc, countX, countY);
	Core::writeImagePGM(med_sad_out_path, imageM_sad, countX, countY);
	Core::writeImagePGM(med_ncc_out_path, imageM_ncc, countX, countY);

	
	performance.push_back("[CPU INFORMATION]");
	performance.push_back("OEM ID: " + std::to_string(siSysInfo.dwOemId));
	performance.push_back("Number of processors: " + std::to_string(siSysInfo.dwNumberOfProcessors));
	performance.push_back("Page size: " + std::to_string(siSysInfo.dwPageSize));
	performance.push_back("Processor type:" + std::to_string(siSysInfo.dwProcessorType));
	performance.push_back("[CPU] Histogram Execution Time: " + std::to_string(hist_time) + "ms");
	performance.push_back("[CPU] SAD Execution Time: " + std::to_string(sad_time) + "ms");
	performance.push_back("[CPU] NCC Execution Time: " + std::to_string(ncc_time) + "ms");
	performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");

	vec::print_performance(performance);
	LOG("Completed");
	return 0;
}
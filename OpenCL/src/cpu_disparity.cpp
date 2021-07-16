#include "cpu_disparity.h"

void cpu_disparity() {
	int idx = -1;	//golobal index for total number of files is directory
	int idx_L = -1;	//Get user index for left image
	int idx_R = -1;	//Get user index for Right image

	stereo stereo;	// Class for stereo functions
	stereo.window = 3;	//window size
	stereo.dmax = 50;	//Max diparity
	std::vector < std::string > filename;	//Variable for file name

	std::string path = "../../OpenCL/test/";	//Path for image directory ends with/
	idx = stereo.load_files(path, filename);	//Load files

	line("Select Left image index: ");
	GetValue(idx_L);
	line("Select Right image index: ");
	GetValue(idx_R);

	if (idx_L > idx || idx_R > idx)
	{
		std::cerr << "Error! Input indexes are out of Range \n";
		return;
	}
	else if (idx_L == idx_R)
	{
		std::cerr << "Warning! You are comparing the same image \n";
	}

	filter filter;

	std::vector<float> inputData , inputData_R;	//Variable for image data
	std::size_t inputWidth, inputHeight;	//Variables for width and height of image
	Core::readImagePGM(filename[idx_L], inputData, inputWidth, inputHeight);	//Read image dat from the specific path
	Core::readImagePGM(filename[idx_R], inputData_R, inputWidth, inputHeight);

	std::size_t countX = inputWidth;
	std::size_t countY = inputHeight;

	static std::vector<float> imageL(inputData.size());	//Left image
	static std::vector<float> imageR(inputData.size());	//Right image
	static std::vector<float> image_sad(inputData.size());	//Disparity map
	static std::vector<float> imageE(inputData.size());	//Hist
	static std::vector<float> imageM_sad(inputData.size());	//Median
	static std::vector<float> imageM_ncc(inputData.size());	//Median
	static std::vector<float> imageH(inputData.size());	//Hist
	static std::vector<float> image_ncc(inputData.size());	//Disparity map


	std::thread t1(&filter::histogram, &filter, std::ref(inputData), std::ref(imageE));
	std::thread t2(&filter::histogram, &filter, std::ref(inputData_R), std::ref(imageH));
	t1.join();
	t2.join();

	std::thread t3(&stereo::load_images, &stereo, std::ref(imageE), std::ref(imageL), inputWidth, inputHeight);
	std::thread t4(&stereo::load_images, &stereo, std::ref(imageH), std::ref(imageR), inputWidth, inputHeight);
	t3.join();
	t4.join();

	stereo.SAD_disparity<std::vector<float>>(imageL, imageR, image_sad, countX, countY);	//SAD Disparity Calculation

	std::thread t5(&stereo::NCC_disparity<std::vector<float>>, &stereo, std::ref(imageL), std::ref(imageR), std::ref(image_ncc), countX, countY);

	std::thread t6(&filter::median_fltr, &filter, std::ref(image_sad), std::ref(imageM_sad), 5, countX, countY);

	Core::writeImagePGM("SAD_out.pgm", image_sad, countX, countY);

	auto sad_start = std::chrono::high_resolution_clock::now();
	t5.join();
	Core::writeImagePGM("NCC_out.pgm", image_ncc, countX, countY);

	std::thread t7(&filter::median_fltr, &filter, std::ref(image_ncc), std::ref(imageM_ncc), 5, countX, countY);

	t6.join();

	Core::writeImagePGM("Median_SAD_out.pgm", imageM_sad, countX, countY);
	t7.join();

	auto sad_end = std::chrono::high_resolution_clock::now();
	auto sad_time = std::chrono::duration_cast<std::chrono::seconds>(sad_end - sad_start).count();
	auto ncc_start = std::chrono::high_resolution_clock::now();

	auto ncc_end = std::chrono::high_resolution_clock::now();
	auto ncc_time = std::chrono::duration_cast<std::chrono::seconds>(ncc_end - ncc_start).count();
	LOG(sad_time);
	LOG(ncc_time);

	Core::writeImagePGM("Median_NCC_out.pgm", imageM_ncc, countX, countY);

	LOG("Completed");
}

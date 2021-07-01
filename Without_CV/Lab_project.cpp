
#include <vec_operation.h>

#define LOG(x) std::cout << x << std::endl;
#define LOG_w(x) std::cout << x << " , ";
#define newline() std::cout << std::endl;
#define GetValue(x) std::cin >> x ;
#define line(x) std::cout << x;

unsigned volatile kernel_size = 3;     //Kernel Size for averging filter
size_t window = 10;
std::pair<float, float> src(0, 50), dst(0, 1);

template<typename tt>
tt SAD_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY)
{
	std::size_t count = window * window;
	std::vector<float> L(count);
	std::vector<float> R(count);
	std::vector<float> diff(count);
	float sum;
	float disparity = 0;
	for (int i = 48; i < (int)countX; i++) {
		for (int j = 0; j < (int)countY; j++) {
			float min = 10000.01;
			vec::Rect<std::vector<float>>(imageL, i, j, window, countX, countY, L);
			for (int t = i - 48; t < i; t++) {
				vec::Rect<std::vector<float>>(imageR, t, j, window, countX, countY, R);
				vec::vec_subtract<std::vector<float>>(L, R, diff, window);
				sum = vec::vec_sum<std::vector<float>>(diff, window);
				if (min > sum) {
					min = sum;
					disparity = i - t;
					//  point_x = x;
					//  point_t = t;
				}
			}

			disparity = vec::map_value(src, dst, disparity);
			disp_img[vec::getIndexGlobal(countX, i, j)] = disparity;
		}
	}
	return disp_img;
}

template<typename tt>
tt NCC_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY)
{
	std::size_t count = window * window;
	std::vector<float> L(count);
	std::vector<float> R(count);
	std::vector<float> diff(count);
	std::vector<float> R_sq(count);
	std::vector<float> L_sq(count);
	std::vector<float> prod(count);
	float sum;
	float disparity = 0;
	for (int i = 48; i < (int)countX; i++) 
	{
		for (int j = 0; j < (int)countY; j++) 
		{
			float max = 0.000001;
			vec::Rect<std::vector<float>>(imageL, i, j, window, countX, countY, L);
			for (int t = i - 48; t < i; t++) 
			{
				vec::Rect<std::vector<float>>(imageR, t, j, window, countX, countY, R);
				vec::vec_multiply(L, R, prod, window);
				auto summ = vec::vec_sum<std::vector<float>>(prod, window);
				vec::vec_multiply(L, L, L_sq, window);
				vec::vec_multiply(R, R, R_sq, window);
				auto denom = std::sqrt(vec::vec_sum<std::vector<float>>(L_sq,window) * vec::vec_sum<std::vector<float>>(R_sq, window));
				auto norm = summ / denom;
				if (norm > max) {
					max = norm;
					disparity = i - t;
					//  point_x = x;
					//  point_t = t;
				}
			}
			// sum = 0;
		   //  for (int p_x = point_x; p_x < 3; p_x++)
		   //  {
			   //  sum = int(imageL.at<uchar>(cv::Point(p_x, y))) + sum;
			   //  cog = sum / 3;
		   //  }
			disparity = vec::map_value(src, dst, disparity);
			disp_img[vec::getIndexGlobal(countX, i, j)] = disparity;
		}
	}
	return disp_img;
}


int main(int argc, char** argv) {

	std::size_t wgSizeX = 16; // Number of work items per work group in X direction
	std::size_t wgSizeY = 16;
	//std::size_t countX = wgSizeX * 40; // Overall number of work items in X direction = Number of elements in X direction
	//std::size_t countY = wgSizeY * 30;
	std::size_t countX = 450; // Overall number of work items in X direction = Number of elements in X direction
	std::size_t countY = 375;
	std::size_t count = countX * countY; // Overall number of elements
	std::vector<float> imageL(count);
	std::vector<float> imageR(count);
	std::vector<float> imageD(count);
	
	{
		std::vector<float> inputData;
		std::size_t inputWidth, inputHeight;
		Core::readImagePGM("D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/im2.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				imageL[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
		Core::readImagePGM("D:/INFOTECH/2nd Semester/High Performance Programming with graphic cards/Exercise/Opencl-ex1/Opencl-ex1/im6.pgm", inputData, inputWidth, inputHeight);
		for (size_t j = 0; j < countY; j++) {
			for (size_t i = 0; i < countX; i++) {
				imageR[i + countX * j] = inputData[(i % inputWidth) + inputWidth * (j % inputHeight)];
			}
		}
	}
		SAD_disparity<std::vector<float>>(imageL, imageR, imageD, countX, countY);
		Core::writeImagePGM("SAD_out.pgm", imageD, countX, countY);
		NCC_disparity<std::vector<float>>(imageL, imageR, imageD, countX, countY);
		Core::writeImagePGM("NCC_out.pgm", imageD, countX, countY);
	LOG("Completed");
	return 0;
}
```
Under Construction
```
# CPU Implementation

This repository contains CPU implementation of the Disparity map. Two different implementations are performed and compare the performance for both of them. In the first implementation, OpenCV is used for preprocessing of images and calculation of Disparity Map. In the second implementation, conventional C++ libraries are used and all the functions are created by the author. Moreover, Multithreading is used to increase the throughput and decrease the total execution time.


## Repository Structure
``` cpp
Abdul (Brach)
├──GPU-LAB-Project (OpenCV Implementation)
│  ├── app
│  │    └── main.cpp
│  ├── im2.pgm
│  └── im6.pgm 
└── Without_CV (Conventional C++ Implementation)
    ├── CMakeLists.txt
    ├── app
    │   └── Lab_project.cpp
    ├── include
    │   └── vec_operation.h
    ├── src
    │   └── vec_operation.cpp
    └── images
        ├── im2.pgm
        ├── im6.pgm
        ├── SAD_out.pgm
        ├── NCC_out.pgm
        ├── Median_NCC_out.pgm
        └── Median_SAD_out.pgm
```
## OpenCV Implementation:
### Usage
TODO
### Functions
The following functions are used for the calculation of the disparity map.
- Load Images
```cpp
      cv::Mat imageL = cv::imread("im2.png");
      cv::Mat imageR = cv::imread("im6.png");
```
- Histogram Equalization
```cpp
       cv::equalizeHist(imageL, imageL);
       cv::equalizeHist(imageR, imageR);
```
- SAD Disparity Map
```cpp
        SAD_disparity<cv::Mat>(imageL, imageR, SAD_out);
```
- NCC Disparity Map
```cpp
        NCC_disparity<cv::Mat>(imageL, imageR, NCC_out);
```
- Denoising
```cpp
        cv::Mat SAD_out_M = myMedian(SAD_out);
        cv::Mat NCC_out_M = myMedian(NCC_out);
```

## Conventional C++ Implementation:
### Usage
Install OpenCV in visual studio and clone GPU-LAB-Project/GPU-LAB-Project repository in the local folder. Import it from visual studio and run the main.cpp file. Make sure that images are also in the same directory. The output will be shown on the screen.
###Functions
The following functions are used for the calculation of the disparity map. 

- Load Images
```cpp
	std::vector<float> stereo::load_images(std::vector<float>& image_in, std::vector<float>& image_out, std::size_t inputWidth, std::size_t inputHeight) {
```
- Histogram Equalization
```cpp
    std::vector<float> filter::histogram(std::vector<float>& image, std::vector<float>& imageE, PerfTime* perf_out)
```
- SAD Disparity Map
```cpp
    tt stereo::SAD_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out)
```
  - Subpixel Calculation.
```cpp
    disp_est = disparity - 0.5 * ( (sad_val[best_match + 1] - sad_val[best_match - 1]) / 
								(sad_val[best_match - 1] - (2 * sad_val[best_match]) + sad_val[best_match + 1]));
```
- NCC Disparity Map
```cpp
    tt stereo::NCC_disparity(tt& imageL, tt& imageR, tt& disp_img, int countX, int countY, PerfTime* perf_out)
```
  - Subpixel Calculation.
```cpp
   disp_est = disparity - 0.5 * ((ncc_val[best_match + 1] - ncc_val[best_match - 1]) /
				(ncc_val[best_match - 1] - (2 * ncc_val[best_match]) + ncc_val[best_match + 1]));
```
- Denoising
```cpp
  std::vector<float> filter::median_fltr(std::vector<float>& image, std::vector<float>& image_out, size_t size, size_t countX, size_t countY, PerfTime* perf_out)
```

## Output:

```bash
Index# 0 im2.pgm
Index# 1 im6.pgm
Index# 2 Median_NCC_out.pgm
Index# 3 Median_SAD_out.pgm
Index# 4 NCC_out.pgm
Index# 5 SAD_out.pgm
Select Left image index: 0
Select Right image index: 1
SAD Processing started...........
NCC Processing started...........
SAD Processing Completed
NCC Processing Completed
Median
Median
[CPU INFORMATION]
OEM ID: 9
Number of processors: 12
Page size: 4096
Processor type:8664
[CPU] Histogram Execution Time: 4ms
[CPU] SAD Execution Time: 1675ms
[CPU] NCC Execution Time: 3122ms
[CPU] Median Execution Time: 391ms
Completed
```
### Instructions
### Files
### References
1. https://en.wikipedia.org/wiki/Sum_of_absolute_differences
2. https://vision.middlebury.edu/stereo/data/scenes2003/
3. https://www.hindawi.com/journals/js/2016/8742920/#EEq3
4. https://robotacademy.net.au/lesson/computing-disparity/
5. https://stackoverflow.com/questions/17607312/what-is-the-difference-between-a-disparity-map-and-a-disparity-image-in-stereo-m
6. http://mccormickml.com/2014/01/10/stereo-vision-tutorial-part-i/
7. https://en.wikipedia.org/wiki/Bayer_filter
8. https://en.wikipedia.org/wiki/Image_rectification

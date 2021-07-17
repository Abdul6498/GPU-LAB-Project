#include <iostream>
#include <math.h>
#include <utility>
#include <chrono>
#include <iostream>
#include <array>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#define LOG(x) std::cout << x << std::endl;
#define LOG_w(x) std::cout << x << " , ";
#define newline() std::cout << std::endl;
#define GetValue(x) std::cin >> x ;

unsigned volatile kernel_size = 3;     //Kernel Size for averging filter
unsigned window = 3;
std::pair<int, int> src(0, 50), dst(0, 255);

template<typename tVal>
tVal map_value(std::pair<tVal, tVal> src, std::pair<tVal, tVal> dst, tVal val)
{
    return(src.first + ((dst.second - dst.first) / (src.second - src.first)) * (val - src.first));
}

template<typename t>
t map(t in_lower, t in_upper, t out_lower, t out_upper, t value)
{
    return(out_lower + ((out_upper - out_lower) / (in_upper - in_lower)) * (value - in_lower));
}

cv::Mat average_fn(cv::Mat img_in)
{
    unsigned int pixel_values;
    unsigned int  avg_val = 0, count = 0, sum_pxl = 0, k = 0;

    for (int y = 0; y < img_in.rows; y++)                                            //First inner loop(Nested Loop) for Rows in the image(Iteration size: 0 to No. of Rows)
    {
        for (int i = 0; i < img_in.cols - kernel_size; i++)                          //Second inner loop(Nested Loop) for Coloumns in the image(Iteration size: 0 to No. of cols - kernel size)
        {
            for (int x = i; x < i + kernel_size; x++)                                   //Apply moving average
            {

                pixel_values = img_in.at<uchar>(y, x);
                sum_pxl = pixel_values + sum_pxl;

                if (count == (kernel_size - 1))
                {
                    k = cv::abs(x - (kernel_size / 2));
                    avg_val = sum_pxl / kernel_size;
                    img_in.at<uchar>(y, k) = avg_val;                                //Change the brightness of pixel according to the average
                    sum_pxl = 0;
                    avg_val = 0;
                    count = 0;
                }
                else
                {
                    count++;
                }
            }
        }
    }
    return img_in;
    
}

template<typename tt>
tt SAD_disparity(tt imageL, tt imageR, tt disp_img)
{
    int disparity = 0;
    for (uint y = 0; y < imageL.rows - window; y++)
    {
        for (uint x = 48; x < imageL.cols - window; x++)
        {
            int min = 10000;
            cv::Mat L(imageL, cv::Rect(x, y, window, window));
            for (uint t = x - 48; t < x; t++) {
                cv::Mat R(imageR, cv::Rect(t, y, window, window));
                cv::Mat diff = cv::abs(L - R);
                int abs_sum = cv::sum(diff)[0];
                if (min > abs_sum) {
                    min = abs_sum;
                    disparity = x - t;
                }
            }
            
            disparity = map_value(src, dst, disparity);
            disp_img.at<uchar>(cv::Point(x, y)) = disparity;
        }
    }
    return(disp_img);
}

template<typename ncc>
ncc NCC_disparity(ncc imageL, ncc imageR, ncc disp_img)
{
    //NCC Code
    int disparity = 0;
    cv::Mat prod;
    cv::Mat temp_R_sq;
    cv::Mat temp_L_sq;

    for (uint y = 0; y < imageL.rows - window; y++)
    {
        for (uint x = 48; x < imageL.cols - window; x++)
        {
            double max = 0.000001;
            cv::Mat L(imageL, cv::Rect(x, y, window, window));
            for (uint t = x - 48; t < x; t++) {
                cv::Mat R(imageR, cv::Rect(t, y, window, window));
                cv::multiply(L, R, prod, 1, CV_32S);
                auto summ = cv::sum(prod)[0];
                cv::multiply(L, L, temp_L_sq, 1, CV_32S);
                cv::multiply(R, R, temp_R_sq, 1, CV_32S);
                auto denom = std::sqrt(cv::sum(temp_L_sq)[0] * cv::sum(temp_R_sq)[0]);
                auto norm = summ / denom;
                if (norm > max) {
                    max = norm;
                    disparity = x - t;
                }
            }

            disparity = map_value(src, dst, disparity);
            disp_img.at<uchar>(cv::Point(x, y)) = disparity;
        }
    }
    return(disp_img);
}

template<typename sh>
sh affine_t(sh image)
{
    //Brigtness and contrast
    cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
    double alpha = 3; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {

            if (cv::saturate_cast<uchar>(alpha * image.at<uchar>(y, x) + beta) > 255)
                new_image.at<char>(y, x) = 255;
            else
                new_image.at<char>(y, x) = cv::saturate_cast<uchar>(alpha * image.at<uchar>(y, x) + beta);
        }
    }
    return(new_image);
}

void load_images(const cv::String& dirname, std::vector< cv::Mat >& img_lst)
{
    std::vector<cv::String> files;
    cv::glob(dirname, files);

    size_t count = files.size(); //number of bmp files in test images folder
    for (size_t i = 0; i < count; i++)
        img_lst.push_back(cv::imread(files[i]));

}
int main()
{
    cv::Mat imageL = cv::imread("im2.png");
    cv::Mat imageR = cv::imread("im6.png");
    cv::cvtColor(imageL, imageL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageR, imageR, cv::COLOR_BGR2GRAY);
    // imageL = average_fn(imageL);               //Take average
    // imageR = average_fn(imageR);
    // imageL = affine_t<cv::Mat>(imageL);
    // imageR = affine_t<cv::Mat>(imageR);
    cv::equalizeHist(imageL, imageL);
    cv::equalizeHist(imageR, imageR);

    cv::Mat M = cv::Mat::zeros(imageL.rows, imageL.cols, CV_8UC1);

    auto start = std::chrono::high_resolution_clock::now();
    SAD_disparity<cv::Mat>(imageL, imageR, M);
    NCC_disparity<cv::Mat>(imageL, imageR, M);
    auto end = std::chrono::high_resolution_clock::now();
    auto tm = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);	// milliseconds
    std::cout << "time: " << tm.count() << " ms" << std::endl;
    LOG("Completed");
    cv::imshow("Disp", M);
    cv::imwrite("output__.bmp", M);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if (k == 's')
    {
        LOG("END");
    }
    return 0;
}



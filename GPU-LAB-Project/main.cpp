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
#include <filesystem>

#define LOG(x) std::cout << x << std::endl;
#define LOG_w(x) std::cout << x << " , ";
#define newline() std::cout << std::endl;
#define GetValue(x) std::cin >> x ;
#define line(x) std::cout << x;

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
    // process_images(img_in);   //Process the images without saving(Optional)//Don't uncomment without the permission of Author. 
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
cv::Mat myMedian(cv::Mat& srcImage)
{
    cv::Mat dstImage = srcImage.clone();
        std::vector<uchar>List;
        for (int k = 1; k < srcImage.cols - 1; k++)
        {
            for (int n = 1; n < srcImage.rows - 1; n++) {

                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                                List.push_back(srcImage.at<uchar>(n + j, k + i));
                        }
                    }
                    sort(List.begin(), List.end());
                    dstImage.at<uchar>(n, k) = List[5];
                    List.clear();
            }
        }
    return dstImage;
}
void print_performance(std::vector<std::string>& performance)
{

    for (int i = 0; i < performance.size(); i++) {
        LOG(performance[i]);
    }
}
int main()
{
    cv::Mat imageL = cv::imread("im2.png");
    cv::Mat imageR = cv::imread("im6.png");
    std::vector<std::string> performance;

    cv::cvtColor(imageL, imageL, cv::COLOR_BGR2GRAY);
    cv::cvtColor(imageR, imageR, cv::COLOR_BGR2GRAY);
    // imageL = average_fn(imageL);               //Take average
    // imageR = average_fn(imageR);
    // imageL = affine_t<cv::Mat>(imageL);
    // imageR = affine_t<cv::Mat>(imageR);
    auto hist_start = std::chrono::high_resolution_clock::now();
    cv::equalizeHist(imageL, imageL);
    cv::equalizeHist(imageR, imageR);
    auto hist_end = std::chrono::high_resolution_clock::now();
    auto hist_time = std::chrono::duration_cast<std::chrono::milliseconds>(hist_end - hist_start).count();

    int op = -1;
    LOG("Option [1]: SAD \nOption [2]: NCC \nOption [3]: SAD and NCC Both");
    line("Please choose option: ");
    GetValue(op);
    cv::Mat SAD_out = cv::Mat::zeros(imageL.rows, imageL.cols, CV_8UC1);
    cv::Mat NCC_out = cv::Mat::zeros(imageL.rows, imageL.cols, CV_8UC1);


    if (op == 1) {
        auto start = std::chrono::high_resolution_clock::now();
        SAD_disparity<cv::Mat>(imageL, imageR, SAD_out);
        auto end = std::chrono::high_resolution_clock::now();
        auto sad_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();	// milliseconds
        
        auto med_start = std::chrono::high_resolution_clock::now();
        cv::Mat SAD_out_M = myMedian(SAD_out);
        auto med_end = std::chrono::high_resolution_clock::now();
        auto med_time = std::chrono::duration_cast<std::chrono::milliseconds>(med_end - med_start).count();

        performance.push_back("[CPU] SAD Execution Time: " + std::to_string(sad_time) + "ms");
        performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");
        print_performance(performance);
        cv::imshow("SAD output", SAD_out);
        cv::imshow("SAD output With Median", SAD_out_M);
    }
    else if (op == 2) {
        auto start = std::chrono::high_resolution_clock::now();
        NCC_disparity<cv::Mat>(imageL, imageR, NCC_out);
        auto end = std::chrono::high_resolution_clock::now();
        auto ncc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();	// milliseconds

        auto med_start = std::chrono::high_resolution_clock::now();
        cv::Mat NCC_out_M = myMedian(NCC_out);
        auto med_end = std::chrono::high_resolution_clock::now();
        auto med_time = std::chrono::duration_cast<std::chrono::milliseconds>(med_end - med_start).count();

        performance.push_back("[CPU] SAD Execution Time: " + std::to_string(ncc_time) + "ms");
        performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");
        print_performance(performance);
        cv::imshow("NCC output", NCC_out);
        cv::imshow("NCC output With Median", NCC_out_M);
    }
        
    else if (op == 3)
    {
        auto start = std::chrono::high_resolution_clock::now();
        SAD_disparity<cv::Mat>(imageL, imageR, SAD_out);
        auto end = std::chrono::high_resolution_clock::now();
        auto sad_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();	// milliseconds

        start = std::chrono::high_resolution_clock::now();
        NCC_disparity<cv::Mat>(imageL, imageR, NCC_out);
        end = std::chrono::high_resolution_clock::now();
        auto ncc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();	// milliseconds

        auto med_start = std::chrono::high_resolution_clock::now();
        cv::Mat SAD_out_M = myMedian(SAD_out);
        cv::Mat NCC_out_M = myMedian(NCC_out);
        auto med_end = std::chrono::high_resolution_clock::now();
        auto med_time = std::chrono::duration_cast<std::chrono::milliseconds>(med_end - med_start).count();

        performance.push_back("[CPU] SAD Execution Time: " + std::to_string(sad_time) + "ms");
        performance.push_back("[CPU] NCC Execution Time: " + std::to_string(ncc_time) + "ms");
        performance.push_back("[CPU] Median Execution Time: " + std::to_string(med_time) + "ms");
        print_performance(performance);
        cv::imshow("SAD output", SAD_out);
        cv::imshow("SAD output With Median", SAD_out_M);

        cv::imshow("NCC output", NCC_out);
        cv::imshow("NCC output With Median", NCC_out_M);
    }
    else
        std::cerr << "Please choose correct option \n";


    LOG("Completed");
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    if (k == 's')
    {
        LOG("END");
    }
    return 0;
}



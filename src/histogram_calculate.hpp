
#ifndef HISTOGRAM_CALCULATE_HPP
#define HISTOGRAM_CALCULATE_HPP

#include "opencv4/opencv2/objdetect/objdetect.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/objdetect.hpp"
#include "opencv4/opencv2/face.hpp"
#include "opencv4/opencv2/ml.hpp"
#include "opencv4/opencv2/calib3d.hpp"
//#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <map>

using namespace std;
using namespace cv;

//------------------------------------------------------------------------------
// cv::elbp
//------------------------------------------------------------------------------
template <typename _Tp> static
inline void elbp_(InputArray _src, OutputArray _dst, int radius, int neighbors) {
    //get matrices
    Mat src = _src.getMat();
    // allocate memory for result
    _dst.create(src.rows-2*radius, src.cols-2*radius, CV_32SC1);
    Mat dst = _dst.getMat();
    // zero
    dst.setTo(0);
    for(int n=0; n<neighbors; n++) {
        // sample points
        float x = static_cast<float>(radius * cos(2.0*CV_PI*n/static_cast<float>(neighbors)));
        float y = static_cast<float>(-radius * sin(2.0*CV_PI*n/static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 =      tx  * (1 - ty);
        float w3 = (1 - tx) *      ty;
        float w4 =      tx  *      ty;
        // iterate through your data
        for(int i=radius; i < src.rows-radius;i++) {
            for(int j=radius;j < src.cols-radius;j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1*src.at<_Tp>(i+fy,j+fx) + w2*src.at<_Tp>(i+fy,j+cx) + w3*src.at<_Tp>(i+cy,j+fx) + w4*src.at<_Tp>(i+cy,j+cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<int>(i-radius,j-radius) += ((t > src.at<_Tp>(i,j)) || (std::abs(t-src.at<_Tp>(i,j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors);
static Mat elbp(InputArray src, int radius, int neighbors);

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false);

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed);

static Mat spatial_histogram(InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/);

void calculate_histogram(InputArray src, int _radius, int _neighbors, int _grid_x, int _grid_y,
        Mat& lbp_image, Mat& histogram_image);

#endif  /* HISTOGRAM_CALCULATE_HPP */

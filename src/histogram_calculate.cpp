
#include "histogram_calculate.hpp"

void calculate_histogram(cv::InputArray src, int _radius, int _neighbors, int _grid_x, int _grid_y, cv::Mat &lbp_image, cv::Mat &histogram_image){
    // calculate lbp image
    lbp_image = elbp(src, _radius, _neighbors);
    Mat tmp = lbp_image.clone();
    // get spatial histogram from this lbp image
    histogram_image = spatial_histogram(
                tmp, /* lbp_image */
                static_cast<int>(std::pow(2.0, static_cast<double>(_neighbors))), /* number of possible patterns */
                _grid_x, /* grid size x */
                _grid_y, /* grid size y */
                true);
    // add to templates
}

cv::Mat spatial_histogram(cv::InputArray _src, int numPatterns, int grid_x, int grid_y, bool)
{
    Mat src = _src.getMat();
    // calculate LBP patch size
    int width = src.cols/grid_x;
    int height = src.rows/grid_y;
    // allocate memory for the spatial histogram
    Mat result = Mat::zeros(grid_x * grid_y, numPatterns, CV_32FC1);
    // return matrix with zeros if no data was given
    if(src.empty())
        return result.reshape(1,1);
    // initial result_row
    int resultRowIdx = 0;
    // iterate through grid
    for(int i = 0; i < grid_y; i++) {
        for(int j = 0; j < grid_x; j++) {
            Mat src_cell = Mat(src, Range(i*height,(i+1)*height), Range(j*width,(j+1)*width));
            Mat cell_hist = histc(src_cell, 0, (numPatterns-1), true);
            // copy to the result matrix
            Mat result_row = result.row(resultRowIdx);
            cell_hist.reshape(1,1).convertTo(result_row, CV_32FC1);
            // increase row count in result matrix
            resultRowIdx++;
        }
    }
    // return result as reshaped feature vector
    return result.reshape(1,1);
}

cv::Mat histc(cv::InputArray _src, int minVal, int maxVal, bool normed)
{
    Mat src = _src.getMat();
    switch (src.type()) {
    case CV_8SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_8UC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    case CV_16SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_16UC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    case CV_32SC1:
        return histc_(Mat_<float>(src), minVal, maxVal, normed);
        break;
    case CV_32FC1:
        return histc_(src, minVal, maxVal, normed);
        break;
    }
    //    CV_Error(Error::StsUnmatchedFormats, "This type is not implemented yet.");
    std::cout << "Error: " << "This type is not implemented yet." << std::endl;
}

cv::Mat histc_(const cv::Mat &src, int minVal, int maxVal, bool normed)
{
    Mat result;
    // Establish the number of bins.
    int histSize = maxVal-minVal+1;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal+1) };
    const float* histRange = { range };
    // calc histogram
    calcHist(&src, 1, 0, Mat(), result, 1, &histSize, &histRange, true, false);
    // normalize
    if(normed) {
        result /= (int)src.total();
    }
    return result.reshape(1,1);
}

void elbp(cv::InputArray src, cv::OutputArray dst, int radius, int neighbors)
{
    int type = src.type();
    switch (type) {
    case CV_8SC1:   elbp_<char>(src,dst, radius, neighbors); break;
    case CV_8UC1:   elbp_<unsigned char>(src, dst, radius, neighbors); break;
    case CV_16SC1:  elbp_<short>(src,dst, radius, neighbors); break;
    case CV_16UC1:  elbp_<unsigned short>(src,dst, radius, neighbors); break;
    case CV_32SC1:  elbp_<int>(src,dst, radius, neighbors); break;
    case CV_32FC1:  elbp_<float>(src,dst, radius, neighbors); break;
    case CV_64FC1:  elbp_<double>(src,dst, radius, neighbors); break;
    default:
        String error_msg = format("Using Original Local Binary Patterns for feature extraction only works on single-channel images (given %d). Please pass the image data as a grayscale image!", type);
        std::cout << "Error: " << error_msg << std::endl;
        //        CV_Error(Error::StsNotImplemented, error_msg);
        break;
    }
}

cv::Mat elbp(cv::InputArray src, int radius, int neighbors) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}

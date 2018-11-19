#include "opencv4/opencv2/objdetect/objdetect.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/imgcodecs.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/objdetect.hpp"
//#include "opencv2/contrib/contrib.hpp"

#include <iostream>
#include <stdio.h>
#include <dirent.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <map>

#include "xlnt/xlnt.hpp"

using namespace std;
using namespace cv;

/** Function Headers */
//void detectAndDisplay(Mat &frame );

/** Global variables "/home/mike/workspace/tmp/opencv_test/data_lbp/cascade.xml";//*/
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";//"haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";//"haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String mouth_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";
String nose_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier mouth_cascade;
CascadeClassifier nose_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

const std::string info_file{"info.dat"};
const int64_t wait_time_ms = 100;



struct face_data{
    std::string path_to_file;
    Mat image;
    std::vector<Rect> faces;
    std::vector<Rect> eyes;
    std::vector<Rect> mouths;
    std::vector<Rect> noses;
};



cv::Mat get_image_parts(face_data data);


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

static void elbp(InputArray src, OutputArray dst, int radius, int neighbors)
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
static Mat elbp(InputArray src, int radius, int neighbors) {
    Mat dst;
    elbp(src, dst, radius, neighbors);
    return dst;
}

static Mat
histc_(const Mat& src, int minVal=0, int maxVal=255, bool normed=false)
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

static Mat histc(InputArray _src, int minVal, int maxVal, bool normed)
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

static Mat spatial_histogram(InputArray _src, int numPatterns,
                             int grid_x, int grid_y, bool /*normed*/)
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

void calculate_histogram(InputArray src, int _radius, int _neighbors, int _grid_x, int _grid_y,
        Mat& lbp_image, Mat& histogram_image){
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


Mat load_image(std::string image_path) {
  Mat image;
  image = imread(image_path.c_str(), IMREAD_GRAYSCALE); // Read the file
  if( image.empty() )                      // Check for invalid input
  {
    cout <<  "Could not open or find the image [" << image_path << "]" << std::endl ;
    return image;
  }

//  cv::Mat frame_gray;

//  cv::cvtColor(image, frame_gray, CV_BGR2GRAY);
//  cv::equalizeHist(frame_gray, frame_gray);
//  return frame_gray;
  return image;
}

void process_reference(std::string reference_path){
    auto reference = load_image(reference_path);
    if (reference.empty()) return;
}

std::string buffer{};
fstream out_stream;
void drop_buffer(){
  buffer = std::string{};
}
void write_to_file(){
  out_stream << buffer;
  drop_buffer();
}
void writ_to_buffer(std::string data) {
  buffer += data;
}


static void set_color_corr(double value, xlnt::cell&& cell){
    // colors
    xlnt::fill great_match = xlnt::fill::solid(xlnt::color(xlnt::rgb_color(128,0,128)));
    xlnt::fill match = xlnt::fill::solid(xlnt::color(xlnt::rgb_color(255,20,147)));
    xlnt::fill looks_same = xlnt::fill::solid(xlnt::color(xlnt::rgb_color(218,112,214)));
    xlnt::fill more_than_half_smae = xlnt::fill::solid(xlnt::color(xlnt::rgb_color(255,182,193)));

    if ( 0.9 <= value) {
        cell.fill(great_match);
    } else if (0.75 <= value) {
        cell.fill(match);
    } else if (0.6 <= value) {
        cell.fill(looks_same);
    } else if (0.5 <= value) {
        cell.fill(more_than_half_smae);
    }
}


//void cascade_demo( int, void* )
//{
//    int blockSize = 2;
//    int apertureSize = 3;
//    double k = 0.04;
//    Mat dst = Mat::zeros( src.size(), CV_32FC1 );
//    cornerHarris( src_gray, dst, blockSize, apertureSize, k );
//    Mat dst_norm, dst_norm_scaled;
//    normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
//    convertScaleAbs( dst_norm, dst_norm_scaled );
//    for( int i = 0; i < dst_norm.rows ; i++ )
//    {
//        for( int j = 0; j < dst_norm.cols; j++ )
//        {
//            if( (int) dst_norm.at<float>(i,j) > thresh )
//            {
//                circle( dst_norm_scaled, Point(j,i), 5,  Scalar(0), 2, 8, 0 );
//            }
//        }
//    }
//    namedWindow( corners_window );
//    imshow( corners_window, dst_norm_scaled );
//}

struct cascade_settings{
    face_data data;

    double scale_face;
    double scale_eyes;
    double scale_mouth;
    double scale_nose;

    int neighbors_face;
    int neighbors_eyes;
    int neighbors_mouth;
    int neighbors_nose;
    
    Size min_size_face;
    Size min_size_eyes;
    Size min_size_mouth;
    Size min_size_nose;

    Size max_size_face;
    Size max_size_eyes;
    Size max_size_mouth;
    Size max_size_nose;
};



void demo_face_scale( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    if (pos == 0) pos = 1;
    set->scale_face = 1.0 +  pos/10.0;
    set->data.faces.clear();
    cv::imshow(set->data.path_to_file, set->data.image);
    // std::cout << "faces: " << set->data.faces.size() << std::endl;
    face_cascade.detectMultiScale( set->data.image, set->data.faces, set->scale_face, set->neighbors_face,
                                   CASCADE_SCALE_IMAGE, set->min_size_face );
    // std::cout << "faces: " << set->data.faces.size() << std::endl;
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
    // cv::imwrite("test_" + std::to_string(pos) + ".png", get_image_parts(set->data));
}
void demo_face_neighbors( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->neighbors_face = pos;
    set->data.faces.clear();
    cv::imshow(set->data.path_to_file, set->data.image);
    face_cascade.detectMultiScale( set->data.image, set->data.faces, set->scale_face, set->neighbors_face,
                                   CASCADE_SCALE_IMAGE, set->min_size_face );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_face_min_size( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->min_size_face = cv::Size(pos, pos);
    set->data.faces.clear();
    face_cascade.detectMultiScale( set->data.image, set->data.faces, set->scale_face, set->neighbors_face,
                                   CASCADE_SCALE_IMAGE, set->min_size_face );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_face_max_size( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->min_size_face = cv::Size(pos, pos);
    face_cascade.detectMultiScale( set->data.image, set->data.faces, set->scale_face, set->neighbors_face,
                                   CASCADE_SCALE_IMAGE, set->min_size_face );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}


// eyes
void demo_eyes_scale( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    if (pos == 0) pos = 1;
    set->scale_eyes = 1.0 +  pos/10.0;
    set->data.eyes.clear();
    eyes_cascade.detectMultiScale( set->data.image, set->data.eyes, set->scale_eyes, set->neighbors_eyes,
                                   CASCADE_SCALE_IMAGE, set->min_size_eyes );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_eyes_neighbors( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->neighbors_eyes = pos;
    set->data.eyes.clear();
    cv::imshow(set->data.path_to_file, set->data.image);
    eyes_cascade.detectMultiScale( set->data.image, set->data.eyes, set->scale_eyes, set->neighbors_eyes,
                                   CASCADE_SCALE_IMAGE, set->min_size_eyes );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_eyes_min_size( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->min_size_eyes = cv::Size(pos, pos);
    set->data.eyes.clear();
    eyes_cascade.detectMultiScale( set->data.image, set->data.eyes, set->scale_eyes, set->neighbors_eyes,
                                   CASCADE_SCALE_IMAGE, set->min_size_eyes );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}


// mouth
void demo_mouth_scale( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    if (pos == 0) pos = 1;
    set->scale_eyes = 1.0 +  pos/10.0;
    set->data.mouths.clear();
    mouth_cascade.detectMultiScale( set->data.image, set->data.mouths, set->scale_eyes, set->neighbors_mouth,
                                   CASCADE_SCALE_IMAGE, set->min_size_mouth );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_mouth_neighbors( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->neighbors_mouth = pos;
    set->data.mouths.clear();
    cv::imshow(set->data.path_to_file, set->data.image);
    mouth_cascade.detectMultiScale( set->data.image, set->data.mouths, set->scale_eyes, set->neighbors_mouth,
                                   CASCADE_SCALE_IMAGE, set->min_size_mouth );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_mouth_min_size( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->min_size_mouth = cv::Size(pos, pos);
    set->data.mouths.clear();
    mouth_cascade.detectMultiScale( set->data.image, set->data.mouths, set->scale_eyes, set->neighbors_eyes,
                                   CASCADE_SCALE_IMAGE, set->min_size_eyes );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}

// nose
void demo_nose_scale( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    if (pos == 0) pos = 1;
    set->scale_eyes = 1.0 +  pos/10.0;
    set->data.noses.clear();
    nose_cascade.detectMultiScale( set->data.image, set->data.noses, set->scale_eyes, set->neighbors_nose,
                                   CASCADE_SCALE_IMAGE, set->min_size_nose );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_nose_neighbors( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->neighbors_nose = pos;
    set->data.noses.clear();
    cv::imshow(set->data.path_to_file, set->data.image);
    nose_cascade.detectMultiScale( set->data.image, set->data.noses, set->scale_eyes, set->neighbors_nose,
                                   CASCADE_SCALE_IMAGE, set->min_size_nose );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}
void demo_nose_min_size( int pos, void* userdata){
    cascade_settings* set = static_cast<cascade_settings*> (userdata);
    set->min_size_nose = cv::Size(pos, pos);
    set->data.noses.clear();
    nose_cascade.detectMultiScale( set->data.image, set->data.noses, set->scale_eyes, set->neighbors_eyes,
                                   CASCADE_SCALE_IMAGE, set->min_size_eyes );
    cv::imshow(set->data.path_to_file, get_image_parts(set->data));
}


void show_cascade_detectors(std::string path, cascade_settings& set){
    namedWindow( path , WINDOW_NORMAL);

//    double scale_face = 1.0;
    int scale_part_face = 2;
    int scale_part_eyes = 2;
    int scale_part_mouth = 2;
    int scale_part_nose = 2;
    int neighbors_face = 4;
    int neighbors_eyes = 4;
    int neighbors_mouth = 4;
    int neighbors_nose = 4;
    int min_size_face = 10;
    int min_size_eyes = 10;
    int min_size_mouth = 10;
    int min_size_nose = 10;


    set.scale_face = scale_part_face;
    set.scale_eyes = scale_part_eyes;
    set.scale_mouth = scale_part_mouth;
    set.scale_nose = scale_part_nose;

    set.neighbors_face = neighbors_face;
    set.neighbors_eyes = neighbors_eyes;
    set.neighbors_mouth = neighbors_mouth;
    set.neighbors_nose = neighbors_nose;

    set.min_size_face = cv::Size(min_size_face, min_size_face);
    set.min_size_eyes = cv::Size(min_size_eyes, min_size_eyes);
    set.min_size_mouth = cv::Size(min_size_mouth, min_size_mouth);
    set.min_size_nose = cv::Size(min_size_nose, min_size_nose);

    // face
    createTrackbar("[face] scale: ", path, &scale_part_face, 10, demo_face_scale, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[face] scale: ", 1);
    createTrackbar("[face] neighbours: ", path, &neighbors_face, 16, demo_face_neighbors, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[face] neighbours: ", 1);
    createTrackbar("[face] min size: ", path, &min_size_face, 50, demo_face_min_size, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[face] min size: ", 1);

    // eyes
    createTrackbar("[eyes] scale: ", path, &scale_part_eyes, 10, demo_eyes_scale, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[eyes] scale: ", 1);
    createTrackbar("[eyes] neighbours: ", path, &neighbors_eyes, 16, demo_eyes_neighbors, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[eyes] neighbours: ", 1);
    createTrackbar("[eyes] min size: ", path, &min_size_eyes, 50, demo_eyes_min_size, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[eyes] min size: ", 1);

    // mouth
    createTrackbar("[mouth] scale: ", path, &scale_part_mouth, 10, demo_mouth_scale, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[mouth] scale: ", 1);
    createTrackbar("[mouth] neighbours: ", path, &neighbors_mouth, 16, demo_mouth_neighbors, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[mouth] neighbours: ", 1);
    createTrackbar("[mouth] min size: ", path, &min_size_mouth, 50, demo_mouth_min_size, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[mouth] min size: ", 1);

    // nose
    createTrackbar("[nose] scale: ", path, &scale_part_nose, 10, demo_nose_scale, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[nose] scale: ", 1);
    createTrackbar("[nose] neighbours: ", path, &neighbors_nose, 16, demo_nose_neighbors, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[nose] neighbours: ", 1);
    createTrackbar("[nose] min size: ", path, &min_size_nose, 50, demo_nose_min_size, static_cast<void*> (std::addressof(set)));
    setTrackbarMin (path, "[nose] min size: ", 1);

    waitKey();
}


face_data get_face_data(std::string path){
    face_data data;
    data.image = load_image(path);
    data.path_to_file = path;

    cascade_settings set;
    set.min_size_face = cv::Size(10,10);
    set.neighbors_face = 3;
    set.scale_face = 1.1;
    set.data = data;

    show_cascade_detectors(path, set);

//    face_cascade.detectMultiScale( data.image, data.faces, 1.2, 4, CASCADE_SCALE_IMAGE, Size(30, 30) );
//    eyes_cascade.detectMultiScale( data.image, data.eyes, 1.2, 4, CASCADE_SCALE_IMAGE, Size(5, 5) );
//    mouth_cascade.detectMultiScale( data.image, data.mouths, 1.2, 4, CASCADE_SCALE_IMAGE, Size(30, 30) );
//    nose_cascade.detectMultiScale( data.image, data.noses, 1.2, 4, CASCADE_SCALE_IMAGE, Size(30, 30) );

    return set.data;
}

cv::Mat get_image_parts(face_data data) {

//    (128,0,128)
//    (255,20,147)
//    (218,112,214)
//    (255,182,193)

    cv::Mat image = data.image.clone();
    std::for_each(data.faces.begin(), data.faces.end(), [&image] (auto rect){
        cv::rectangle(image, rect, cv::Scalar(128,0,128));
    });
    std::for_each(data.eyes.begin(), data.eyes.end(), [&image] (auto rect){
        cv::rectangle(image, rect, cv::Scalar(255,20,147));
    });
    std::for_each(data.mouths.begin(), data.mouths.end(), [&image] (auto rect){
        cv::rectangle(image, rect, cv::Scalar(218,112,214));
    });
    std::for_each(data.noses.begin(), data.noses.end(), [&image] (auto rect){
        cv::rectangle(image, rect, cv::Scalar(255,182,193));
    });

    return image;
}

struct orb_settings{
    int nfeatures=50;
    int nlevels=4;
    int edgeThreshold=30;
    int WTA_K=2;
    float scaleFactor=1.2f;
    int firstLevel=0;
};


struct orb_result_data{
    orb_settings set;
    std::string window;
    cv::Mat image;
    cv::Mat descriptor;
    vector<KeyPoint> points;
};

cv::Mat gen_orb_image(cv::Mat img1, cv::Ptr<cv::ORB>& orb, cv::Mat& descriptor, vector<KeyPoint>& points){
    orb->detect(img1, points);
    orb->compute(img1, points, descriptor);

    std::cout << "Points size: [" << points.size() << "]" << std::endl;
    cv::Mat points_img = img1.clone();
    if (points.size()) {
        cv::drawKeypoints(img1, points, points_img);
//        cv::imwrite("points.png", points_img);
    }
    return points_img;
}

cv::Ptr<cv::ORB> create_orb(orb_settings set){
    auto orb = cv::ORB::create(
        set.nfeatures,
        set.scaleFactor,
        set.nlevels,
        set.edgeThreshold,
        set.firstLevel,
        set.WTA_K,
        ORB::HARRIS_SCORE,
        set.edgeThreshold // same as edgeThreshold
    );
    return orb;
}

//using nfeatures_type = int;
//typedef int scaleFactor_type;
//typedef int nlevels_type;
//using nlevels_type = int;
//using edgeThreshold_type = int;
//using firstLevel_type = int;
//using WTA_K_type = int;

//class nfeatures_type;
class nfeatures_type
{
public:
    nfeatures_type() {}
};
class scaleFactor_type
{
public:
    scaleFactor_type() {}
};
class nlevels_type
{
public:
    nlevels_type() {}
};
class edgeThreshold_type
{
public:
    edgeThreshold_type() {}
};
class firstLevel_type
{
public:
    firstLevel_type() {}
};
class WTA_K_type
{
public:
    WTA_K_type() {}
};

template <typename T>
void demo_orb(int pos, void* userdata){
    orb_result_data* data = static_cast<orb_result_data*> (userdata);
    std::cout << "nf: [" << data->set.nfeatures
              << "], scale: [" << data->set.scaleFactor
              << "], nlevels: [" << data->set.nlevels
              << "], edgeThreshold: [" << data->set.edgeThreshold
              << "], WTA_K: [" << data->set.WTA_K << "]" << std::endl;
    // apply pos to data
    if (std::is_same<T, nfeatures_type>::value){
        data->set.nfeatures = (pos != 0) ? pos : 1;
    } else if (std::is_same<T, scaleFactor_type>::value){
        float tmp = (pos != 0) ? pos : 1;
        data->set.scaleFactor = 1.0f + (tmp)/10.0f;
    } else if (std::is_same<T, nlevels_type>::value){
        data->set.nlevels = (pos != 0) ? pos : 1;
    } else if (std::is_same<T, edgeThreshold_type>::value){
        data->set.edgeThreshold = (pos < 3) ? pos : 3;
    } else if (std::is_same<T, firstLevel_type>::value){
        data->set.WTA_K = (pos != 0) ? pos : 1;
    } else {
        throw std::invalid_argument("Wrong use of function template [demo_orb]");
    }

    // create orgb
    auto orb = create_orb(data->set);
    // detect points
//    data->descriptor;
//    cv::Mat descriptor;
//    vector<KeyPoint> points;
    auto image = gen_orb_image(data->image, orb, data->descriptor, data->points);
    // pass image to window
    cv::imshow(data->window, image);
}

void set_orb_points(orb_result_data& data){
    std::string window = "orb";
    data.window = window;
    namedWindow(window , WINDOW_NORMAL);

    int nfeatures=50;
    int nlevels=4;
    int edgeThreshold=30;
    int WTA_K=2;
    int scale_part = 2;

    float scaleFactor=1.2f;
    int firstLevel=0;

//    ORB::ScoreType scoreType=ORB::HARRIS_SCORE;
//    int patchSize=edgeThreshold;

    data.set.nfeatures = nfeatures;
    data.set.nlevels = nlevels;
    data.set.edgeThreshold = edgeThreshold;
    data.set.WTA_K = WTA_K;
    data.set.scaleFactor = scaleFactor;
    data.set.firstLevel = firstLevel;

    // face
    createTrackbar("nfeatures: ", window, &nfeatures, 100, demo_orb<nfeatures_type>, static_cast<void*> (std::addressof(data)));
    createTrackbar("nlevels: ", window, &nlevels, 20, demo_orb<nlevels_type>, static_cast<void*> (std::addressof(data)));
    createTrackbar("edgeThreshold: ", window, &edgeThreshold, 300, demo_orb<edgeThreshold_type>, static_cast<void*> (std::addressof(data)));
    createTrackbar("WTA_K: ", window, &WTA_K, 10, demo_orb<WTA_K_type>, static_cast<void*> (std::addressof(data)));
    createTrackbar("scaleFactor: ", window, &scale_part, 10, demo_orb<scaleFactor_type>, static_cast<void*> (std::addressof(data)));

    // setTrackbarMin (path, "[face] scale: ", 1);
    // setTrackbarMin (path, "[face] neighbours: ", 1);
    // setTrackbarMin (path, "[face] min size: ", 1);
    // setTrackbarMin (path, "[face] scale: ", 1);
    // setTrackbarMin (path, "[face] neighbours: ", 1);

    waitKey();
}



//void compare_two_images(cv::Mat orig_img1, cv::Mat orig_img2, cv::Ptr<cv::ORB>& orb){
//    Mat lbp_img1;
//    Mat histogram_img1;
//    calculate_histogram(orig_img1, 1, 8, 8, 8,
//                        lbp_img1, histogram_img1);

//    Mat lbp_img2;
//    Mat histogram_img2;
//    calculate_histogram(orig_img2, 1, 8, 8, 8,
//                        lbp_img2, histogram_img2);


//    cv::imwrite("orig.png", lbp_img1);

//    vector<cv::Mat> test_images;
//    test_images.push_back(lbp_img1);

//    cv::Mat img1;
//    lbp_img1.convertTo(img1, CV_8UC1);

//    int stype = img1.type();

//    std::cout << "type: [" << stype << "]" << std::endl;
//    std::cout << "channels: [" << CV_MAT_CN(stype) << "]" << std::endl;

//    cv::Mat descriptor;
//    vector<KeyPoint> points;
//    orb->detect(img1, points);
//    orb->compute(img1, points, descriptor);

//    std::cout << "Points size: [" << points.size() << "]" << std::endl;
//    if (points.size()) {
//        cv::Mat points_img = img1;

//        cv::drawKeypoints(img1, points, points_img);
//        cv::imwrite("points.png", points_img);
//    }

//    cv::Mat test_descriptor;
//    vector<KeyPoint> test_points;

//    cv::Mat img2;
//    lbp_img2.convertTo(img2, CV_8UC1);

//    orb->detect(img2, test_points);
//    orb->compute(img2, test_points, test_descriptor);
//    std::cout << "Points size: [" << test_points.size() << "]" << std::endl;
//    if (test_points.size()) {
//        cv::Mat points_img = img2;
//        cv::drawKeypoints(img2, test_points, points_img);
//        cv::imwrite("points_test.png", points_img);
//    }

//    cv::BFMatcher bf_matcher(NORM_HAMMING, true);
//    vector<DMatch> matches;
//    bf_matcher.match(descriptor, test_descriptor, matches);

//    std::sort(matches.begin(), matches.end(), [] (const DMatch& a, const DMatch& b) -> bool{
//        return a.distance > b.distance;
//    });

//    cv::Mat match_result;
//    cv::drawMatches(
//            img1, points,
//            img2, test_points,
//            matches, match_result
//    );

//    cv::imwrite("match_result.png", match_result);
//}

cv::Mat match_images(const cv::Mat& img1, const orb_result_data& data1,
                     const cv::Mat& img2, const orb_result_data& data2) {
    cv::BFMatcher bf_matcher(NORM_HAMMING, true);
    vector<DMatch> matches;
    bf_matcher.match(data1.descriptor, data2.descriptor, matches);

    std::sort(matches.begin(), matches.end(), [] (const DMatch& a, const DMatch& b) -> bool{
        return a.distance > b.distance;
    });

    cv::Mat match_result;
    cv::drawMatches(
            img1, data1.points,
            img2, data2.points,
            matches, match_result
    );

    return match_result;
}


cv::Mat compare_two_images(cv::Mat orig_image1, cv::Mat orig_image2){
    orb_result_data data1;
    Mat lbp_img1;
    Mat histogram_img1;
    calculate_histogram(orig_image1, 1, 8, 8, 8,
                        lbp_img1, histogram_img1);

    cv::Mat img1;
    lbp_img1.convertTo(img1, CV_8UC1);
    data1.image = img1;
    set_orb_points(data1);

    orb_result_data data2;
    Mat lbp_img2;
    Mat histogram_img2;
    calculate_histogram(orig_image2, 1, 8, 8, 8,
                        lbp_img2, histogram_img2);

    cv::Mat img2;
    lbp_img2.convertTo(img2, CV_8UC1);
    data2.image = img2;
    set_orb_points(data2);

    cv::Mat match_result = match_images(img1, data1, img2, data2);

    return match_result;
}



//void compare_face_data() {

//}



/** @function main */
int main( int argc, const char** argv )
{
    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/resources/mike/test0.png";
//    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/cube/0001.png";

//    std::string test_path = "/home/mike/workspace/tmp/histogram_matcher/cube/0003.png";
    std::string test_path = "/home/mike/workspace/tmp/histogram_matcher/resources/ilya/test50.png";
//    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/test.png";
    std::string path_to_images_collection = "/home/mike/workspace/tmp/histogram_matcher/resources/all_data.txt";
//    std::string path_to_images_collection = "/home/mike/worksls pace/tmp/opencv_crop/resource/data.txt";
    // Получить эталонную фотку
    if (argc>1) reference_path = argv[1];
    if (argc>2) path_to_images_collection = argv[2];

    std::fstream data_file(path_to_images_collection, std::ios_base::in);
    std::fstream out_file("match_result.txt", std::ios_base::out);
    std::vector<char> path_buf(256);

    Mat reference_image = load_image(reference_path);
    Mat reference_lbp_image;
    Mat reference_histogram_image;
    calculate_histogram(reference_image, 1, 8, 8, 8,
                        reference_lbp_image, reference_histogram_image);

//    cv::Mat test_img = Mat(reference_lbp_image.rows, reference_lbp_image.cols, CV_8UC1);
//    reference_lbp_image.convertTo(test_img, CV_8UC1);

//    int stype = test_img.type();
//    std::cout << "type: [" << stype << "]" << std::endl;
//    std::cout << "channels: [" << CV_MAT_CN(stype) << "]" << std::endl;
//    cv::imwrite("one_channel.png", test_img);


//    reference_image = reference_lbp_image;


//    auto orb = cv::ORB::create("test_orb");

//    auto facemark = cv::crea


    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face_cascade\n"); return -1; };
    if (!eyes_cascade.load(eyes_cascade_name)) {std::cout << "error load eye cascade" << std::endl;}
    if (!mouth_cascade.load(mouth_cascade_name)) {std::cout << "error load mouth cascade" << std::endl;}
    if (!nose_cascade.load(nose_cascade_name)) {std::cout << "error load nose cascade" << std::endl;}


    face_data img1 = get_face_data(reference_path);
    face_data img2 = get_face_data(test_path);

    cv::imwrite("prc1.png", get_image_parts(img1));
    cv::imwrite("prc2.png", get_image_parts(img2));

    int nfeatures=50;
    float scaleFactor=1.2f;
    int nlevels=4;
    int edgeThreshold=30;
    int firstLevel=0;
    int WTA_K=2;
    ORB::ScoreType scoreType=ORB::HARRIS_SCORE;
    int patchSize=edgeThreshold;

    auto orb = cv::ORB::create(
        nfeatures,
        scaleFactor,
        nlevels,
        edgeThreshold,
        firstLevel,
        WTA_K,
        scoreType,
        patchSize
    );


    if (img1.noses.size() && img2.noses.size()) {
        cv::Mat nose1 = img1.image(img1.noses[0]);
        cv::Mat nose2 = img2.image(img2.noses[0]);

        cv::Mat match_result = compare_two_images(nose1, nose2);

        cv::imwrite("match_result.png", match_result);
    } else {
        std::cout << "not found some parts" << std::endl;
    }


    return 0;
}

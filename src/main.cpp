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
#include <sys/stat.h>
#include <algorithm>

#include "xlnt/xlnt.hpp"
#include "points_processor.hpp"
#include "histogram_calculate.hpp"

#include "snNet.h"
#include "snOperator.h"
#include "snTensor.h"
#include "snType.h"

using namespace std;
using namespace cv;
namespace sn = SN_API;

/** Function Headers */
//void detectAndDisplay(Mat &frame );

/** Global variables "/home/mike/workspace/tmp/opencv_test/data_lbp/cascade.xml";//*/
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";//"haarcascade_frontalface_alt.xml";
String profile_face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";//"haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String mouth_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml";
String nose_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml";

String landmark_model_path = "/home/mike/workspace/utils/opencv/debug_modules/share/opencv4/testdata/cv/face/face_landmark_model.dat";

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


Mat load_image(std::string image_path) {
  Mat image;
  image = imread(image_path.c_str(), IMREAD_GRAYSCALE); // Read the file
  if( image.empty() )                      // Check for invalid input
  {
    cout <<  "Could not open or find the image [" << image_path << "]" << std::endl ;
//    return image;
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


vector< vector<Point2f> > test_landmark(const cv::Mat& image, vector<Rect>& faces) {
//    CascadeClassifier face_cascade;
//    face_cascade.load(cascade_name);
    Mat img = image.clone();
    Ptr<cv::face::Facemark> facemark = cv::face::createFacemarkKazemi();
    facemark->loadModel(landmark_model_path);
    std::cout << "Loaded model" << std::endl;

    cv::Mat test_imgae = img.clone();
//    resize(img,test_imgae,Size(460,460),0,0,INTER_LINEAR_EXACT);

    vector< vector<Point2f> > shapes;
    if (facemark->fit(test_imgae,faces,shapes))
    {
        for ( size_t i = 0; i < faces.size(); i++ )
        {
            cv::rectangle(img,faces[i],Scalar( 255, 0, 0 ));
        }
        for (unsigned long i=0;i<faces.size();i++){
            for(unsigned long k=20;k<shapes[i].size();k++) {
                cv::circle(img,shapes[i][k],1,cv::Scalar(0,0,255),FILLED);
                cv::putText(img, std::to_string(k), shapes[i][k],FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255));
            }
        }
        namedWindow("Detected_shape");
        imshow("Detected_shape", img);
        imwrite("shape.png", img);
        waitKey(10);
    }
    return shapes;
}

// SVM

struct point_data{
    cv::Point2f point;
    int point_class;
};

struct svm_data{
//    vector<point_data> points;
    vector<cv::Point2f> points;
    vector<int> point_classes;
};


cv::Mat create_svm_training_data(vector<cv::Point2f>& points){
    int dimension = 2;
    cv::Mat training_data(points.size(), dimension, CV_32F);

    for(int i = 0; i < points.size(); ++i) {
        training_data.at<float>(i, 0) = points[i].x;
        training_data.at<float>(i, 1) = points[i].y;
    }

    return training_data;
}



void apply_svm(cv::Mat& training_data, cv::Mat& lables, svm_data& data){
//    const int points_data_dimension = 2;

    // Train the SVM
    Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(training_data, cv::ml::ROW_SAMPLE, lables);

    // Data for visual representation
    int width = 512, height = 512;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Show the decision regions given by the SVM
    Vec3b green(0,255,0), blue(255,0,0);
    for (int i = 0; i < image.rows; i++)
    {
        for (int j = 0; j < image.cols; j++)
        {
            Mat sampleMat = (Mat_<float>(1,training_data.cols) << j,i);
            float response = svm->predict(sampleMat);
            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                image.at<Vec3b>(i,j)  = blue;
        }
    }
    // Show the training data
    int thickness = -1;

    int counter = 0;
    std::for_each(data.points.begin(), data.points.end(), [&image, &counter, &data](cv::Point2f point){
        if (data.point_classes[counter] == 1) {
            circle( image, point, 5, Scalar(  0x0A,   0x67,   0xA3), -1 );
        } else {
            circle( image, point, 5, Scalar(  0xFF,   0xB4,   0x8C), -1 );
        }
        counter++;
    });
//    circle( image, Point(255,  10), 5, Scalar(255, 255, 255), thickness );
//    circle( image, Point(501, 255), 5, Scalar(255, 255, 255), thickness );
//    circle( image, Point( 10, 501), 5, Scalar(255, 255, 255), thickness );
    // Show support vectors
    thickness = 2;
    Mat sv = svm->getUncompressedSupportVectors();
    for (int i = 0; i < sv.rows; i++)
    {
        const float* v = sv.ptr<float>(i);
        circle(image,  Point( (int) v[0], (int) v[1]), 6, Scalar(128, 128, 128), thickness);
    }
    imwrite("result.png", image);        // save the image
    imshow("SVM Simple Example", image); // show it to the user
    waitKey();
}

void apply_svm(svm_data& data){
    const int classes_data_dimension = 1;

    //    cv::Mat training_data(points_data_dimension, data.points.size(), CV_32F, data.points.data());
    cv::Mat training_data = create_svm_training_data(data.points);
    cv::Mat lables(classes_data_dimension, data.point_classes.size(), CV_32SC1, data.point_classes.data());

    apply_svm(training_data, lables, data);
}

cv::Mat unisize_image(cv::Mat input, cv::Rect aoi){
    cv::Mat image;
    cv::resize(input(aoi), image, cv::Size(400,400));
    return image;
}

cv::Mat get_landmarks(const std::string path, vector< vector<Point2f> >& shapes){
    cv::Mat test_image = imread(path);
    cascade_settings set;
    set.data.image = test_image;
    set.scale_face = 1.2;
    set.neighbors_face = 3;
    set.min_size_face = cv::Size(10,10);

    face_cascade.detectMultiScale( set.data.image, set.data.faces, set.scale_face, set.neighbors_face,
                                   CASCADE_SCALE_IMAGE, set.min_size_face );

    if (set.data.faces.size()) {
        cv::Rect face = set.data.faces[0];

        cv::Mat fece_rect_img = test_image.clone();
        cv::rectangle(fece_rect_img,face,Scalar( 255, 0, 0 ),2);
        cv::imwrite(path + ".face_rect.png", fece_rect_img);

        const float scale_factor = 1.2;
        const float scale_shift = (scale_factor - 1)/2;

        std::cout << "image : "
                  << test_image.cols << " | "
                  << test_image.rows << " | "
                  << std::endl;

        std::cout << "face: "
                  << face.x << " | "
                  << face.y << " | "
                  << face.width << " | "
                  << face.height << " | "
                  << std::endl;

        int pos_x = face.x - face.width*scale_shift;
        int pos_y = face.y - face.height*scale_shift;
        int width = face.width*scale_factor;
        int height = face.height*scale_factor;

        if (pos_y < 1) pos_y = 1;
        if (pos_x < 1) pos_x = 1;

        if ((width + pos_x) > test_image.cols ) width = test_image.cols-1 - pos_x;
        if ((height + pos_y) > test_image.rows ) height = test_image.rows-1 - pos_y;

        std::cout << "face sized: "
                  << pos_x << " | "
                  << pos_y << " | "
                  << width << " | "
                  << height << " | "
                  << std::endl;
        cv::Rect crop_part(pos_x, pos_y, width, height);

        set.data.faces[0].x = face.width*scale_shift;
        set.data.faces[0].y = face.height*scale_shift;

        cv::Mat crop = unisize_image(test_image, crop_part);//test_image(crop_part);

        cv::imwrite(path + ".face.png", crop);

        set.data.faces.clear();
        std::cout << "detect scaled face" << std::endl;
        face_cascade.detectMultiScale( crop, set.data.faces, set.scale_face, set.neighbors_face,
                                       CASCADE_SCALE_IMAGE, set.min_size_face );

        namedWindow("Detected_shape");

//        imshow("Detected_shape",crop);
//        waitKey(0);
//        cv::rectangle(crop,set.data.faces[0],Scalar( 255, 0, 0 ));
//        imshow("Detected_shape",crop);
//        waitKey(0);

        shapes = test_landmark(crop, set.data.faces);
        cv::Mat img = crop.clone();

        for (unsigned long i=0;i<set.data.faces.size();i++){
            for(unsigned long k=0;k<shapes[i].size();k++) {
                cv::circle(img,shapes[i][k],2,cv::Scalar(255,0,0),FILLED);
//                cv::putText(img, std::to_string(k), shapes[i][k],FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255));
            }
            cv::imwrite(path + ".points68.png", img);
        }

        cv::Mat selected_points = crop.clone();
        std::vector<size_t> pts = {31, 32, 33, 34, 35,
            27, 28, 29, 30,
            21, 22,
            48, 54, 59, 55, 49, 53,
            36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47};
        for (auto& el: pts) {
            cv::circle(selected_points, shapes[0][el],2,cv::Scalar(255,0,0),FILLED);
        }
        cv::imwrite(path + ".points29.png", selected_points);

        cv::Mat rectangle_points = crop.clone();
        cv::Mat lbp_readey;
        cv::cvtColor(crop, lbp_readey, COLOR_RGB2GRAY);
        cv::Mat lbp_rectangles;
        cv::Mat hist_rectangles;
        calculate_histogram(lbp_readey, 1, 8, 8, 8,
                lbp_rectangles, hist_rectangles);
        cv::imwrite(path + ".lbp.png", lbp_rectangles);
        int counter = 0;
        for (auto& el: pts) {
            int x = shapes[0][el].x - 16;
            int y = shapes[0][el].y - 16;
            cv::Rect rect(x,y,32,32);
            cv::imwrite(path + ".lbp_rect_"+ std::to_string(counter) + ".png", lbp_rectangles(rect));
            counter++;
        }
        for (auto& el: pts) {
            int x = shapes[0][el].x - 16;
            int y = shapes[0][el].y - 16;
            cv::Rect rect(x,y,32,32);
            cv::rectangle(rectangle_points,rect,Scalar( 255, 0, 0 ),2);
            cv::rectangle(lbp_rectangles,rect,Scalar( 0, 0, 0 ),2);
            cv::circle(rectangle_points, shapes[0][el],2,cv::Scalar(255,0,0),FILLED);
        }
        cv::imwrite(path + ".points_rectangles.png", rectangle_points);
        cv::imwrite(path + ".lbp_rectangles.png", lbp_rectangles);

        return crop;
    }
    return cv::Mat();
}

cv::Mat get_homography_matrix(std::vector<cv::Point2f> src, std::vector<cv::Point2f> dest){
    cv::Mat h = cv::findHomography(src, dest);
//    cout << "H:\n" << h << endl;
    return h;
}


std::vector<std::string> get_paths(const std::string& data, int limit){
    std::fstream data_file(data, std::ios_base::in);
    std::vector<char> path_buf(256);
    std::vector<std::string> dataset_paths;
    int counter = 0;
    while (data_file.getline(path_buf.data(), path_buf.size())){
        std::string path(path_buf.data());
        Mat image = load_image(path);
        if (image.empty()) continue;
        dataset_paths.push_back(path);
        if (counter >= limit) {
            break;
        }
        counter++;
    }
    return dataset_paths;
}

std::vector<std::string> get_paths(const std::string& data){
    return get_paths(data, 5);
}

vector< face_matcher::training_data > get_training_data(const std::vector<std::string>& dataset_paths, face_matcher::match_index match_index){
    vector< face_matcher::training_data > training_data;
    vector< vector<Point2f> > shapes;
    int test = dataset_paths.size();
//    std::for_each(dataset_paths.begin(), dataset_paths.end(), [&shapes, &training_data, &match_index] (auto path){
    for (auto& path : dataset_paths) {
        vector< vector<Point2f> > shapes_possible;
        cv::Mat res_image = get_landmarks(path, shapes_possible);
        if (shapes_possible.size()) {
            shapes.push_back(shapes_possible[0]);
            points_processor processor(res_image, shapes_possible[0]);
            processor.generate_first_layer();
            face_matcher::training_data tmp;
            tmp.value = processor.get_data();
            tmp.index = match_index;
            training_data.push_back(tmp);
        }
    }
    return training_data;
}


cv::Mat load_test_sample(const std::string& path){
    vector< vector<Point2f> > shapes_possible;
    cv::Mat res_image = get_landmarks(path, shapes_possible);
    cv::Mat res;
    if (shapes_possible.size() && !res_image.empty()) {
        points_processor processor(res_image, shapes_possible[0]);
        processor.generate_first_layer();



        res = face_matcher::construct_data_vector(processor.get_data());
    }
    return res;
}

inline bool exists_test3(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

void matrix_to_vector(const cv::Mat& orig, cv::Mat& dest){
    std::vector<cv::Mat> rows;

    for (int i = 0; i < orig.rows; ++i) {
        rows.push_back(orig.row(i));
    }

    cv::hconcat(rows, dest);
}



void remove_tested(std::vector<std::string>& paths, const std::string& path_to_data) {
    std::fstream in_file(path_to_data, std::ios_base::in);
    std::vector<char> path_buf(256);
    while(in_file.getline(path_buf.data(), path_buf.size())){
        std::string path(path_buf.data());
        path = path.substr(0, path.find(' '));
        paths.erase( std::remove( paths.begin(), paths.end(), path), paths.end() );
    }
}


void run_cnn(){
//    sn::Net snet;

//    snet.addNode("Input", sn::Input(), "C1")
//        .addNode("C1", sn::Convolution(15, 0, sn::calcMode::CPU), "C2")
//        .addNode("C2", sn::Convolution(15, 0, sn::calcMode::CPU), "P1")
//        .addNode("P1", sn::Pooling(sn::calcMode::CPU), "FC1")
//        .addNode("FC1", sn::FullyConnected(128, sn::calcMode::CPU), "FC2")
//        .addNode("FC2", sn::FullyConnected(10, sn::calcMode::CPU), "LS")
//        .addNode("LS", sn::LossFunction(sn::lossType::softMaxToCrossEntropy), "Output");

////    snet.training()

//    sn::snLSize

//    cout << "Hello " <<  SN_API::versionLib() << endl;
}


/** @function main */
int main( int argc, const char** argv )
{
//    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/resources/mike/test0.png";
//    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/cube/0001.png";

//    std::string test_path = "/home/mike/workspace/tmp/histogram_matcher/cube/0003.png";
//    std::string test_path = "/home/mike/workspace/tmp/histogram_matcher/resources/ilya/test50.png";
//    std::string reference_path = "/home/mike/workspace/tmp/histogram_matcher/test.png";
//    std::string path_to_images_collection = "/home/mike/workspace/tmp/histogram_matcher/resources/all_data.txt";
//    std::string path_to_images_collection = "/home/mike/worksls pace/tmp/opencv_crop/resource/data.txt";
    // Получить эталонную фотку
//    if (argc>1) reference_path = argv[1];
//    if (argc>2) path_to_images_collection = argv[2];

//    std::fstream data_file(path_to_images_collection, std::ios_base::in);
    std::fstream out_file("match_result.txt", std::ios_base::out);
//    std::vector<char> path_buf(256);

//    Mat reference_image = load_image(reference_path);
//    Mat reference_lbp_image;
//    Mat reference_histogram_image;
//    calculate_histogram(reference_image, 1, 8, 8, 8,
//                        reference_lbp_image, reference_histogram_image);

//    std::string dataset_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/align_mike/data.txt";
//    std::fstream dataset_file(dataset_path, std::ios_base::in);
//    std::string debug_window_name = "debug";



//    cv::Mat test_img = Mat(reference_lbp_image.rows, reference_lbp_image.cols, CV_8UC1);
//    reference_lbp_image.convertTo(test_img, CV_8UC1);

//    int stype = test_img.type();
//    std::cout << "type: [" << stype << "]" << std::endl;
//    std::cout << "channels: [" << CV_MAT_CN(stype) << "]" << std::endl;
//    cv::imwrite("one_channel.png", test_img);


//    reference_image = reference_lbp_image;


//    auto orb = cv::ORB::create("test_orb");

//    auto facemark = cv::crea



//    cv::Mat tt;

//    cv::Mat row_1(1,5, CV_8UC1, cv::Scalar(1));
//    cv::Mat row_2(1,5, CV_8UC1, cv::Scalar(2));
//    std::cout << "size:" << row_1.rows << " | " << row_1.cols << std::endl;
//    std::cout << "size:" << row_2.rows << " | " << row_2.cols << std::endl;
//    for (int i = 0; i < 5; ++i) row_1.at<int>(0,i) = 1;
//    for (int i = 0; i < 5; ++i) row_2.at<int>(0,i) = 2;

//    std::vector<cv::Mat> vc;
//    vc.push_back(row_1);
//    vc.push_back(row_2);

////    cv::hconcat(row_1, tt);
////    std::cout << "Test image: " << tt << std::endl;
//    cv::hconcat(vc, tt);
//    std::cout << "Test image: " << tt << std::endl;
////    tt.push_back(row_1);
////    tt.push_back(row_2);
//    std::cout << "size:" << tt.rows << " | " << tt.cols << std::endl;

//    cv::Mat m(3,3, CV_8UC1, cv::Scalar(1));
//    std::cout << m << std::endl;
//    cv::Mat dest;
//    matrix_to_vector(m, dest);
//    std::cout << dest << std::endl;
//    dest.convertTo(dest, CV_32F, 1.0 / 255, 0);
//    std::cout << dest << std::endl;

//    return 0;


    //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face_cascade\n"); return -1; };
    if (!eyes_cascade.load(eyes_cascade_name)) {std::cout << "error load eye cascade" << std::endl;}
    if (!mouth_cascade.load(mouth_cascade_name)) {std::cout << "error load mouth cascade" << std::endl;}
    if (!nose_cascade.load(nose_cascade_name)) {std::cout << "error load nose cascade" << std::endl;}

//    namedWindow(debug_window_name);

    std::string path_to_test_data = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/align_data.txt";
    std::string path_to_result_data = "svm_result.txt";
    if (argc>1) path_to_test_data = argv[1];
    if (argc>2) path_to_result_data = argv[2];

//    std::fstream out_data(path_to_out_data, std::ios_base::out);


    std::string svm_path = "/home/mike/workspace/tmp/opencv_patterns_compare/build/svm_data.txt";
    if (exists_test3(svm_path)) {
        // load svm
        face_matcher matcher;
        matcher.load_svm(svm_path);
        std::string samples_file = path_to_test_data;
        std::vector<std::string> samples_paths = get_paths(samples_file, 500);
        remove_tested(samples_paths, path_to_result_data);
        std::fstream svm_result(path_to_result_data, std::ios_base::app);
        for (auto& path: samples_paths) {
            std::cout << path << std::endl;

            cv::Mat test_sample;
            try {
                test_sample = load_test_sample(path);
                std::string dump =path.append(".mat");
                cv::FileStorage file(dump, cv::FileStorage::WRITE);
                // Write to file!
                file << "Training" << test_sample;
            } catch (...) {
                std::cout << "Assert error: [" << path << "]" << std::endl;
                continue;
            }
            if (test_sample.empty()) continue;
            cv::Mat result;
            float res = matcher.predict(test_sample, result);

            std::cout << "predict res: " << res << std::endl;
            std::cout << "result matrix: " << result << std::endl;
            svm_result << path << " res: [" << res << "][" << result << "]" << std::endl;
        }

        return 0;
    }



    // Получить точки
//    face_matcher::training_data training_data;
    std::string mike_collection = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/align_mike/data.txt";
    std::vector<std::string> mike_paths = get_paths(mike_collection);
    vector< face_matcher::training_data > mikes_training_data = get_training_data(mike_paths, face_matcher::match_index::match);

//    std::string ilya_collection = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/align_ilya/data.txt";
    std::string ilya_collection = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/train_negative.txt";
    std::vector<std::string> ilya_paths = get_paths(ilya_collection, 30);
    vector< face_matcher::training_data > ilyas_training_data = get_training_data(ilya_paths, face_matcher::match_index::not_match);

    vector< face_matcher::training_data > training_data;

    training_data.insert(training_data.end(), mikes_training_data.begin(), mikes_training_data.end());
    training_data.insert(training_data.end(), ilyas_training_data.begin(), ilyas_training_data.end());

//    vector< vector<Point2f> > shapes;
//    face_matcher::match_index match_index = face_matcher::match_index::match;
//    std::for_each(dataset_paths.begin(), dataset_paths.end(), [&shapes, &training_data, &match_index] (auto path){
//        vector< vector<Point2f> > shapes_possible;
//        cv::Mat res_image = get_landmarks(path, shapes_possible);
//        if (shapes_possible.size()) {
//            shapes.push_back(shapes_possible[0]);
//            points_processor processor(res_image, shapes_possible[0]);
//            processor.generate_first_layer();
//            face_matcher::training_data tmp;
//            tmp.value = processor.get_data();
//            tmp.index = match_index;
//            training_data.push_back(tmp);
//        }
//    });

//    std::cout << "shapes: [" << shapes.size() << "]" << std::endl;

//    vector<Point2f> orig = shapes[0];
//    std::for_each(shapes.begin(), shapes.end(), [&orig] (auto dest) {
//        get_homography_matrix(orig, dest);
//    });

    // Теперь нужно эти точки привести к единому изображению
    // Пусть пока это будет первое изображение
    std::cout << "Dump training vectors ..." << std::endl;

    face_matcher mike_saver;
    mike_saver.fill_training_data(mikes_training_data);
    mike_saver.save_training_vectors("mikes_training_data.mat");

    face_matcher ilya_saver;
    ilya_saver.fill_training_data(ilyas_training_data);
    ilya_saver.save_training_vectors("ilya_training_data.mat");

    // Делаем обработку точек
    face_matcher matcher;
    matcher.fill_training_data(training_data);
    cv::Mat test = matcher.get_training_vectors();

    imwrite("test_vectors.png", test);

    std::cout << "test_vectors: " << test.cols << std::endl;
    std::cout << "test_vectors: " << test.rows << std::endl;
    out_file << test;

    std::vector<int> tmp = matcher.get_lables();
    for(auto& el: tmp) {
        std::cout << el << std::endl;
    }

    matcher.apply_svm("svm_data.txt");

    std::string sample_path = "/home/mike/workspace/tmp/opencv_patterns_compare/resources/align_mike/photo_14.png";
    cv::Mat test_sample = load_test_sample(sample_path);
    cv::Mat result;
    float res = matcher.predict(test_sample, result);

    std::cout << "predict res: " << res << std::endl;
    std::cout << "result matrix: " << result.rows << " | " << result.cols << std::endl;
    std::cout << "result matrix: " << result << std::endl;

    return 0;

//    face_data img1 = get_face_data(reference_path);
//    face_data img2 = get_face_data(test_path);

//    cv::imwrite("prc1.png", get_image_parts(img1));
//    cv::imwrite("prc2.png", get_image_parts(img2));

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


//    if (img1.noses.size() && img2.noses.size()) {
//        cv::Mat nose1 = img1.image(img1.noses[0]);
//        cv::Mat nose2 = img2.image(img2.noses[0]);

//        cv::Mat match_result = compare_two_images(nose1, nose2);

//        cv::imwrite("match_result.png", match_result);
//    } else {
//        std::cout << "not found some parts" << std::endl;
//    }

//    cv::Mat test_image = imread("/home/mike/workspace/tmp/wilton_video_handler/build/photo.png");
//    cascade_settings set;
//    set.data.image = test_image;
//    set.scale_face = 1.2;
//    set.neighbors_face = 3;
//    set.min_size_face = cv::Size(10,10);

//    face_cascade.detectMultiScale( set.data.image, set.data.faces, set.scale_face, set.neighbors_face,
//                                   CASCADE_SCALE_IMAGE, set.min_size_face );

//    if (set.data.faces.size()) {
//        cv::Rect face = set.data.faces[0];
//        int pos_x = face.x - face.width/2;
//        int pos_y = face.y - face.height/2;
//        int width = face.width*2;
//        int height = face.height*2;
//        cv::Rect crop_part(pos_x, pos_y, width, height);

//        set.data.faces[0].x = face.width/2;
//        set.data.faces[0].y = face.height/2;

//        cv::Mat crop = test_image(crop_part);
//        namedWindow("Detected_shape");
//        imshow("Detected_shape",crop);
//        waitKey(0);
////        cv::rectangle(crop,set.data.faces[0],Scalar( 255, 0, 0 ));
////        imshow("Detected_shape",crop);
////        waitKey(0);
//        test_landmark(crop, set.data.faces);
//    }


    svm_data svm;
    std::string path_1 = "/home/mike/workspace/tmp/opencv_patterns_compare/mike.png";
    vector< vector<Point2f> > shapes_1;
    std::cout << "Test image: " << path_1 << std::endl;
    get_landmarks(path_1, shapes_1);

    std::string path_2 = "/home/mike/workspace/dataset/downey_1.jpg";
    vector< vector<Point2f> > shapes_2;
    std::cout << "Test image: " << path_2 << std::endl;
    get_landmarks(path_2, shapes_2);

    std::cout << "Test svm: " << std::endl;
    if (shapes_1.size() && shapes_2.size()) {
        std::cout << "Size 1 image: " << shapes_1[0].size() << std::endl;
        std::cout << "Size 2 image: " << shapes_2[0].size() << std::endl;
        size_t size = shapes_1[0].size() + shapes_2[0].size();
        svm.points.reserve(size);
        svm.point_classes.reserve(size);

        svm.points.insert(svm.points.end(), shapes_1[0].begin(), shapes_1[0].end());
        svm.point_classes.insert(svm.point_classes.end(), shapes_1[0].size(), 1);

        svm.points.insert(svm.points.end(), shapes_2[0].begin(), shapes_2[0].end());
        svm.point_classes.insert(svm.point_classes.end(), shapes_2[0].size(), -1);

//        cv::findHomography();
        apply_svm(svm);
    }



//    svm.point_classes = vector<int>{1,1,1,-1,-1,-1};
//    svm.points = vector<cv::Point2f>{{1,1},{1,2},{1,3},{4,5},{5,6},{4,7}};
//    svm.points.push_back(cv::Point2f(10,10));
//    svm.points.push_back(cv::Point2f(10,20));
//    svm.points.push_back(cv::Point2f(10,30));
//    svm.points.push_back(cv::Point2f(400,350));
//    svm.points.push_back(cv::Point2f(450,360));
//    svm.points.push_back(cv::Point2f(450,370));
//    apply_svm(svm);


//    set.data.faces.push_back(cv::Rect(0, 0, set.data.image.cols, set.data.image.rows));


    return 0;
}

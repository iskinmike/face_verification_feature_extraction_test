#include "histogram_calculate.hpp"
#include "points_processor.hpp"

namespace {

#define SETUP_POINTS(input_points, dest) \
    std::for_each(input_points.begin(), input_points.end(), [this] (auto point){ \
        this->dest.push_back(this->points[point]);\
    });

#define SETUP_FIRST_LAYER(input_points) \
    std::for_each(input_points.begin(), input_points.end(), [this] (auto point){ \
        patches_data pt_dat; \
        pt_dat.center_pose = point; \
        pt_dat.layer_first = this->origin_image(get_patch(point, this->first_layer_offest)); \
        calculate_histogram(pt_dat.layer_first, 1, 8, 3, 3, \
                            pt_dat.layer_first_lbp, pt_dat.layer_first_hist); \
        std::cout << "orig: " << pt_dat.layer_first.rows << " | " << pt_dat.layer_first.cols << std::endl; \
        this->data.push_back(pt_dat); \
    });

//std::cout << "lbp: " << pt_dat.layer_first_lbp.rows << " | " << pt_dat.layer_first_lbp.cols << std::endl; \
//std::cout << "hist: " << pt_dat.layer_first_hist.rows << " | " << pt_dat.layer_first_hist.cols << std::endl; \
//cv::imwrite("test.png", pt_dat.layer_first_lbp);

    // std::std::vector<size_t> positions;
    std::vector<size_t> nose_positions = {31, /*32, 33, 34,*/ 35};
    std::vector<size_t> nose_bridge_positions = {27, /*28, 29, */30};
    std::vector<size_t> eyebrow_positions = {21, 22};
    std::vector<size_t> mouth_positions = {48, /*44, 59, 55, 49,*/ 53};
    std::vector<size_t> left_eye_positions = {36, /*37, 38, 39, 40,*/ 41};
    std::vector<size_t> right_eye_positions = {42, /*43, 44, 45, 46,*/ 47};
}

points_processor::points_processor(Mat image, std::vector<Point2f> shapes): points(shapes),
    first_layer_offest(4)
{
    cv::cvtColor(image, origin_image, COLOR_RGB2GRAY);
    SETUP_POINTS(nose_positions, nose_points);
    SETUP_POINTS(nose_bridge_positions, nose_bridge_points);
    SETUP_POINTS(eyebrow_positions, eyebrow_points);
    SETUP_POINTS(mouth_positions, mouth_points);
    SETUP_POINTS(left_eye_positions, left_eye_points);
    SETUP_POINTS(right_eye_positions, right_eye_points);
}

Rect points_processor::get_patch(Point2f center, int offset) {
    int x = center.x - offset;
    int y = center.y - offset;
    int width = offset*2;
    int height = offset*2;
    return cv::Rect(x, y, width, height);
}

void points_processor::generate_first_layer() {
//    // проходимся по точкам соотнесенным с изображением.
//    std::for_each(left_eye_points.begin(), left_eye_points.end(), [this] (auto point){
//        patches_data pt_dat;
//        pt_dat.center_pose = point;
//        // выбираем в указанной точке область.
//        pt_dat.layer_first = this->origin_image(get_patch(point, this->first_layer_offest));
//        // для этой области вычисляем LBP.
//        calculate_histogram(pt_dat.layer_first, 1, 8, 8, 8,
//                            pt_dat.layer_first_lbp, pt_dat.layer_first_hist);
//        // Сохраняем.
//        this->data.push_back(pt_dat);
//    });

    SETUP_FIRST_LAYER(nose_points);
    SETUP_FIRST_LAYER(nose_bridge_points);
    SETUP_FIRST_LAYER(eyebrow_points);
    SETUP_FIRST_LAYER(mouth_points);
    SETUP_FIRST_LAYER(left_eye_points);
    SETUP_FIRST_LAYER(right_eye_points);

//    patches_data tmp = data[0];
//    int size = tmp.layer_first_hist.cols;
//    cv::Mat lbp = tmp.layer_first_lbp.clone();
    /// Set the ranges ( for B,G,R) )
//    float range[] = { 0, 256 } ;
//    const float* histRange = { range };
    /// Establish the number of bins
//    int histSize = 256;

//    std::vector<int> histSize;
//    histSize.push_back(6);
//    std::vector<float> ranges;
//    ranges.push_back(0);
//    ranges.push_back(6);
//    std::vector<int> channels;
//    channels.push_back(1);
//    cv::Mat hist;

//    int channels = 0;
//    int histSize = 6;
//    float range[] = {0,1};
//    const float* histRange = { range };
//    float** ranges = static_cast<float**> (&range);

//    void cv::calcHist( const Mat* images, int nimages, const int* channels,
//                   InputArray _mask, SparseMat& hist, int dims, const int* histSize,
//                   const float** ranges, bool uniform, bool accumulate )
//    std::cout << "lbp channels: " << lbp.channels() << std::endl;
//    cv::calcHist(&lbp, 1, &channels, cv::Mat(), hist,
//                 1, &histSize, &histRange, true, false);
//    cv::calcHist(lbp, channels, cv::Mat(), hist, histSize, ranges);
//    std::cout << "val: " << hist.cols << std::endl;
//    std::cout << "val: " << hist.rows << std::endl;
//    for (int i = 0; i < size; ++i) {
//        int64_t val = tmp.layer_first_hist.at<int64_t>(cv::Point(0,i));
//        if (val) {
//            std::cout << "val [" << i << "]: " << val << std::endl;
//        }
//    }
}

std::vector<patches_data> points_processor::get_data() const
{
    return data;
}

void face_matcher::set_test_set(const std::vector<patches_data> &value)
{
    test_set = value;
}

cv::Mat face_matcher::get_training_vectors() const
{
    return training_vectors;
}

std::vector<int> face_matcher::get_lables() const
{
    return lables;
}

void face_matcher::apply_svm(std::string save_path) {
    // Train the SVM
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
    svm->train(training_vectors, cv::ml::ROW_SAMPLE, lables);
    cout << "Finished training process" << endl;

    Mat sv = svm->getSupportVectors();
    cout << "SupportVectors: " << sv.rows << " | " << sv.cols << endl;
    svm->save(save_path);
}

float face_matcher::predict(Mat &sample, Mat &result){
    return svm->predict(sample, result, 1);
}

void face_matcher::set_first_set(const std::vector<patches_data> &value)
{
    first_set = value;
}

/*
 * Copyright 2018, myasnikov.mike at gmail.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef POINTS_PROCESSOR_HPP
#define POINTS_PROCESSOR_HPP

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

struct patches_data
{
    cv::Point2f center_pose;
    cv::Mat layer_first;
    cv::Mat layer_first_lbp;
    cv::Mat layer_first_hist;

    cv::Mat layer_second;
    cv::Mat layer_second_lbp;
    cv::Mat layer_second_hist;
};


class points_processor
{
    std::vector<cv::Point2f> nose_points;
    std::vector<cv::Point2f> nose_bridge_points;
    std::vector<cv::Point2f> eyebrow_points;
    std::vector<cv::Point2f> mouth_points;
    std::vector<cv::Point2f> left_eye_points;
    std::vector<cv::Point2f> right_eye_points;

    cv::Mat origin_image;
    std::vector<cv::Point2f> points;
    std::vector<patches_data> data;

    int first_layer_offest;

    static cv::Rect get_patch(cv::Point2f center, int offset);
public:
    points_processor(cv::Mat image, std::vector<cv::Point2f> shapes);
    ~points_processor(){}

    void generate_first_layer();

    std::vector<patches_data> get_data() const;
};



class face_matcher
{
    std::vector<patches_data> first_set;
    std::vector<patches_data> test_set;

    cv::Mat training_vectors;
    std::vector<int> lables;
    cv::Ptr<cv::ml::SVM> svm;


public:
    static cv::Mat construct_data_vector(const std::vector<patches_data>& data){
        cv::Mat res;
        std::vector<cv::Mat> tmp;
        std::for_each(data.begin(), data.end(), [&tmp] (auto el) {
            tmp.push_back(el.layer_first_hist.clone());
        });

        cv::hconcat(tmp, res);

//        std::cout << "concat_vector: " << res << std::endl;
        return res;
    }
    enum class match_index {
        match = 1, not_match = -1
    };

    struct training_data{
        std::vector<patches_data> value;
        match_index index;
    };

    face_matcher():svm(cv::ml::SVM::create()) {}
    void set_first_set(const std::vector<patches_data> &value);
    void set_test_set(const std::vector<patches_data> &value);

    void add_training_data(std::vector<patches_data>& data, match_index index){
        lables.push_back(static_cast<int> (index));
        training_vectors.push_back(construct_data_vector(data));
    }

    void fill_training_data(const std::vector<training_data>& data) {
        std::for_each(data.begin(), data.end(), [this] (auto el) {
            this->add_training_data(el.value, el.index);
        });
    }

    std::vector<double> match() {
        std::vector<double> res;
        if (!first_set.size()) return res;
        if (!test_set.size()) return res;

        for (int i = 0; i < first_set.size(); ++i) {
            res.push_back(cv::compareHist(first_set[i].layer_first_hist, test_set[i].layer_first_hist, 0));
        }
        return res;
    }
    cv::Mat get_training_vectors() const;
    std::vector<int> get_lables() const;

    void apply_svm(std::string save_path);

    float predict(cv::Mat& sample, cv::Mat &result);

    void load_svm(std::string svm_path){
        svm = cv::ml::SVM::load(svm_path);
    }
};


#endif  /* POINTS_PROCESSOR_HPP */

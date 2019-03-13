#ifndef lzy_functions
#define lzy_functions

#include <iostream>

#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <list>
#include <chrono>
#include <cmath>


using namespace std;
using namespace cv;



void using_rect ( 
  const Mat& img_1, 
  const Mat& img_2,
  Mat& img1_clone,
  Mat& img2_clone,
  const Rect& rect,
  const vector<KeyPoint>& keypoints_1,
  const vector<KeyPoint>& keypoints_2,
  vector<KeyPoint>& kpt_1_result,
  vector<KeyPoint>& kpt_2_result,
  vector<DMatch>& matches_part,
  vector<DMatch>& good_matches,
  vector<DMatch>& all_matches
);

void draw_results( 
  const vector< DMatch > matches_part,
  const Mat& img_1, 
  const Mat& img_2, 
  const vector< KeyPoint > kpt_1_result, 
  const vector< KeyPoint > kpt_2_result,
  const vector<DMatch>& good_matches,
  vector<DMatch>& all_matches
);

void get_rect(
  const Mat& img_1,
  const vector<KeyPoint>& keypoints_1,
  vector<Rect>& rect_array
);


void gaoxiang_find_feature_matches(
    const Mat& img_1, const Mat& img_2
);

void hanpeng_flann (
  const Mat& img_1, const Mat& img_2
);


void calculate_keypoints (
  const vector<KeyPoint> keypoints_1
);
#endif

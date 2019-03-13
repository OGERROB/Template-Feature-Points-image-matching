#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <list>
#include <chrono>

#include <functions.h>
using namespace std;
using namespace cv;



int main ( int argc, char** argv )
{
    if ( argc != 3 )
    {
        cout<<"usage: feature_extraction img1 img2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );
    Mat img2_clone = img_2.clone();
    Mat img1_clone = img_1.clone();    
    Ptr<FeatureDetector> detector = ORB::create();
    vector<KeyPoint> keypoints_1, keypoints_2;
    Mat descriptors_1, descriptors_2;


    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );


    Mat outimg1;
    drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB特征点",outimg1);
//     imwrite("featurepoints2.png",outimg1);
 ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////   
//     calculate_keypoints( keypoints_1 );


    vector<DMatch> matches_part;
    vector<DMatch> good_matches;
    vector<DMatch> all_matches;
    vector<KeyPoint> kpt_1_result;
    vector<KeyPoint> kpt_2_result;
    vector<Rect> rect_array;
    cout << "Start calculating!" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    get_rect(img_1, keypoints_1,rect_array);
    chrono::steady_clock::time_point t1z = chrono::steady_clock::now();
//     int i=9;
    for ( int i=0; i< rect_array.size(); i++)
//     for ( int i=0; i< 30; i++)
    {
	using_rect( img_1, img_2,img1_clone, img2_clone, rect_array[i], keypoints_1, keypoints_2, kpt_1_result, kpt_2_result, matches_part, good_matches,all_matches );
    }
    
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    chrono::duration<double> time_usedz = chrono::duration_cast<chrono::duration<double>>(t1z-t1);
    cout << "Calculation done." << endl;
    cout<<"template without Time cost: "<<time_used.count()<<" seconds."<<endl;
        cout<<"template grabbing Time cost: "<<time_usedz.count()<<" seconds."<<endl;
    
    
//     draw_results( matches_part, img_1, img_2, kpt_1_result, kpt_2_result, good_matches,all_matches );
    draw_results( matches_part, img1_clone, img2_clone, kpt_1_result, kpt_2_result, good_matches,all_matches );
    
        cout << "Start calculating!" << endl;
    chrono::steady_clock::time_point t11 = chrono::steady_clock::now();
    gaoxiang_find_feature_matches(img_1, img_2);
        chrono::steady_clock::time_point t22 = chrono::steady_clock::now();
    chrono::duration<double> time_used2 = chrono::duration_cast<chrono::duration<double>>(t22-t11);
    cout << "Calculation done." << endl;
    cout<<"Time cost: "<<time_used2.count()<<" seconds."<<endl;
    
    hanpeng_flann(img_1, img_2);

    waitKey(0);
    return 0;
}










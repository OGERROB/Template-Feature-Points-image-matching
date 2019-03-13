#include "functions.h"

void using_rect ( 
  const Mat& img_1, 
  const Mat& img_2, 
  Mat& img1_clone,
  Mat& img2_clone,
  const Rect& rect, 
  const vector< KeyPoint >& keypoints_1, 
  const vector< KeyPoint >& keypoints_2, 
  vector< KeyPoint >& kpt_1_result, 
  vector< KeyPoint >& kpt_2_result, 
  vector< DMatch >& matches_part, 
  vector<DMatch>& good_matches, 
  vector<DMatch>& all_matches
)
{
      Mat part;
    part = img_1( rect );
    
//     imshow( "part of the image", part );
        Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    
        Mat descriptors_1_part, descriptors_2_part;
    
    Mat image_matched;
    
    matchTemplate(img_2, part, image_matched, cv::TM_CCOEFF_NORMED);
    
    double minVal, maxVal;
    Point minLoc, maxLoc;
	//寻找最佳匹配位置
    minMaxLoc(image_matched, &minVal, &maxVal, &minLoc, &maxLoc);
//     cout<< minVal << "    " << maxVal << endl;

    if( maxVal < 0.7 )
      return;
//     Rect rect2 = Rect(rect.x+rect.width*0.5,rect.y+rect.height*0.5,rect.width,rect.height);
        rectangle(img1_clone, 
	   rect,
	   Scalar(0, 0, 255),
	   2,
	   8,
	   0
	  );

    rectangle(img2_clone, 
	   Point(maxLoc.x, maxLoc.y ),
	   Point(maxLoc.x + part.cols, maxLoc.y+part.rows),
	   Scalar(0, 0, 255),
	   2,
	   8,
	   0
	  );
    imshow( "模板", img1_clone );
    imshow( "当前帧中匹配上的模板", img2_clone );
    
    /////////////////////////////////////////////////////////模板匹配////////////DONE///////////////////////////////////////////////////////

    /////////////////////////////////////////////////////////////////////////////模板内配对////////////////////////////////////////////////////////////////////////
    
    vector<KeyPoint> img_1_part, img_2_part;
    
    for ( auto kpt_1:keypoints_1 )
    {
      if (kpt_1.pt.x<rect.x+rect.width && kpt_1.pt.x>rect.x && kpt_1.pt.y<rect.y+rect.height && kpt_1.pt.y>rect.y )
      {
	img_1_part.push_back(kpt_1);
// 	kpt_1_result.push_back(kpt_1);
      }
      
    }
    
        for ( auto kpt_2:keypoints_2 )
    {
      if (kpt_2.pt.x<maxLoc.x+rect.width && kpt_2.pt.x>maxLoc.x && kpt_2.pt.y<maxLoc.y+rect.height && kpt_2.pt.y>maxLoc.y )
      {
	img_2_part.push_back(kpt_2);
// 	kpt_2_result.push_back(kpt_2);
      }
      
    }
    
    
    
    
    descriptor->compute ( img_1, img_1_part, descriptors_1_part );
    descriptor->compute ( img_2, img_2_part, descriptors_2_part );
    
  
    if(descriptors_1_part.cols!=descriptors_2_part.cols)
      return;
    
    vector<DMatch> matches_part_part;

    matcher->match ( descriptors_1_part, descriptors_2_part, matches_part_part );
    
    Mat result_lzy;
    drawMatches ( img_1, img_1_part, img_2, img_2_part, matches_part_part, result_lzy );

    
    for(DMatch match:matches_part_part)
    {
      
      match.queryIdx += kpt_1_result.size();
      match.trainIdx += kpt_2_result.size();
      
          matches_part.push_back(match);
	  all_matches.push_back(match);
    }

    
    for(auto kp:img_1_part)
    {
      kpt_1_result.push_back(kp);
    }
        for(auto kp:img_2_part)
    {
      kpt_2_result.push_back(kp);
    }
//     imshow( "part matches",result_lzy );
    
    

              double min_dist=10000, max_dist=0;
     for ( int i = matches_part.size() - matches_part_part.size(); i < matches_part.size(); i++ )
    {
        double dist = matches_part[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }
    
//     printf ( "-- Max dist : %f \n", max_dist );
//     printf ( "-- Min dist : %f \n", min_dist );

    for ( int i = matches_part.size() - matches_part_part.size(); i < matches_part.size(); i++ )
    {
        if ( matches_part[i].distance <= max ( 1.5*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches_part[i] );
        }
    }

}




void draw_results( 
  const vector<DMatch> matches_part, 
  const Mat& img_1, 
  const Mat& img_2, 
  const vector<KeyPoint> kpt_1_result, 
  const vector<KeyPoint> kpt_2_result,
  const vector<DMatch>& good_matches,
  vector<DMatch>& all_matches
)
{
 
    Mat img_goodmatch;
    Mat img_all_matches;
    
    drawMatches ( img_1, kpt_1_result, img_2, kpt_2_result, good_matches, img_goodmatch );
    drawMatches ( img_1, kpt_1_result, img_2, kpt_2_result, all_matches, img_all_matches );
    imshow ( "模板匹配最终汉明距离优化之前", img_all_matches );
    imshow ( "模板匹配优化之后", img_goodmatch );
    
//     imwrite("template matching result.jpg",img_goodmatch);
    imwrite("matching with templates.jpg",img_goodmatch);
    cout<< "一共找到 " <<  matches_part.size() <<"对匹配点"<< endl;
    cout<< "筛选剩余 " <<  good_matches.size() <<"对匹配点"<< endl;

    waitKey(0);
}


void get_rect(
  const Mat& img_1,
  const vector<KeyPoint>& keypoints_1,
  vector<Rect>& rect_array
)
{
  int cols = img_1.cols;
  int rows = img_1.rows;
  
  // cols = 640, rows = 480
  
//   for( int i=0; i<8; i++)
//   {
//     for( int j=0; j<6; j++ )
//     {
// 	Rect rect( i*cols/8, j*rows/6, cols/8, rows/6 );
// 	int k = 0;
// 	for ( auto kpt_1:keypoints_1 )
// 	{	  
// 	    if (kpt_1.pt.x<rect.x+rect.width && kpt_1.pt.x>rect.x && kpt_1.pt.y<rect.y+rect.height && kpt_1.pt.y>rect.y )
// 		k++;
// 	 }  
// 	 if (k > 25)
// 		rect_array.push_back(rect);  
//     }
//   }
//   
  list<KeyPoint> pointlist;

  for (auto pt:keypoints_1)
  {
    pointlist.push_back(pt);
  }
  

  
  for ( int i=0; i<60; i++ )
  {
    for( int j=0; j<40; j++ )
    {
  Rect rect( cols*i/60,rows*j/40,cols/16, rows/12 );
      int k = 0;
      for ( auto kpt_1:pointlist )
	{	  
	    if (kpt_1.pt.x<rect.x+rect.width && kpt_1.pt.x>rect.x && kpt_1.pt.y<rect.y+rect.height && kpt_1.pt.y>rect.y )
		k++;
	 }  
	 if (k > 5)
	 {
		rect_array.push_back(rect);
		unsigned char status[500]={0};		  
		
		int status_id = 0;		
		 for ( auto kpt_1:pointlist )
		{	  
		    if (kpt_1.pt.x<rect.x+rect.width && kpt_1.pt.x>rect.x && kpt_1.pt.y<rect.y+rect.height && kpt_1.pt.y>rect.y )
			status[status_id]=1;
		    status_id++;
		}  
		
		int count = 0;
		 for ( auto iter=pointlist.begin(); iter!=pointlist.end(); count++)
		{
		    if ( status[count] == 1 )
		    {
			iter = pointlist.erase(iter);
			continue;
		      }
		    iter++;
		  }
	 }
      
    }
  }
  
  
  
}


void gaoxiang_find_feature_matches(
    const Mat& img_1, const Mat& img_2
)
{
   vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
     //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
    
    Mat img_goodmatch;
    Mat img_all_matches;
    
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_goodmatch );
    drawMatches ( img_1, keypoints_1, img_2, keypoints_2, match, img_all_matches );
    imshow ( "gaoxaing优化之前", img_all_matches );
    imshow ( "gaoxaing优化之后", img_goodmatch );
    
//     imwrite("gaoxaing.jpg",img_goodmatch);
    
    cout<< "一共找到 " <<  match.size() <<"对匹配点"<< endl;
    cout<< "筛选剩余 " <<  matches.size() <<"对匹配点"<< endl;

}


void hanpeng_flann (
  const Mat& img_1, const Mat& img_2
)
{
  
  	std::vector<KeyPoint> keypoints_1, keypoints_2;//关键点
	Mat descriptors_1, descriptors_2;//描述子
	Ptr<FeatureDetector> detector1 = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	// Ptr<FeatureDetector> detector = FeatureDetector::create(detector_name);
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create(descriptor_name);
	

	//-- 第一步:检测 Oriented FAST 角点位置
	detector1->detect(img_1, keypoints_1);
	detector1->detect(img_2, keypoints_2);

	//-- 第二步:根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);

	Mat outimg1,outimg2;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img_2, keypoints_2, outimg2, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//imwrite("D:\\lenovo\\goce_feature.png",outimg1);

// 	imwrite("ORB1.png",outimg1);
// 	imwrite("ORB2.png",outimg2);

	clock_t start_time, end_time;
	
	
	vector<DMatch> matches,good_matches;
	/***********FLANN匹配**********/
	flann::Index flannIndex(descriptors_1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	Mat matchIndex(descriptors_2.rows, 2, CV_32SC1), matchDistance(descriptors_2.rows,2, CV_32FC1);

	start_time = clock();
	flannIndex.knnSearch(descriptors_2, matchIndex, matchDistance, 2, flann::SearchParams());
	end_time = clock();
// 	cout << "FLANN match points cost time:" << (double)(end_time - start_time) / 1000 * CLOCKS_PER_SEC << "ms" << endl;

	for (int i = 0; i < matchDistance.rows; i++)
	{
		DMatch match(i, matchIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
		matches.push_back(match);
		if (matchDistance.at<float>(i, 0) < 0.7*matchDistance.at<float>(i, 1))
			good_matches.push_back(match);
	}
// 	cout << "queryidx" << "     " << "trainidx" << endl;
// 	for (int i = 0; i < good_matches.size(); i++)
// 	{
// // 		cout << good_matches[i].queryIdx << "           " << good_matches[i].trainIdx << endl;
// 	}
	
/*************************************/
	vector<DMatch> good_matches2, real_good_matches;
	flann::Index flannIndex2(descriptors_2, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);
	Mat matchIndex2(descriptors_1.rows, 2, CV_32SC1), matchDistance2(descriptors_1.rows, 2, CV_32FC1);

	start_time = clock();
	flannIndex2.knnSearch(descriptors_1, matchIndex2, matchDistance2, 2, flann::SearchParams());
	end_time = clock();
// 	cout << "match points cost time:" << (double)(end_time - start_time) / 1000 * CLOCKS_PER_SEC << "ms" << endl;

	for (int i = 0; i < matchDistance2.rows; i++)
	{
		DMatch match2(i, matchIndex2.at<int>(i, 0), matchDistance2.at<float>(i, 0));
	
		if (matchDistance2.at<float>(i, 0) < 0.7*matchDistance2.at<float>(i, 1))
			good_matches2.push_back(match2);
	}

// 	cout << "queryidx2" << "     " << "trainidx2" << endl;
// 	for (int i = 0; i < good_matches2.size(); i++)
// 	{
// // 		cout << good_matches2[i].queryIdx << "           " << good_matches2[i].trainIdx << endl;
// 	}
	

	for (int i = 0; i < good_matches.size(); i++)
	{
		for (int j = 0; j < good_matches2.size(); j++)
		{
			if (good_matches[i].queryIdx == good_matches2[j].trainIdx&&good_matches[i].trainIdx == good_matches2[j].queryIdx)
				real_good_matches.push_back(good_matches[i]);
		}
	}
	cout << "FLANN results: " << endl;
	cout << "正向: " << good_matches.size() << endl;
	cout << "反向: " << good_matches2.size() << endl;
	cout << "最终: " << real_good_matches.size() << endl;
	/***************************************************/

	/***********FLANN匹配**********/
	

	Mat img_match;
	Mat img_goodmatch,img_goodmatch1,img_goodmatch2;
// 	drawMatches(img_2, keypoints_2, img_1, keypoints_1, matches, img_match);
// 	drawMatches(img_2, keypoints_2, img_1, keypoints_1, good_matches, img_goodmatch);
// 	drawMatches(img_2, keypoints_2, img_1, keypoints_1, real_good_matches, img_goodmatch);
// 	drawMatches(img_2, keypoints_2, img_1, keypoints_1, good_matches, img_goodmatch1);
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches2, img_goodmatch2);
// 	imshow("yuanshipeidui", img_match);
// 	imshow("shiaxuanhoude", img_goodmatch);
// 	imshow("shaixuanhoude1", img_goodmatch1);
	imshow("FLANN 筛选后", img_goodmatch2);
// 	imwrite("extraction1.png", img_goodmatch1);
//  	imwrite("falnn.jpg", img_goodmatch2);
// 	imwrite("extraction.jpg", img_goodmatch);
	

}


void calculate_keypoints(
  const vector< KeyPoint > keypoints_1
)
{
  
  
  vector<int> inner_index;
  float distence_parameter = 25;
  
//   for ( int i=0;i<640;i+=10 )
//   {
//     for( int j=0;j<480;j+=10 )
//     {
//     int amount = 0;
//     for ( auto pt:keypoints_1 )
//     {
//       if ( (pow((pt.pt.x-i),2)+pow((pt.pt.y-j),2)) < distence_parameter )
// 	amount ++;
//     }
//     inner_index.push_back(amount);
//     }
//   }
  
  for( auto pt:keypoints_1 )
  {
    int amount = 0;
    for ( auto pt2:keypoints_1 )
    {
        if ( (pow((pt.pt.x-pt2.pt.x),2)+pow((pt.pt.y-pt2.pt.y),2)) < distence_parameter )
	amount ++;
    }
        inner_index.push_back(amount-1);
  }
  
  
  
  
  
  
      ofstream fout3;
 fout3.open("index_part.txt",ios_base::out);
//  fout3.open("index_all.txt",ios_base::out);
 if(fout3.is_open())
  {
   for( int index:inner_index )
    {

     fout3<<index;
     fout3 << endl;

    }
  }
 fout3.close();
  
  
        ofstream fout1;
 fout1.open("x_part.txt",ios_base::out);
//   fout1.open("x_all.txt",ios_base::out);
 if(fout1.is_open())
  {
    
//    for( int i=0; i<640; i+=10 )
//     {
// for( int j=0;j<480;j+=10)
// {
//      fout1<<i;
//      fout1 << endl;
// }
//     }
    
    for(auto pt:keypoints_1)
    {
      fout1<<pt.pt.x;
      fout1<<endl;
    }
    
    
    
  }

 fout1.close();
 
        ofstream fout2;
 fout2.open("y_part.txt",ios_base::out);
//   fout2.open("y_all.txt",ios_base::out);
 if(fout2.is_open())
  {
    
    
    
    
    
    
    
/*    
   for( int i=0; i<640; i+=10 )
    {
for( int j=0;j<480;j+=10)
{
     fout2<<480-j;
     fout2 << endl;
}
    }
    */
        for(auto pt:keypoints_1)
    {
      fout2<<480-pt.pt.y;
      fout2<<endl;
    }
    
    
  }
 fout2.close();
 
  
  

}

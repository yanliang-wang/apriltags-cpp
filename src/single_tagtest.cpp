/*********************************************************************
 * This file is distributed as part of the C++ port of the APRIL tags
 * library. The code is licensed under GPLv2.
 *
 * Original author: Edwin Olson <ebolson@umich.edu>
 * C++ port and modifications: Matt Zucker <mzucker1@swarthmore.edu>
 ********************************************************************/

#include <cstdio>
#include <iostream>
#include <string>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "TagDetector.h"
#include "DebugImage.h"

#define DEFAULT_TAG_FAMILY "Tag36h11"

void getCorners(const cv::Mat& im , cv::Point2d corners[])
{
    // up , down
    for(int i = 0; i < im.rows - 1 ; i++)
    {
        cv::Scalar current_mean_temp = cv::mean(im.rowRange(i,i+1));
        cv::Scalar next_mean_temp = cv::mean(im.rowRange(i+1,i+2));
        double current_sum(0) , next_sum(0);
        for(int j = 0; j < 4 ;j++)
        {
            current_sum += current_mean_temp.val[j];
            next_sum += next_mean_temp.val[j];
        }
        //up
        if(current_sum == 0 && next_sum > 0)
        {
            int j;
            for(j = 0; j < im.cols ;j ++)
            {
                if( im.at<cv::Vec3b>(i+1,j)[0] ||
                    im.at<cv::Vec3b>(i+1,j)[1] ||
                    im.at<cv::Vec3b>(i+1,j)[2])
                {
                    break;
                }
            }
            corners[0] = cv::Point2d(j , i+1) ;
        }
        //down
        if(current_sum > 0 && next_sum == 0)
        {
            int j;
            for(j = 0; j < im.cols ;j ++)
            {
                if( im.at<cv::Vec3b>(i,j)[0] ||
                    im.at<cv::Vec3b>(i,j)[1] ||
                    im.at<cv::Vec3b>(i,j)[2])
                {
                    break;
                }
            }
            corners[2] = cv::Point2d( j , i) ;
        }
    }
    // left right
    for(int i = 0; i < im.cols - 1 ; i++)
    {
        cv::Scalar current_mean_temp = cv::mean(im.colRange(i,i+1));
        cv::Scalar next_mean_temp = cv::mean(im.colRange(i+1,i+2));
        double current_sum(0) , next_sum(0);
        for(int j = 0; j < 4 ;j++)
        {
            current_sum += current_mean_temp.val[j];
            next_sum += next_mean_temp.val[j];
        }
        //left
        if(current_sum == 0 && next_sum > 0)
        {
            int j;
            for(j = 0; j < im.rows ;j ++)
            {
                if( im.at<cv::Vec3b>(j,i+1)[0] ||
                    im.at<cv::Vec3b>(j,i+1)[1] ||
                    im.at<cv::Vec3b>(j,i+1)[2])
                {
                    break;
                }
            }
            corners[3] = cv::Point2d( i+1,j ) ;
        }
        //down
        if(current_sum > 0 && next_sum == 0)
        {
            int j;
            for(j = 0; j < im.cols ;j ++)
            {
                if( im.at<cv::Vec3b>(j,i)[0] ||
                    im.at<cv::Vec3b>(j,i)[1] ||
                    im.at<cv::Vec3b>(j,i)[2])
                {
                    break;
                }
            }
            corners[1] = cv::Point2d(i , j) ;
        }
    }

}
int main(int argc, char** argv) {

    const std::string win = "Single tag test";
    const std::string family_str = "Tag16h5";
    double error_fraction ;
    std::stringstream ss;
    ss << argv[1];
    ss >> error_fraction;
    TagFamily family(family_str);
    TagDetectorParams params;
    params.adaptiveThresholdRadius += (params.adaptiveThresholdRadius+1) % 2;
    if (error_fraction >= 0 && error_fraction < 1) {
        family.setErrorRecoveryFraction(error_fraction);
    }

    TagDetector detector(family, params);
    detector.debug = false;//默认为false
    detector.debugWindowName = win;
    TagDetectionArray detections;

    char file_name[100] ;
    for (int i = 2; i < 4; ++i) {
        sprintf(file_name, "/home/wang/wang/git_files/apriltags-cpp/images/image_orig/%d.jpg", i);
        cv::Mat src = cv::imread(file_name);
        if (src.empty()) { continue; }
        cv::Point2d opticalCenter(0.5*src.rows, 0.5*src.cols);

        clock_t start = clock();
        detector.process(src, opticalCenter, detections);
        clock_t end = clock();
        //显示识别结果
        std::cout << "Got " << detections.size() << " detections in "
        << double(end-start)/CLOCKS_PER_SEC << " seconds.\n";
//        cv::Mat img = family.superimposeDetections(src, detections);
//        labelAndWaitForKey(win, "Detected", img, ScaleNone, true);
        //获取角点
        for (size_t i=0; i<detections.size(); ++i) {
            cv::Mat img = family.detectionImage(detections[i], src.size(), src.type());
            cv::Point2d corners[4];
            clock_t start = clock();
            getCorners(img,corners);
            clock_t end = clock();
            std::cout << "Got corners in "
                      << double(end-start)/CLOCKS_PER_SEC << " seconds.\n";
/*            const TagDetection& d = detections[i];
            std::cout<<src.type()<<std::endl;
            cv::Mat dst(src.size(), src.type());
            cv::Mat im = family.makeImage(d.id);
            if (im.depth() != dst.depth()) {
                cv::Mat i2;
                at::real scl = 1.0;
                if (dst.depth() == CV_32F || dst.depth() == CV_64F) {
                    scl = 1.0/255;
                }
                im.convertTo(i2, scl);
                im = i2;
            }
            if (im.channels() < dst.channels()) {
                cv::Mat i2;
                cv::cvtColor(im, i2, cv::COLOR_GRAY2RGB);
                im = i2;
            }
            cv::Mat W = family.getWarp(detections[i]);
            std::cout << "W: \n" << W << "\n" <<std::endl;

            cv::Mat pts_src = (cv::Mat_<float>(4,3) <<
                    0,0,1,
                    im.cols - 1, 0 ,1,
                    im.cols - 1, im.rows -1 ,1,
                    0, im.rows - 1 , 1);
//            std::cout << pts_src.t() <<std::endl;
            cv::Mat corners = W * pts_src.t();
            std::cout << "corners: \n" << corners << "\n" <<std::endl;*/

            for (uint32_t i = 0; i < 4; i++)
            {
                uint32_t next = (i== 4-1)?0:i+1;
                cv::Point2d p0 = corners[i];
                cv::Point2d p1 = corners[next];
                // draw vertex
                cv::circle(src, p0, double(src.rows) /150 , cv::Scalar(0,0,255),src.rows/150 + 3);
                // draw line
                cv::line(src,p0, p1, cv::Scalar(0,255,0) , src.rows/200);
            }
        }
        labelAndWaitForKey(win, "Detected", src, ScaleNone, true);
    }
    std::cout<<std::endl;
//    detector.reportTimers();
    return 0;
}

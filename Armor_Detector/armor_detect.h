//
// Created by mrm on 19-12-23.
//

#ifndef COLOR_TEST_ARMOR_DETECT_H
#define COLOR_TEST_ARMOR_DETECT_H

#include <iostream>
#include <fstream>
#include <string.h>

#include <opencv2/opencv.hpp>

#include "lightbar.h"
#include "camera.h"
#include "enemy.h"
#include "optimize.h"
#include "predict.h"

using namespace std;
using namespace cv;
using namespace hitcrt;
using namespace Eigen;

namespace hitcrt {

    class Armor_detector{
    public:
        
        ofstream odometry, odometryop, pdfcenter;
        Mat trans_matrix;
        optimize op;
        Enemy enemy;
        Camera camera;
        Mat hsv;
        vector<Point3d > history_center;
        vector<Point3d > origin_center;
        vector<Point2d > tails;
        vector<Point3d > CENTER5,VARIANCE5,CENTER15,VARIANCE15;
        vector<pair<double,double> > trust_prob; // front_prob > leap_prob + 0.15?
        timespec tstart1,tend1,tstart2,tend2,tstart3,tend3,tstart4,tend4;
        vector<Point3d > predict;
        Rect last_rect;
        int count = 0;
        double s = 0.0;

        Armor_detector(int sN = 3, int mN = 1):stateNum(sN),measureNum(mN),KF(3,1,0.2,10),
                KF2(3,1,0.2,10),KF3(3,1,0.2,10),camera(DAHUA2){
            odometry.open(ODOMETRY);
            odometryop.open(ODOMETRYOP);
            pdfcenter.open(PDFCENTER);


            // set KF filter
            setIdentity(KF.processNoiseCov,Scalar::all(0.0001));
            setIdentity(KF.measurementNoiseCov,Scalar::all(0.01));
            KF.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
            KF.measurementMatrix.at<float>(0,0) = 1;
            randn(KF.statePost,Scalar::all(-1),Scalar::all(1));
            setIdentity(KF.errorCovPost,Scalar::all(1));
            KF.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
                    0,1,(1-exp(-alpha*delta))/alpha,
                    0,0,exp(-alpha*delta));


            setIdentity(KF2.processNoiseCov,Scalar::all(0.001));
            setIdentity(KF2.measurementNoiseCov,Scalar::all(0.01));
            KF2.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
            KF2.measurementMatrix.at<float>(0,0) = 1;
            randn(KF2.statePost,Scalar::all(-1),Scalar::all(1));
            setIdentity(KF2.errorCovPost,Scalar::all(1));
            KF2.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
                    0,1,(1-exp(-alpha*delta))/alpha,
                    0,0,exp(-alpha*delta));

            setIdentity(KF3.processNoiseCov,Scalar::all(0.001));
            setIdentity(KF3.measurementNoiseCov,Scalar::all(0.01));
            KF3.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
            KF3.measurementMatrix.at<float>(0,0) = 1;
            randn(KF3.statePost,Scalar::all(180),Scalar::all(1));
            setIdentity(KF3.errorCovPost,Scalar::all(1));
            KF3.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
                    0,1,(1-exp(-alpha*delta))/alpha,
                    0,0,exp(-alpha*delta));

            trans_matrix = (Mat_<float> (3,3) << 1,delta1,(alpha*delta1-1+exp(-alpha*delta1))/(alpha*alpha),
                    0,1,(1-exp(-alpha*delta1))/alpha,
                    0,0,exp(-alpha*delta1));

            // set camera param
            camera.target_height = 0.128457;

        };
        bool rectinimage(const Rect &rect, const Mat &image_temp);
        void refine(const Mat &image, Rect &rect);
        int prompt(const Mat &image_temp, const Mat &hsv,int xmin,int ymin, const Rect &light_max_rect);
        double prob(const Mat &image_temp, const Rect &rect, bool negtive_flag, Rect &max_prob_rect, int state);
        double robotheta(const Enemy &enemy);
        double dist(const Point2d &p1, const Point2d &p2);
        double trans(double t);
        void exchange(Point2d &p1, Point2d &p2);
        pair<int, int> find_pair(const vector<Point2d> &UV, const vector<Point2d > &lightbars_position, const vector<double > &lines);
		// image-输入图像 EM-绘图图像 Image_temp-绘图图像 Tw_c-相机外参
        int run(Mat &image, Mat &EM, Mat &Image_temp, Matrix4d Tw_c);
        ~Armor_detector(){
            odometry.close();
            odometryop.close();
            pdfcenter.close();
        };

    private:

        double delta = 1;
        double alpha = 10;
        double alpha2 = 150;

		string POSITIVE = "../SIGN/POSITIVE/";
        string NEGTIVE = "../SIGN/NEGTIVE/";
        string LPOSITIVE = "../SIGN/POSITIVE_LIGHT/";
        string LNEGTIVE = "../SIGN/NEGTIVE_LIGHT/";
        string POSITIVE_CONFIG = "../SIGN/POSITIVE/config.txt";
        string NEGTIVE_CONFIG = "../SIGN/NEGTIVE/config.txt";
        string LPOSITIVE_CONFIG = "../SIGN/POSITIVE_LIGHT/config.txt";
        string LNEGTIVE_CONFIG = "../SIGN/NEGTIVE_LIGHT/config.txt";
        string VIDEO_PATH = "../video/demo.avi";
        string ODOMETRY = "../output/path.txt";
        string ODOMETRYOP = "../output/pathop.txt";
        string PDFCENTER = "../output/pdfcenter.txt";

        bool negtive_flag = false;
        double ave_y = 0.0;
        vector<pair<double, int>> positive_data, light_positive_data;
        vector<pair<double, int>> negtive_data, light_negtive_data;
        vector<Mat > positive, light_positive;
        vector<Mat > negtive, light_negtive;

		SKF KF, KF2, KF3;

		int stateNum = 3;
        int measureNum = 1;
        double delta1 = 6;
    };
}

#endif //COLOR_TEST_ARMOR_DETECT_H

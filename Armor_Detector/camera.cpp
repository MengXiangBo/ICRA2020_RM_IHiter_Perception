//
// Created by mrm on 19-11-13.
//

#include "camera.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace hitcrt{

    /*
    Camera::Camera(string filename){
        FileStorage fs(filename, FileStorage::READ);
        //FileStorage fs("test.xml", FileStorage::READ);

        if (!fs.isOpened())
            cout << "Warning : No file!" << endl;

        fx = (double)fs["fx"];
        fy = (double)fs["fy"];
        cx = (double)fs["cx"];
        cy = (double)fs["cy"];
        fs["k1k2Distortion"] >> *k1k2Distortion;
        fs["p1p2p3Distortion"] >> *p1p2p3Distortion;
        fs.release();
        paramtersInit();
    }*/

    void Camera::paramtersInit()
    {
        matrix = Mat(3, 3, CV_64F);
        matrix.at<double>(0, 0) = fx;
        matrix.at<double>(0, 1) = 0;
        matrix.at<double>(0, 2) = cx;
        matrix.at<double>(1, 0) = 0;
        matrix.at<double>(1, 1) = fy;
        matrix.at<double>(1, 2) = cy;
        matrix.at<double>(2, 0) = 0;
        matrix.at<double>(2, 1) = 0;
        matrix.at<double>(2, 2) = 1;
        coeff = Mat(1, 4, CV_64F);
        coeff.at<double>(0, 0) = k1k2Distortion[0];
        coeff.at<double>(0, 1) = k1k2Distortion[1];
        coeff.at<double>(0, 2) = p1p2p3Distortion[0];
        coeff.at<double>(0, 3) = p1p2p3Distortion[1];
        K(0,0) = fx;
        K(0,2) = cx;
        K(1,1) = fy;
        K(1,2) = cy;
    }

    // return points in normalization_coordinate(z=1)
    Point3d Camera::undistort(Point2d pt){
        vector<Point2d > distortpt;
        vector<Point2d > rawpt;
        rawpt.push_back(pt);
        undistortPoints(rawpt,distortpt, matrix, coeff);
        return Point3d(distortpt[0].x,distortpt[0].y,1.0);
    }

    Point2d Camera::back_project(cv::Point2d &p){
        Point3d pc = undistort(p);
        double alpha = (target_height-Tw_c(2,3))/(Tw_c(2,0)*pc.x+Tw_c(2,1)*pc.y+pc.z*Tw_c(2,2));
        pc = alpha*pc;
        Vector4d pc_h;
        pc_h(0,0) = pc.x;
        pc_h(1,0) = pc.y;
        pc_h(2,0) = pc.z;
        pc_h(3,0) = 1.0;
        pc_h = Tw_c*pc_h;
        return Point2d(pc_h(0,0),pc_h(1,0));
    }

}

//
// Created by mrm on 19-11-20.
//

#ifndef ARMOR_DETECTOR_OPTIMIZE_H
#define ARMOR_DETECTOR_OPTIMIZE_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "lightbar.h"
#include "camera.h"

using namespace Eigen;
using namespace std;
using namespace cv;

namespace hitcrt{

class optimize{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    double min_error = 0.00000000003;
    
    optimize() = default;
	// 根据3个灯条的位置，计算不确定椭圆并在已知灯条间距先验的情况下优化灯条位置
    vector<Point2d> direct_optimize(int ite_num, const vector<Point2d > &UV, vector<Point2d > position, pair<int, int> &reflect,
            Camera &cam, vector<double > distance, double the, Mat &em, Point2d c);
	// 根据机器人中心的历史位置信息以及当前机器人位置信息以及当前灯条信息
	// 相比于direct_optimize多引入了一个历史位置信息作为约束，确保机器人位置不会发散
    Point3d center_optimize(int ite_num, const vector<Point2d > &UV, vector<Point2d > position, pair<int, int> &reflect,
            Point3d center, int state, Camera &cam, vector<Point3d > &history_center, Mat &em, Point2d c, vector<Point3d > &origin_center);
	// 归一化
    Point2d norm(const Point2d p);
    double m_error(const Point2d p);
    Point2d m_coord(const Point2d ave, const Point2d p, const pair<double, double> std, const pair<Point2d, Point2d> axis);
	// 计算雅可比de/dxyt
    pair<Point3d, Point3d> Je_xyt(const Point3d center, const pair<Point2d, Point2d> r_r_l, const vector<pair<double, double>> std,
            const int state, const vector<pair<Point2d, Point2d>> &axis);
	// 计算雅可比de/dxy
    pair<double, double> Je_xy(const Point2d ave, const Point2d p, const pair<double, double> std, const pair<Point2d, Point2d> axis);
    pair<Point2d, Point2d> center2p(const Point2d center, const double theta, const int state);
	// 计算协方差椭圆
    pair<double, double> calc_variance(const Point2d uv, const pair<Point2d, Point2d> direct, Camera &cam, const double index);
    Point2d project(const Point2d proj, const Matrix4d Tw_cd, const Matrix3d Kd);
    double compute_error(double realdist,Point2d &p1,Point2d &p2);
    ~optimize() = default;
};

}


#endif //ARMOR_DETECTOR_OPTIMIZE_H

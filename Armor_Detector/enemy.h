//
// Created by mrm on 19-11-14.
//

#ifndef ARMOR_DETECTOR_ENEMY_H
#define ARMOR_DETECTOR_ENEMY_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Eigen/Core>

#include "lightbar.h"

namespace hitcrt{

class Enemy{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    int count = 0;

	// 右前/左前/右后/左后轮子以及尾灯在机器人坐标系下的坐标
    Eigen::Matrix<float, 8, 3> rf_wheel, lf_wheel, rb_wheel, lb_wheel, light;
    cv::Point2d positive_x_axis, positive_y_axis;
    cv::Point2d robo_center;
    cv::Point2d tail;

    enum wheel{
        RF = 0,
        RB = 1,
        LF = 2,
        LB = 3
    };

    Enemy(){

        rf_wheel << 0.173762,0.131047,0.076325,0.219643,0.131047,0.076325,0.173762,0.2075,0,0.219643,0.2075,0,
                0.173762,0.2075,0.15265,0.219643,0.2075,0.15265,0.173762,0.283953,0.076325,0.219643,0.28953,0.076325;
        lf_wheel << -0.173762,0.131047,0.076325,-0.219643,0.131047,0.076325,-0.173762,0.2075,0,-0.219643,0.2075,0,
                -0.173762,0.2075,0.15265,-0.219643,0.2075,0.15265,-0.173762,0.283953,0.076325,-0.219643,0.28953,0.076325;
        rb_wheel << 0.173762,-0.131047,0.076325,0.219643,-0.131047,0.076325,0.173762,-0.2075,0,0.219643,-0.2075,0,
                0.173762,-0.2075,0.15265,0.219643,-0.2075,0.15265,0.173762,-0.283953,0.076325,0.219643,-0.28953,0.076325;
        lb_wheel << -0.173762,-0.131047,0.076325,-0.219643,-0.131047,0.076325,-0.173762,-0.2075,0,-0.219643,-0.2075,0,
                -0.173762,-0.2075,0.15265,-0.219643,-0.2075,0.15265,-0.173762,-0.283953,0.076325,-0.219643,-0.28953,0.076325;
		light << 0.145,-0.192310,0.225045,0.145,-0.192310,0.268142,0.145,-0.239377,0.225045,0.145,-0.239377,0.268142,
                -0.145,-0.192310,0.225045,-0.145,-0.192310,0.268142,-0.145,-0.239377,0.225045,-0.145,-0.239377,0.268142;
        

    }
    ~Enemy() = default;

    // 在观测是正向装甲板的假设下计算机器人中心位置
    cv::Point2d find_center_front(const std::vector<cv::Point2d > &optimized_points, const std::pair<int, int> armor_indices, cv::Mat &em, const cv::Point2d center);
	// 在观测是侧向装甲板的假设下计算机器人中心位置    
	cv::Point2d find_center_leap(const std::vector<cv::Point2d > &optimized_points, const std::pair<int, int> armor_indices, cv::Mat &em, const cv::Point2d center);
    cv::Rect direct_def(const Eigen::Matrix4d Tw_c, const Eigen::Matrix3d K);
    cv::Rect wheel_world_coor(wheel state, Eigen::Matrix4d Tw_c, Eigen::Matrix3d K);
    // 用来计算尾部装甲板的坐标，并返回是否计算成功
    bool show_tail(cv::Mat &em, cv::Point2d center, std::vector<cv::Point2d > &tails);
    void exchange();
    void inv();
};
}

#endif //ARMOR_DETECTOR_ENEMY_H

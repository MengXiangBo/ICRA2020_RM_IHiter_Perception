//
// Created by mrm on 19-11-14.
//

#include "enemy.h"

using namespace cv;
using namespace Eigen;
using namespace std;

namespace hitcrt{

    double dist(const Point2d &p1, const Point2d &p2){
        return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
    }

    // 在观测是侧向装甲板的假设下计算机器人中心位置
    Point2d Enemy::find_center_leap(const vector<Point2d > &optimized_points, const pair<int, int> armor_indices, Mat &em,
                               const Point2d center){
        Point2d line_center, vert_direct, para_direct;
        line_center = (optimized_points[armor_indices.first]+optimized_points[armor_indices.second])/2;
#ifdef _DEBUG
        cout << "line center : " << line_center.x << "," << line_center.y << endl;
#endif
        para_direct = optimized_points[armor_indices.first]-optimized_points[armor_indices.second];
        vert_direct.x = -para_direct.y;
        vert_direct.y = para_direct.x;
        if (vert_direct.y < 0)
            vert_direct = -vert_direct;
        vert_direct = vert_direct/sqrt(pow(vert_direct.x,2)+pow(vert_direct.y,2));
        positive_x_axis = -vert_direct;
        positive_y_axis = Point2d(-positive_x_axis.y,positive_x_axis.x);
#ifdef _DEBUG
        cout << "vert_direct : " << vert_direct.x << "," << vert_direct.y << endl;
#endif
        robo_center = line_center + 0.187848*vert_direct;

        circle(em, Point2d(center.x + robo_center.x * 150, center.y - robo_center.y * 150), 4,
               Scalar(0, 173, 205), 2);
        return robo_center;
    }

    // 在观测是正向装甲板的假设下计算机器人中心位置
    Point2d Enemy::find_center_front(const vector<Point2d > &optimized_points, const pair<int, int> armor_indices, Mat &em,
                               const Point2d center){
        Point2d line_center, vert_direct, para_direct;
        line_center = (optimized_points[armor_indices.first]+optimized_points[armor_indices.second])/2;
#ifdef _DEBUG
        cout << "line center : " << line_center.x << "," << line_center.y << endl;
#endif
        para_direct = optimized_points[armor_indices.first]-optimized_points[armor_indices.second];
        vert_direct.x = -para_direct.y;
        vert_direct.y = para_direct.x;
        if (vert_direct.y < 0)
            vert_direct = -vert_direct;
        vert_direct = vert_direct/sqrt(pow(vert_direct.x,2)+pow(vert_direct.y,2));
        positive_y_axis = -vert_direct;
        positive_x_axis = Point2d(positive_y_axis.y,-positive_y_axis.x);
#ifdef _DEBUG
        cout << "vert_direct : " << vert_direct.x << "," << vert_direct.y << endl;
#endif
        robo_center = line_center + 0.268348*vert_direct;

        circle(em, Point2d(center.x + robo_center.x * 150, center.y - robo_center.y * 150), 4,
               Scalar(0, 173, 205), 2);
        return robo_center;
    }

    // 反转计算得到的敌方机器人坐标系
    void Enemy::inv()
    {
        positive_y_axis *= -1;
        positive_x_axis *= -1;
    }

   // 正交计算得到的敌方机器人坐标系
    void Enemy::exchange()
    {
        Point2d temp = positive_y_axis;
        positive_y_axis = positive_x_axis;
        positive_x_axis = -temp;
    }

    Rect Enemy::direct_def(const Matrix4d Tw_c, const Matrix3d K){
        if (positive_y_axis.y < 0) {
            positive_y_axis *= -1;
            positive_x_axis *= -1;
        }
        vector<Point3d > position;
        for (int i = 0; i < 8; ++i) {
            Point2d plane_coor = positive_x_axis*light(i,0)+positive_y_axis*light(i,1)+robo_center;
            position.emplace_back(Point3d(plane_coor.x,plane_coor.y,light(i,2)));
        }

        assert(position.size() == 8);
        Matrix4d Tc_w = Matrix4d::Identity();
        Tc_w.block(0,0,3,3) = Tw_c.block(0,0,3,3).transpose();
        Tc_w.block(0,3,3,1) = -Tc_w.block(0,0,3,3)*Tw_c.block(0,3,3,1);
        Matrix<double, 4, 8> w_points_h;
        for (int j = 0; j < 8; ++j) {
            w_points_h(0,j) = position[j].x;
            w_points_h(1,j) = position[j].y;
            w_points_h(2,j) = position[j].z;
            w_points_h(3,j) = 1.0;
        }
        w_points_h = Tc_w*w_points_h;
        Matrix<double, 3, 8> w_points = w_points_h.block(0,0,3,8);
        for (int k = 0; k < 8; ++k) {
            w_points(0,k) = w_points(0,k)/w_points(2,k);
            w_points(1,k) = w_points(1,k)/w_points(2,k);
            w_points(2,k) = 1.0;
        }
        w_points = K*w_points;
        vector<Point > UV;
        for (int l = 0; l < 8; ++l) {
            UV.emplace_back(Point(int(w_points(0, l)), int(w_points(1, l))));
            //cout << "Point : " << w_points(0, l) << "," << w_points(1, l) << endl;
        }

        Rect rect = boundingRect(UV);
        return rect;
    }

    // 求解机器人某个轮子的坐标
    Rect Enemy::wheel_world_coor(wheel state, Matrix4d Tw_c, Matrix3d K)
    {
        vector<Point3d > position;
        switch (state){
            case RF:{
                for (int i = 0; i < 8; ++i) {
                    Point2d plane_coor = positive_x_axis*rf_wheel(i,0)+positive_y_axis*rf_wheel(i,1)+robo_center;
                    position.emplace_back(Point3d(plane_coor.x,plane_coor.y,rf_wheel(i,2)));
                }
                break;
            }
            case LF:{
                for (int i = 0; i < 8; ++i) {
                    Point2d plane_coor = positive_x_axis*lf_wheel(i,0)+positive_y_axis*lf_wheel(i,1)+robo_center;
                    position.emplace_back(Point3d(plane_coor.x,plane_coor.y,lf_wheel(i,2)));
                }
                break;
            }
            case RB:{
                for (int i = 0; i < 8; ++i) {
                    Point2d plane_coor = positive_x_axis*rb_wheel(i,0)+positive_y_axis*rb_wheel(i,1)+robo_center;
                    position.emplace_back(Point3d(plane_coor.x,plane_coor.y,rb_wheel(i,2)));
                }
                break;
            }
            case LB:{
                for (int i = 0; i < 8; ++i) {
                    Point2d plane_coor = positive_x_axis*lb_wheel(i,0)+positive_y_axis*lb_wheel(i,1)+robo_center;
                    position.emplace_back(Point3d(plane_coor.x,plane_coor.y,lb_wheel(i,2)));
                }
                break;
            }
            default:
                break;
        }
        assert(position.size() == 8);
        Matrix4d Tc_w = Matrix4d::Identity();
        Tc_w.block(0,0,3,3) = Tw_c.block(0,0,3,3).transpose();
        Tc_w.block(0,3,3,1) = -Tc_w.block(0,0,3,3)*Tw_c.block(0,3,3,1);
        Matrix<double, 4, 8> w_points_h;
        for (int j = 0; j < 8; ++j) {
            w_points_h(0,j) = position[j].x;
            w_points_h(1,j) = position[j].y;
            w_points_h(2,j) = position[j].z;
            w_points_h(3,j) = 1.0;
        }
        w_points_h = Tc_w*w_points_h;
        Matrix<double, 3, 8> w_points = w_points_h.block(0,0,3,8);
        for (int k = 0; k < 8; ++k) {
            w_points(0,k) = w_points(0,k)/w_points(2,k);
            w_points(1,k) = w_points(1,k)/w_points(2,k);
            w_points(2,k) = 1.0;
        }
        w_points = K*w_points;
        vector<Point > UV;
        for (int l = 0; l < 8; ++l) {
            UV.emplace_back(Point(int(w_points(0, l)), int(w_points(1, l))));
            //cout << "Point : " << w_points(0, l) << "," << w_points(1, l) << endl;
        }

        Rect rect = boundingRect(UV);

        return rect;
    }

    // 用来计算尾部装甲板的坐标，并返回是否计算成功
    bool Enemy::show_tail(Mat &em, Point2d center,vector<Point2d > &tails)
    {
        tail = robo_center - 0.268348*positive_y_axis;
        Point2d tail2 = 2*robo_center-tail;
        Point2d ave4;

        if (tails.size() > 4) {
            for (int i = 0; i < 4; ++i)
                ave4 += tails[tails.size() - 1 - i];
            ave4 = ave4/4.0;
            if (dist(tail,ave4) > 0.2) {
                double d_k = (tails.back().y-tail.y)/(tails.back().x-tail.x);
                double d_c2l = abs((d_k*robo_center.x-robo_center.y-d_k*tail.x+tail.y)/sqrt(pow(d_k,2)+1));
#ifdef _DEBUG
                cout << "d_c2l : " << d_c2l << endl;
                cout << "tails.back : " << tails.back().x << "," << tails.back().y << endl;
                cout << "tail : " << tail.x << "," << tail.y << endl;
                cout << "robocenter : " << robo_center.x << "," << robo_center.y << endl;
#endif
                if (d_c2l > 0.08) {
                    count ++;
#ifdef _DEBUG
                    cout << "verticle wrong" << endl;
#endif
                    return true;
                }
                tail = tail2;
                count++;
            }
        }

        if (count > 3){
            count = 0;
            tails.clear();
            tail = robo_center - 0.268348*positive_y_axis;
            return false;
        }

        circle(em, Point2d(center.x + tail.x * 150, center.y - tail.y * 150), 4,
               Scalar(47,255,173), 4);
        return false;
    }

}

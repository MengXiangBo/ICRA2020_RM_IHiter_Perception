//
// Created by mrm on 19-11-20.
//

#include "optimize.h"

using namespace std;
using namespace Eigen;
using namespace cv;

namespace hitcrt {

    double dist(Point2d p1, Point2d p2)
    {
        return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
    }

    pair<double, double> optimize::calc_variance(const Point2d uv, const pair<Point2d, Point2d> direct, Camera &cam, const double index)
    {
        vector<Point2d > surround;
        surround.emplace_back(Point2d(3,0)+uv);
        surround.emplace_back(Point2d(-3,0)+uv);
        surround.emplace_back(Point2d(0,3)+uv);
        surround.emplace_back(Point2d(0,-3)+uv);
#ifdef _DEBUG
        cout << "uv : " << uv.x << "," << uv.y << endl;
#endif
        double relative_x_std = 0.0, relative_y_std = 0.0;
        Point2d relative_x = cam.back_project(surround[0])-cam.back_project(surround[1]);
        Point2d relative_y = cam.back_project(surround[2])-cam.back_project(surround[3]);
        Point2d relative_exact(0.01*index,0);
#ifdef _DEBUG
        cout << "surronund[0] : " << cam.back_project(surround[0]).x << "," << cam.back_project(surround[0]).y << endl;
        cout << "surronund[1] : " << cam.back_project(surround[1]).x << "," << cam.back_project(surround[1]).y << endl;
        cout << "surronund[2] : " << cam.back_project(surround[2]).x << "," << cam.back_project(surround[2]).y << endl;
        cout << "surronund[3] : " << cam.back_project(surround[3]).x << "," << cam.back_project(surround[3]).y << endl;
        cout << "relative_X : "  << relative_x.x << "," << relative_x.y << endl;
        cout << "relative_y : " << relative_y.x << "," << relative_y.y << endl;
        cout << "direct0 : " << direct.first.x << "," << direct.first.y << endl;
        cout << "direct1 : " << direct.second.x << "," << direct.second.y << endl;
#endif
        relative_x_std = abs(relative_x.x*direct.second.x+relative_x.y*direct.second.y)/dist(Point2d(0,0),direct.second)
                +abs(relative_exact.x*direct.second.x+relative_exact.y*direct.second.y)/dist(Point2d(0,0),direct.second);
        relative_y_std = abs(relative_y.x*direct.first.x+relative_y.y*direct.first.y)/dist(Point2d(0,0),direct.first)
                +abs(relative_exact.x*direct.first.x+relative_exact.y*direct.first.y)/dist(Point2d(0,0),direct.first);
        return make_pair(relative_x_std/2,relative_y_std/2);
    }

    // 装甲板朝左边是theta负\，朝右边是theta正/
    // state=1代表0.187848侧装加板，state=2代表0.268348正面装加板
    pair<Point2d, Point2d> optimize::center2p(const Point2d center, const double theta, const int state)
    {
        assert(state == 1 || state == 2);
        double d;
        double r = 0.0675;
        if (state == 1)
            d = 0.187848;
        else
            d = 0.268348;
        Point2d armor_center = center+Point2d(d*sin(theta*M_PI/180),-d*cos(theta*M_PI/180));
        Point2d p_right, p_left;
        p_right = armor_center+Point2d(r*cos(theta*M_PI/180),r*sin(theta*M_PI/180));
        p_left = armor_center-Point2d(r*cos(theta*M_PI/180),r*sin(theta*M_PI/180));
        return make_pair(p_right,p_left);
    }

    double optimize::m_error(const Point2d p)
    {
        return sqrt(pow(p.x,2)+pow(p.y,2));
    }

    Point2d optimize::m_coord(const Point2d ave, const Point2d p, const pair<double, double> std, const pair<Point2d, Point2d> axis)
    {
        double x, y;
        Point2d delta = p-ave;
        y = (delta.x*axis.first.x+delta.y*axis.first.y)/dist(Point2d(0,0),axis.first)/std.second;
        x = (delta.x*axis.second.x+delta.y*axis.second.y)/dist(Point2d(0,0),axis.second)/std.first;
        return Point2d(x,y);
    }

    pair<double, double> optimize::Je_xy(const Point2d ave, const Point2d p, const pair<double, double> std, const pair<Point2d, Point2d> axis)
    {
        // axis是先y后x，std是先x后y
        if (ave == p)
            return make_pair(0.0,0.0);
        double x, y;
        double Jx, Jy;
        Point2d delta = p-ave;
        y = (delta.x*axis.first.x+delta.y*axis.first.y)/dist(Point2d(0,0),axis.first)/std.second;
        x = (delta.x*axis.second.x+delta.y*axis.second.y)/dist(Point2d(0,0),axis.second)/std.first;
        Point2d mc(x,y);
        double me = m_error(mc);
        Jy = y/me*axis.first.y/dist(Point2d(0,0),axis.first)/std.second+x/me*axis.second.y/dist(Point2d(0,0),axis.second)/std.first;
        Jx = y/me*axis.first.x/dist(Point2d(0,0),axis.first)/std.second+x/me*axis.second.x/dist(Point2d(0,0),axis.second)/std.first;
        return make_pair(Jx,Jy);
    }

    // right-left J
    pair<Point3d, Point3d> optimize::Je_xyt(const Point3d center, const pair<Point2d, Point2d> r_r_l, const vector<pair<double, double>> std,
                                            const int state, const vector<pair<Point2d, Point2d>> &axis)
    {
        pair<Point3d, Point3d> J;
        pair<Point2d, Point2d> r_l = center2p(Point2d(center.x,center.y),center.z,state);
        double error_r_orig = m_error(m_coord(r_r_l.first,r_l.first,std[0],axis[0]));
        double error_l_orig = m_error(m_coord(r_r_l.second,r_l.second,std[1],axis[1]));
        double d_error_l, d_error_r;
        pair<Point2d, Point2d> d_r_l;

        for (int i = 0; i < 3; ++i) {
            switch (i){
                case 0:{
                    d_r_l = center2p(Point2d(center.x+0.01,center.y),center.z,state);
                    d_error_r = m_error(m_coord(r_r_l.first,d_r_l.first,std[0],axis[0]));
                    d_error_l = m_error(m_coord(r_r_l.second,d_r_l.second,std[1],axis[1]));
                    J.first.x = (d_error_r-error_r_orig)*100;
                    J.second.x = (d_error_l-error_l_orig)*100;
                    break;
                }
                case 1: {
                    d_r_l = center2p(Point2d(center.x, center.y + 0.01), center.z, state);
                    d_error_r = m_error(m_coord(r_r_l.first,d_r_l.first,std[0],axis[0]));
                    d_error_l = m_error(m_coord(r_r_l.second,d_r_l.second,std[1],axis[1]));
                    J.first.y = (d_error_r-error_r_orig)*100;
                    J.second.y = (d_error_l-error_l_orig)*100;
                    break;
                }
                case 2: {
                    d_r_l = center2p(Point2d(center.x, center.y), center.z + 0.01, state);
                    d_error_r = m_error(m_coord(r_r_l.first,d_r_l.first,std[0],axis[0]));
                    d_error_l = m_error(m_coord(r_r_l.second,d_r_l.second,std[1],axis[1]));
                    J.first.z = (d_error_r-error_r_orig)*100;
                    J.second.z = (d_error_l-error_l_orig)*100;
                    break;
                }
                default:
                    break;
            }
        }

        return J;
    }

    Point2d optimize::norm(const Point2d p)
    {
        double length = sqrt(pow(p.x,2)+pow(p.y,2));
        return p/length;
    }

    Point2d optimize::project(const Point2d proj, const Matrix4d Tw_cd, const Matrix3d K_inv){
        Vector3d uv;
        uv(0,0) = proj.x;
        uv(1,0) = proj.y;
        uv(2,0) = 1;
        Vector4d cc;
        cc.setOnes();
        cc.block(0,0,3,1) = K_inv*uv;
        double alpha = (0.128457-Tw_cd(2,3))/(Tw_cd(2,0)*cc(0,0)+Tw_cd(2,1)*cc(1,0)+Tw_cd(2,2)*cc(2,0));
        cc.block(0,0,3,1) = alpha*cc.block(0,0,3,1);
        cc = Tw_cd*cc;
        return Point2d(cc(0,0),cc(1,0));
    }

    // real-calc
    double optimize::compute_error(double realdist, Point2d &p1, Point2d &p2) {
        return realdist - sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
    }

	// 根据3个灯条的位置，计算不确定椭圆并在已知灯条间距先验的情况下优化灯条位置
    vector<Point2d> optimize::direct_optimize(int ite_num, const vector<Point2d > &UV, vector<Point2d > position, pair<int, int> &reflect,
                                              Camera &cam,vector<double > distance, double the, Mat &em, Point2d c) {
        double gain = 66; // dist_std = 0.015m
        vector<double > angles;
        Matrix<double, 4, 1> variable;
        Matrix<double, 4, 1> variable_origin;
        Matrix4d H;
        Matrix<double, 4, 1> b;
        Matrix<double, 3, 4> J = Matrix<double, 3, 4>::Zero(); // (variable.size(),error.size())
        Matrix<double, 3, 1> edge;
        vector<double > theta;
        vector<pair<Point2d, Point2d>> directs;
        vector<pair<double, double>> variance;
        theta.push_back(the);
        assert(theta.size() == 1);

        if (position[reflect.first].x < position[reflect.second].x){
            pair<int, int> reflect2;
            reflect2.first = reflect.second;
            reflect2.second = reflect.first;
            reflect = reflect2;
        }

        directs.emplace_back(make_pair(optimize::norm(position[reflect.first]),optimize::norm(Point2d(-position[reflect.first].y,position[reflect.first].x))));
        directs.emplace_back(make_pair(optimize::norm(position[reflect.second]),optimize::norm(Point2d(-position[reflect.second].y,position[reflect.second].x))));

        //line(em,Point2d(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y),
        //     Point2d(c.x+150*position[reflect.first].x+150*directs[0].first.x,c.y-150*position[reflect.first].y-150*directs[0].first.y),Scalar(0,0,255),2);
        //line(em,Point2d(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y),
        //     Point2d(c.x+150*position[reflect.first].x+150*directs[0].second.x,c.y-150*position[reflect.first].y-150*directs[0].second.y),Scalar(0,0,255),2);
        //line(em,Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y),
        //     Point2d(c.x+150*position[reflect.second].x+150*directs[1].first.x,c.y-150*position[reflect.second].y-150*directs[1].first.y),Scalar(0,0,255),2);
        //line(em,Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y),
        //     Point2d(c.x+150*position[reflect.second].x+150*directs[1].second.x,c.y-150*position[reflect.second].y-150*directs[1].second.y),Scalar(0,0,255),2);

        double index1 = position[reflect.first].y/1.7;
        double index2 = position[reflect.second].y/1.7;
        variance.emplace_back(calc_variance(UV[reflect.first], directs[0], cam, index1));
        variance.emplace_back(calc_variance(UV[reflect.second], directs[1], cam, index2));

        Matrix<double, 4, 1> delta_variable;
        Matrix4d I = Matrix4d::Identity();
        variable(0,0) = position[reflect.first].x;
        variable(1,0) = position[reflect.first].y;
        variable(2,0) = position[reflect.second].x;
        variable(3,0) = position[reflect.second].y;
        variable_origin = variable;

        position[reflect.first] += 0.005*variance[0].first*directs[0].first+0.005*variance[0].second*directs[0].second;
        position[reflect.second] += 0.005*variance[1].first*directs[1].first+0.005*variance[1].second*directs[1].second;
        variable(0,0) = position[reflect.first].x;
        variable(1,0) = position[reflect.first].y;
        variable(2,0) = position[reflect.second].x;
        variable(3,0) = position[reflect.second].y;
/*
        position[reflect.first] += 0.15*variance[0].first*directs[0].first+0.15*variance[0].second*directs[0].second;
        position[reflect.second] += 0.15*variance[1].first*directs[1].first+0.15*variance[1].second*directs[1].second;

        variable(0,0) = position[reflect.first].x;
        variable(1,0) = position[reflect.first].y;
        variable(2,0) = position[reflect.second].x;
        variable(3,0) = position[reflect.second].y;*/
        //
        Point2d zhongxin(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y);
        double angle = atan(position[reflect.first].y/position[reflect.first].x)/M_PI*180;
        angle = 180-angle;
        //ellipse(em,zhongxin,Size(int(2*variance[0].second*150),int(2*variance[0].first*150)),angle,0,360,Scalar(255,0,0),1);
        zhongxin = Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y);
        angle = atan(position[reflect.second].y/position[reflect.second].x)/M_PI*180;
        angle = 180-angle;
        //ellipse(em,zhongxin,Size(int(2*variance[1].second*150),int(2*variance[1].first*150)),angle,0,360,Scalar(255,0,0),1);

        //cout << "std right : " << variance[0].first << "," << variance[0].second << endl;
        //cout << "std left : " << variance[1].first << "," << variance[1].second << endl;
        double lambda = 0.01;
        double v = 2.0;
        double ERROR_CACHE = 0.0,ERROR = 0.0;

        for (int j = 0; j < ite_num; ++j) {
            ERROR = 0.0;
#ifdef _DEBUG
            cout << "lambda : " << lambda << endl;
#endif
            // generate edge
            double e = compute_error(0.135, position[reflect.first], position[reflect.second]);
            Point2d p1 = position[reflect.first];
            Point2d p2 = position[reflect.second];
#ifdef _DEBUG
            cout << "real dist : " << sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2)) << endl;
#endif
            // e对p1求导
            double d = sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
            J(0,0) = -gain*(p1.x-p2.x)/d;
            J(0,1) = -gain*(p1.y-p2.y)/d;
            J(0,2) = gain*(p1.x-p2.x)/d;
            J(0,3) = gain*(p1.y-p2.y)/d;
            edge(0,0) = gain*e;
            ERROR += pow(edge(0,0),2);

            edge(1,0) = m_error(m_coord(position[reflect.first],Point2d(variable_origin(0,0),variable(1,0)),variance[0],directs[0]));
            edge(2,0) = m_error(m_coord(position[reflect.second],Point2d(variable_origin(2,0),variable(3,0)),variance[1],directs[1]));
            pair<double, double> Je_xy_r = Je_xy(Point2d(variable_origin(0,0),variable_origin(1,0)),position[reflect.first],
                                                 variance[0], directs[0]);
            pair<double, double> Je_xy_l = Je_xy(Point2d(variable_origin(2,0),variable_origin(3,0)),position[reflect.second],
                                                 variance[1], directs[1]);
            J(1,0) = Je_xy_r.first;
            J(1,1) = Je_xy_r.second;
            J(2,2) = Je_xy_l.first;
            J(2,3) = Je_xy_l.second;
            ERROR += pow(edge(1,0),2)+pow(edge(2,0),2);

            if (ERROR < min_error)
                break;


            H = J.transpose() * J + lambda * I;
            b = -J.transpose() * edge;
            delta_variable = H.inverse() * b;
#ifdef _DEBUG
            cout << "delta variable : " << delta_variable.transpose() << endl;

            cout << "---------------------------" << endl;
            cout << "Ite : " << j << endl;
            cout << "ERROR : " << ERROR << endl;

            cout << "JT : \n" << J.transpose() << endl;
            cout << "JTJ : \n" << J.transpose()*J <<endl;
            cout << "JTJ-1 : \n" << (J.transpose()*J).inverse() <<endl;
            cout << "edge : \n" << edge << endl;
            cout << "b : \n" << b << endl;
#endif

            // 更新points坐标
            Matrix<double, 4, 1> variable_temp;
            /*if (j == 0)
                variable_temp = variable + delta_variable;
            else*/
                variable_temp = variable + delta_variable;

            vector<Point2d > position_cache = position;
            position_cache[reflect.first] = Point2d(variable_temp(0,0),variable_temp(1,0));
            position_cache[reflect.second] = Point2d(variable_temp(2,0),variable_temp(3,0));

            // 计算ERROR_CACHE
            ERROR_CACHE = 0;
            ERROR_CACHE += gain*gain*pow(compute_error(0.135, position_cache[reflect.first], position_cache[reflect.second]),2);
            cout << ERROR_CACHE << endl;

            ERROR_CACHE += pow(m_error(m_coord(position_cache[reflect.first],Point2d(variable_origin(0,0),variable(1,0)),variance[0],directs[0])),2);
            ERROR_CACHE += pow(m_error(m_coord(position_cache[reflect.second],Point2d(variable_origin(2,0),variable(3,0)),variance[1],directs[1])),2);
#ifdef _DEBUG
            cout << "ERROR_CACHE : " << ERROR_CACHE << endl;
#endif

            double delta_error = ERROR - ERROR_CACHE;
            double delta_appro = -2*delta_variable.transpose() * J.transpose() * edge;
            delta_appro -= delta_variable.transpose() * J.transpose() * J * delta_variable;
            double Q = delta_error / delta_appro;
#ifdef _DEBUG
            cout << "Q : " << Q << endl;
#endif

            //if (j != 0)
            //{
                if (Q > 0) {
#ifdef _DEBUG
                    cout << "update" << endl;
#endif
                    lambda = lambda * max(0.333, 1 - pow(2 * Q - 1, 3));
                    variable = variable_temp;
                    position = position_cache;
                }
            //}
            /*else {
                cout << "update" << endl;
                variable = variable;
                position = position_cache;
            }*/

            if (Q <= 0) {
                lambda *= v;
                v *= 2;
            }
#ifdef _DEBUG
            cout << "---------------------------" << endl;
#endif
        }

        return position;
    }

	// 根据机器人中心的历史位置信息以及当前机器人位置信息以及当前灯条信息
	// 相比于direct_optimize多引入了一个历史位置信息作为约束，确保机器人位置不会发散
    Point3d optimize::center_optimize(int ite_num, const vector<Point2d > &UV, vector<Point2d > position,
            pair<int, int> &reflect, Point3d center, int state, Camera &cam, vector<Point3d > &history_center, Mat &em, Point2d c,
                                      vector<Point3d > &origin_center) {
        double gain = 2;
        double low_gain = 0.6;
        vector<double > angles;
        Matrix<double, 3, 1> variable;
        Matrix<double, 3, 1> variable_origin;
        Matrix3d sigma = Matrix3d::Zero();
        Matrix3d sigma_sqrt = Matrix3d::Zero();
        Matrix3d H = Matrix3d::Zero();
        Matrix<double, 3, 1> b;
        Matrix<double, 8, 3> J = Matrix<double, 8, 3>::Zero(); // (variable.size(),error.size())
        Matrix<double, 8, 1> edge;
        Matrix<double, 3, 1> delta_variable;
        Matrix3d I = Matrix3d::Identity();
        vector<pair<Point2d, Point2d>> directs;
        vector<pair<double, double>> variance;
        Point3d last_center = history_center[history_center.size()-1];
        vector<double > v_history;
        vector<double > valocity;
        //Point3d center15(0,0,0), variance15(0,0,0);
        //Point3d center5(0,0,0), variance5(0,0,0);
        //circle(em,Point2d(c.x+last_center.x*150,c.y-last_center.y*150),4,Scalar(255,0,0),2);

        // 自适应
        /*
        if (origin_center.size() > 15){
            for (unsigned long i = origin_center.size()-15; i < origin_center.size(); ++i)
                center15 += origin_center[i];
            center15 = center15/15.0;
            for (unsigned long i = origin_center.size()-15; i < origin_center.size(); ++i) {
                variance15.x += pow(origin_center[i].x-center15.x,2);
                variance15.y += pow(origin_center[i].y-center15.y,2);
                variance15.z += pow(origin_center[i].z-center15.z,2);
            }
            variance15 = variance15/15.0;

            for (unsigned long k = origin_center.size()-5; k < origin_center.size(); ++k)
                center5 += origin_center[k];
            center5 = center5/5.0;
            for (unsigned long i = origin_center.size()-5; i < origin_center.size(); ++i) {
                variance5.x += pow(origin_center[i].x-center5.x,2);
                variance5.y += pow(origin_center[i].y-center5.y,2);
                variance5.z += pow(origin_center[i].z-center5.z,2);
            }
            variance5 = variance5/5.0;
        }*/


        v_history.push_back(last_center.x);
        v_history.push_back(last_center.y);
        v_history.push_back(last_center.z);
        if (abs(v_history[2]-center.z) > 25){
            if (v_history[2] > 0)
                v_history[2] -= 90;
            else
                v_history[2] += 90;
        }

        if (history_center.size() > 2) {
            valocity.push_back(30 * (history_center[history_center.size() - 1].x - history_center[history_center.size() - 2].x));
            valocity.push_back(30 * (history_center[history_center.size() - 1].y - history_center[history_center.size() - 2].y));
            valocity.push_back(30 * (history_center[history_center.size() - 1].z - history_center[history_center.size() - 2].z));
        }

        // right-left
        directs.emplace_back(make_pair(optimize::norm(position[reflect.first]),optimize::norm(Point2d(-position[reflect.first].y,position[reflect.first].x))));
        directs.emplace_back(make_pair(optimize::norm(position[reflect.second]),optimize::norm(Point2d(-position[reflect.second].y,position[reflect.second].x))));
        //cout << "right axis : " << directs[0].first.x << "," << directs[0].first.y << "--" << directs[0].second.x << "," << directs[0].second.y << endl;
        //cout << "left axis : " << directs[1].first.x << "," << directs[1].first.y << "--" << directs[1].second.x << "," << directs[1].second.y << endl;

        //
        line(em,Point2d(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y),
             Point2d(c.x+150*position[reflect.first].x+150*directs[0].first.x,c.y-150*position[reflect.first].y-150*directs[0].first.y),Scalar(0,0,255),1);
        line(em,Point2d(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y),
             Point2d(c.x+150*position[reflect.first].x+150*directs[0].second.x,c.y-150*position[reflect.first].y-150*directs[0].second.y),Scalar(0,0,255),1);
        line(em,Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y),
             Point2d(c.x+150*position[reflect.second].x+150*directs[1].first.x,c.y-150*position[reflect.second].y-150*directs[1].first.y),Scalar(0,0,255),1);
        line(em,Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y),
             Point2d(c.x+150*position[reflect.second].x+150*directs[1].second.x,c.y-150*position[reflect.second].y-150*directs[1].second.y),Scalar(0,0,255),1);
        //

        // right-left
        double index1 = position[reflect.first].y/1.7;
        double index2 = position[reflect.second].y/1.7;
        variance.emplace_back(calc_variance(UV[reflect.first], directs[0], cam, index1));
        variance.emplace_back(calc_variance(UV[reflect.second], directs[1], cam, index2));
        //cout << "std right : " << variance[0].first << "," << variance[0].second << endl;
        //cout << "std left : " << variance[1].first << "," << variance[1].second << endl;

        //
        Point2d zhongxin(c.x+150*position[reflect.first].x,c.y-150*position[reflect.first].y);
        double angle = atan(position[reflect.first].y/position[reflect.first].x)/M_PI*180;
        angle = 180-angle;
        ellipse(em,zhongxin,Size(int(2*variance[0].second*150),int(2*variance[0].first*150)),angle,0,360,Scalar(255,0,0),1);
        zhongxin = Point2d(c.x+150*position[reflect.second].x,c.y-150*position[reflect.second].y);
        angle = atan(position[reflect.second].y/position[reflect.second].x)/M_PI*180;
        angle = 180-angle;
        ellipse(em,zhongxin,Size(int(2*variance[1].second*150),int(2*variance[1].first*150)),angle,0,360,Scalar(255,0,0),1);

        // x,y,theta的std分别是2cm,2cm,2度
        sigma(0,0) = pow(0.03,2);
        sigma(1,1) = pow(0.03,2);
        sigma(2,2) = pow(2.5,2);
        //cout << "sigma : \n" << sigma << endl;
        sigma = sigma.inverse().eval();
        for (int i = 0; i < 3; ++i)
            sigma_sqrt(i,i) = sqrt(sigma(i,i));

        variable(0,0) = center.x;
        variable(1,0) = center.y;
        variable(2,0) = center.z;
        variable_origin = variable;

        circle(em,Point2d(c.x+150*center.x,c.y-150*center.y),3,Scalar(0,255,0),1);

        double lambda = 0.1;
        double v = 2.0;
        double ERROR_CACHE = 0.0,ERROR = 0.0;

        for (int j = 0; j < ite_num; ++j) {
            ERROR = 0.0;
            vector<double > v_center;
            v_center.push_back(center.x);
            v_center.push_back(center.y);
            v_center.push_back(center.z);

            // generate edge
            for (int m = 0; m < 3; ++m) {
                double e = (v_history[m]-v_center[m])*sigma_sqrt(m,m);
                // e对p1求导
                J(m,m) = -gain*sigma_sqrt(m,m);
                edge(m,0) = gain*e;
                ERROR += pow(e,2);
            }

            // state=1代表0.187848侧装甲板，state=2代表0.268348正面装加板
            pair<Point2d, Point2d> r_l = center2p(Point2d(center.x,center.y),center.z,state);

            //cout << "right : " << r_l.first.x << "," << r_l.first.y << endl;
            //cout << "left : " << r_l.second.x << "," << r_l.second.y << endl;

            // d_distance/d_xytheta通过数值解法求
            edge(3,0) = m_error(m_coord(position[reflect.first],r_l.first,variance[0],directs[0]));
            edge(4,0) = m_error(m_coord(position[reflect.second],r_l.second,variance[1],directs[1]));
            pair<Point3d, Point3d> Je = Je_xyt(center,make_pair(position[reflect.first],position[reflect.second]),variance,state,directs);
            J(3,0) = Je.first.x;
            J(3,1) = Je.first.y;
            J(3,2) = Je.first.z;
            J(4,0) = Je.second.x;
            J(4,1) = Je.second.y;
            J(4,2) = Je.second.z;
            ERROR += pow(edge(3,0),2)+pow(edge(4,0),2);

            // generate edge
            for (int m = 5; m < 8; ++m) {
                double e = (v_center[m-5]-variable_origin(m-5,0))*sigma_sqrt(m-5,m-5);
                // e对p1求导
                J(m,m-5) = low_gain*sigma_sqrt(m-5,m-5);
                edge(m,0) = low_gain*e;
                ERROR += pow(e,2);
            }

            if (ERROR < min_error)
                break;

            H = J.transpose() * J + lambda * I;
            b = -J.transpose() * edge;
            delta_variable = H.inverse() * b;
            /*
            cout << "---------------------------" << endl;
            cout << "delta variable : " << delta_variable.transpose() << endl;
            cout << "Ite : " << j << endl;
            cout << "ERROR : " << ERROR << endl;
            cout << "JT : \n" << J.transpose() << endl;
            cout << "JTJ : \n" << J.transpose()*J <<endl;
            cout << "JTJ-1 : \n" << (J.transpose()*J).inverse() <<endl;
            cout << "edge : \n" << edge << endl;
            cout << "b : \n" << b << endl;*/

            // 更新points坐标
            Matrix<double, 3, 1> variable_temp = variable + delta_variable;
            vector<Point2d > position_cache;

            // 计算ERROR_CACHE
            ERROR_CACHE = 0;
            for (int m = 0; m < 3; ++m) {
                double e = (v_history[m]-variable_temp(m,0))*sigma_sqrt(m,m);
                ERROR_CACHE += low_gain*low_gain*pow(e,2);
            }

            // state=1代表0.187848侧装加板，state=2代表0.268348正面装加板
            r_l = center2p(Point2d(variable_temp(0,0),variable_temp(1,0)),variable_temp(2,0),state);
            double e1 = m_error(m_coord(position[reflect.first],r_l.first,variance[0],directs[0]));
            double e2 = m_error(m_coord(position[reflect.second],r_l.second,variance[1],directs[1]));
            ERROR_CACHE += pow(e1,2)+pow(e2,2);

            // generate edge
            for (int m = 5; m < 8; ++m) {
                double e = (v_center[m-5]-variable_origin(m-5,0))*sigma_sqrt(m-5,m-5);
                ERROR_CACHE += pow(e,2);
            }
#ifdef _DEBUG
            cout << "ERROR_CACHE : " << ERROR_CACHE << endl;
#endif

            double delta_error = ERROR - ERROR_CACHE;
            double delta_appro = -2*delta_variable.transpose() * J.transpose() * edge;
            delta_appro -= delta_variable.transpose() * J.transpose() * J * delta_variable;
            double Q = delta_error / delta_appro;
#ifdef _DEBUG
            cout << "Q : " << Q << endl;
#endif

            if (Q > 0) {
#ifdef _DEBUG
                cout << "update" << endl;
#endif
                lambda = lambda * max(0.333, 1 - pow(2 * Q - 1, 3));
                center.x = variable_temp(0,0);
                center.y = variable_temp(1,0);
                center.z = variable_temp(2,0);
                variable = variable_temp;
            }

            if (Q <= 0) {
                lambda *= v;
                v *= 2;
            }
#ifdef _DEBUG
            cout << "---------------------------" << endl;
#endif
        }


        double d = dist(Point2d(center.x,center.y),Point2d(last_center.x,last_center.y));
        if (d > 0.05 && d < 0.09)
            //history_center.clear();
            center = last_center;
        else if (d > 0.09) {
            history_center.clear();
            return last_center;
        }


        history_center.push_back(center);
#ifdef _DEBUG
        cout << "optimized center : " << center.x << "," << center.y << endl;
#endif
        //circle(em,Point2d(c.x+150*center.x,c.y-150*center.y),6,Scalar(0,0,0),2);

        return center;
    }

}

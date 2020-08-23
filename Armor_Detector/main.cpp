#include <iostream>
#include <string.h>
#include <fstream>

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

// 麦克纳姆轮的正反模板
/*string POSITIVE = "/home/mrm/ICRA/CODE/SIGN/POSITIVE/";
string NEGTIVE = "/home/mrm/ICRA/CODE/SIGN/NEGTIVE/";*/
string POSITIVE = "../SIGN/POSITIVE/";
string NEGTIVE = "../SIGN/NEGTIVE/";

// 灯条的正反模板
/*string LPOSITIVE = "/home/mrm/ICRA/CODE/SIGN/POSITIVE_LIGHT/";
string LNEGTIVE = "/home/mrm/ICRA/CODE/SIGN/NEGTIVE_LIGHT/";*/
string LPOSITIVE = "../SIGN/POSITIVE_LIGHT/";
string LNEGTIVE = "../SIGN/NEGTIVE_LIGHT/";

// 麦克纳姆轮的正反模板参数
/*string POSITIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/POSITIVE/config.txt";
string NEGTIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/NEGTIVE/config.txt";*/
string POSITIVE_CONFIG = "../SIGN/POSITIVE/config.txt";
string NEGTIVE_CONFIG = "../SIGN/NEGTIVE/config.txt";

// 灯条的正反模板参数
/*string LPOSITIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/POSITIVE_LIGHT/config.txt";
string LNEGTIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/NEGTIVE_LIGHT/config.txt";
string VIDEO_PATH = "/home/mrm/ICRA/CODE/Dahuacam/video/demo.avi";*/
string LPOSITIVE_CONFIG = "../SIGN/POSITIVE_LIGHT/config.txt";
string LNEGTIVE_CONFIG = "../SIGN/NEGTIVE_LIGHT/config.txt";
string VIDEO_PATH = "../video/demo.avi";

// 识别结果存储路径
/*string ODOMETRY = "/home/mrm/path.txt";
string ODOMETRYOP = "/home/mrm/pathop.txt";
string PDFCENTER = "/home/mrm/pdfcenter.txt";*/
string ODOMETRY = "../output/path.txt";
string ODOMETRYOP = "../output/pathop.txt";
string PDFCENTER = "../output/pdfcenter.txt";

// 模板存储容器
bool negtive_flag = false; // 用来决定麦克纳姆轮正反匹配方式
double ave_y = 0.0; // 灯条距离己方机器人坐标系的y方向平均坐标
vector<pair<double, int>> positive_data, light_positive_data;
vector<pair<double, int>> negtive_data, light_negtive_data;
vector<Mat > positive, light_positive;
vector<Mat > negtive, light_negtive;

Point2d find_target(float x_1, float y_1, float t_1);

// 读取模板
void read_data()
{
    ifstream positive_reader(POSITIVE_CONFIG);
    ifstream negtive_reader(NEGTIVE_CONFIG);
    ifstream light_positive_reader(LPOSITIVE_CONFIG);
    ifstream light_negtive_reader(LNEGTIVE_CONFIG);
    if (!positive_reader.is_open() || !negtive_reader.is_open() || !light_positive_reader.is_open() || !light_negtive_reader.is_open()){
        cout << "config files error !" << endl;
        return;
    }

    while (!positive_reader.eof()){
        double ratio;
        int index;
        string c;
        for (int i = 0; i < 4; ++i) {
            if (i == 0)
                positive_reader >> index;
            else if (i == 3)
                positive_reader >> ratio;
            else
                positive_reader >> c;
        }
        positive_data.emplace_back(make_pair(ratio,index));
        Mat picture = imread(POSITIVE+to_string(index)+".png",0);
        positive.push_back(picture);
    }

    while (!negtive_reader.eof()){
        double ratio;
        int index;
        string c;
        for (int i = 0; i < 4; ++i) {
            if (i == 0)
                negtive_reader >> index;
            else if (i == 3)
                negtive_reader >> ratio;
            else
                negtive_reader >> c;
        }
        negtive_data.emplace_back(make_pair(ratio,index));
        Mat picture = imread(NEGTIVE+to_string(index)+".png",0);
        negtive.push_back(picture);
    }

    while (!light_positive_reader.eof()){
        double ratio;
        int index;
        string c;
        for (int i = 0; i < 4; ++i) {
            if (i == 0)
                light_positive_reader >> index;
            else if (i == 3)
                light_positive_reader >> ratio;
            else
                light_positive_reader >> c;
        }
        light_positive_data.emplace_back(make_pair(ratio,index));
        Mat picture = imread(LPOSITIVE+to_string(index)+".png",0);
        light_positive.push_back(picture);
    }

    while (!light_negtive_reader.eof()){
        double ratio;
        int index;
        string c;
        for (int i = 0; i < 4; ++i) {
            if (i == 0)
                light_negtive_reader >> index;
            else if (i == 3)
                light_negtive_reader >> ratio;
            else
                light_negtive_reader >> c;
        }
        light_negtive_data.emplace_back(make_pair(ratio,index));
        Mat picture = imread(LNEGTIVE+to_string(index)+".png",0);
        light_negtive.push_back(picture);
    }
    light_negtive.pop_back();
    light_positive.pop_back();
    negtive_data.pop_back();
    positive_data.pop_back();
    light_negtive_data.pop_back();
    light_positive_data.pop_back();
    negtive.pop_back();
    positive.pop_back();

    positive_reader.close();
    negtive_reader.close();
    light_negtive_reader.close();
    light_positive_reader.close();
}

// 判断矩形是否在图片内
bool rectinimage(const Rect &rect, const Mat &image_temp)
{
    if (rect.x < 0)
        return false;
        //rect.x = 0;
    else if (rect.x > image_temp.cols)
        return false;

    if (rect.x + rect.width > image_temp.cols)
        return false;
    //rect.width = image_temp.cols - rect.x -1;

    if (rect.y < 0)
        return false;
        //rect.y = 0;
    else if (rect.y > image_temp.rows)
        return false;

    if (rect.y + rect.height > image_temp.rows)
        return false;
    //rect.height = image_temp.rows - rect.y -1;

    return true;
}

// 将原矩形扩大，用于在更大范围内寻找合适的模板
void refine(const Mat &image, Rect &rect)
{
    if (rect.empty())
        return;

    if (rect.x - 30 < 0)
        rect.x = 0;
    else if (rect.x - 30 >= 0)
        rect.x = rect.x - 30;

    if (rect.y - 30 < 0)
        rect.y = 0;
    else if (rect.y -30 >= 0)
        rect.y = rect.y - 30;

    if (rect.x + rect.width + 60 > image.cols)
        rect.width = image.cols-rect.x-1;
    else if (rect.x + rect.width + 60 <= image.cols)
        rect.width = rect.width + 60;

    if (rect.y + rect.height + 60 > image.rows)
        rect.height = image.rows-rect.y-1;
    else if (rect.y + rect.height + 60 <= image.rows)
        rect.height = rect.height + 60;

}

// 假设血量灯条边界框内部的白色点应该大于某一阈值，返回的值以0.3作为上界
// 判断血量灯条边界框是否符合假设
int prompt(const Mat &image_temp, const Mat &hsv, int xmin, int ymin, const Rect &light_max_rect)
{
    double white_count = 0.0;
    Rect hsv_rect;
    if (light_max_rect.height > 30){
        hsv_rect = Rect(light_max_rect.x-xmin+20,light_max_rect.y-ymin+20,light_max_rect.width,light_max_rect.height);
        if (hsv_rect.x > hsv.cols || hsv_rect.y > hsv.rows || hsv_rect.x + hsv_rect.width < 0 || hsv_rect.y + hsv_rect.height < 0)
            return 1;
        if (hsv_rect.x < 0)
            hsv_rect.x = 0;
        if (hsv_rect.y < 0)
            hsv_rect.y = 0;
        if (hsv_rect.x + hsv_rect.width > hsv.cols)
            hsv_rect.width = hsv.cols - hsv_rect.x -1;
        if (hsv_rect.y + hsv_rect.height > hsv.rows)
            hsv_rect.height = hsv.rows - hsv_rect.y -1;

        for (int j = hsv_rect.y; j < hsv_rect.y+hsv_rect.height; ++j) {
            for (int i = hsv_rect.x; i < hsv_rect.x+hsv_rect.width; ++i) {
                if (hsv.ptr<uchar >(j)[i] > 230)
                    white_count++;
            }
        }
    }
    else if (light_max_rect.height < 30){
        hsv_rect = light_max_rect;
        Mat temp = image_temp(hsv_rect);
        if (temp.empty())
            return 1;
        cvtColor(temp,temp,CV_RGB2GRAY);
        threshold(temp,temp,200,255,THRESH_BINARY);
        for (int j = 0; j < temp.rows; ++j) {
            for (int i = 0; i < temp.cols; ++i) {
                if (temp.ptr<uchar >(j)[i] > 230)
                    white_count++;
            }
        }
    }

    double area = hsv_rect.area();
#ifdef _DEBUG
    cout << "white_count : " << white_count << endl;
#endif
    if (white_count < 5.0) {
        return 1;
    }
    else if (white_count/area > 0.3)
        return 2;

    return 3;
}


// 计算和输入的麦克纳姆轮或后方灯条图片最符合的模板及匹配度
double prob(const Mat &image_temp, const Rect &rect, bool negtive_flag, Rect &max_prob_rect, int state)
{
    assert(state == 1 || state == 2);
    // state = 1代表车轮的匹配2代表灯条的匹配
    if (state == 1)
    {
        Mat target, temp;
        if (rect.empty())
            return 0.0;

        if (!rectinimage(rect,image_temp))
            return 0.0;

        // 如果输入的图片高度过大，则进行缩放
        double s = 60.0/rect.height;
        if (s > 1)
            s = 1.0;
        image_temp(rect).copyTo(target);
        resize(target,target,Size(int(s*rect.width),int(s*rect.height)),0,0,INTER_LINEAR);
        double ratio = double(rect.height)/rect.width;
        vector<Mat > candicate;
        //cvtColor(target,target,CV_RGB2GRAY);

        // 遍历模板找到模板中和当前图像属性最相近的一组候选模板
        if (negtive_flag){
            for (int i = 0; i < negtive_data.size(); ++i) {
                double r = negtive_data[i].first/ratio;
                if (r < 1)
                    r = 1.0/r;
                if (r < 1.3){
                    temp = negtive[i];
                    double scale;
                    scale = double(target.rows)/temp.rows;
                    resize(temp,temp,Size(int(temp.cols*scale)-1,target.rows),0,0,INTER_LINEAR);
                    equalizeHist(temp,temp);
                    candicate.push_back(temp);
                }
            }
        }
        else {
            for (int i = 0; i < positive_data.size(); ++i) {
                double r = positive_data[i].first/ratio;
                if (r < 1)
                    r = 1.0/r;
                if (r < 1.3){
                    temp = positive[i];
                    double scale;
                    scale = double(target.cols)/temp.cols;
                    resize(temp,temp,Size(int(temp.cols*scale)-1,target.rows),0,0,INTER_LINEAR);
                    equalizeHist(temp,temp);
                    candicate.push_back(temp);
                }
            }
        }

#ifdef _DEBUG
        cout << "candicate.size : " << candicate.size() << endl;
#endif
        if (candicate.empty())
            return 0.0;

        double max_prob = -1;
        Rect rect2;
        double width_ratio = 1, height_ratio = 1;

        // 根据计算出的边界框大小适当的扩大搜索范围
        if (rect.width < 30)
            width_ratio = 0.7;
        else if (rect.width >= 30 && rect.width < 50)
            width_ratio = 0.3;
        else if (rect.width >= 50 && rect.width < 80)
            width_ratio = 0.25;
        else if (rect.width >= 80 && rect.width <120)
            width_ratio = 0.2;
        else if (rect.width >= 120)
            width_ratio = 0.15;

        if (rect.height < 50)
            height_ratio = 0.2;
        else if (rect.height >= 50 && rect.height <80)
            height_ratio = 0.15;
        else if (rect.height >= 80 && rect.height <120)
            height_ratio = 0.1;
        else if (rect.height >= 120)
            height_ratio = 0.06;

        rect2 = Rect(int(rect.x-rect.width*width_ratio),int(rect.y-rect.height*height_ratio),
                int(rect.width*(1+2*width_ratio)),int(rect.height*(1+2*height_ratio)));
        if (!rectinimage(rect2,image_temp))
            return 0.0;

        Mat target2;
        image_temp(rect2).copyTo(target2);
        resize(target2,target2,Size(int(target2.cols*s),int(target2.rows*s)),0,0,INTER_LINEAR);
        cvtColor(target2,target2,CV_RGB2GRAY);
        // 为了防止亮度影响，进行直方图均衡化处理
        equalizeHist(target2,target2);
        // 在目标区域内进行逐位置匹配，寻找最符合的位置、最符合的模板以及最大匹配概率
        for (int j = 0; j < candicate.size(); ++j) {
            int result_cols = target2.cols - candicate[j].cols + 1;
            int result_rows = target2.rows - candicate[j].rows + 1;
            if (result_cols >= 1 && result_rows >= 1){
                Mat result(result_cols, result_rows, CV_32FC1,Scalar::all(0));
                matchTemplate(target2,candicate[j],result,TM_CCOEFF_NORMED);   //最好匹配为1,值越小匹配越差
                double minVal = -1;
                double maxVal;
                Point minLoc;
                Point maxLoc;
                minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                if (maxVal > max_prob){
                    max_prob = maxVal;
                    max_prob_rect = Rect(int(maxLoc.x/s),int(maxLoc.y/s),int(candicate[j].cols/s),int(candicate[j].rows/s));
                }
            }
        }
        max_prob_rect.x += rect2.x;
        max_prob_rect.y += rect2.y;
        return max_prob;
    }
    else if (state == 2){
        // 对灯条的匹配同理
        Mat target, temp;
        if (rect.empty())
            return 0.0;

        if (!rectinimage(rect,image_temp))
            return 0.0;

        double s = 25.0/rect.height;
        if (s > 1)
            s = 1.0;
        image_temp(rect).copyTo(target);
        resize(target,target,Size(int(s*rect.width),int(s*rect.height)),0,0,INTER_LINEAR);
        double ratio = double(rect.height)/rect.width;
        vector<Mat > candicate;
        //cvtColor(target,target,CV_RGB2GRAY);
        if (negtive_flag){
            for (int i = 0; i < light_negtive_data.size(); ++i) {
                double r = light_negtive_data[i].first/ratio;
                if (r < 1)
                    r = 1.0/r;
                if (r < 1.2){
                    temp = light_negtive[i];
                    double scale;
                    scale = double(target.rows)/temp.rows;
                    resize(temp,temp,Size(int(temp.cols*scale)-1,target.rows),0,0,INTER_LINEAR);
                    //equalizeHist(temp,temp);
                    candicate.push_back(temp);
                }
            }
        }
        else {
            for (int i = 0; i < light_positive_data.size(); ++i) {
                double r = light_positive_data[i].first/ratio;
                if (r < 1)
                    r = 1.0/r;
                if (r < 1.2){
                    temp = light_positive[i];
                    double scale;
                    scale = double(target.cols)/temp.cols;
                    resize(temp,temp,Size(int(temp.cols*scale)-1,target.rows),0,0,INTER_LINEAR);
                    //equalizeHist(temp,temp);
                    candicate.push_back(temp);
                }
            }
        }

#ifdef _DEBUG
        cout << "candicate.size : " << candicate.size() << endl;
#endif
        if (candicate.empty())
            return 0.0;

        double max_prob = -1;
        Rect rect2;
        double width_ratio = 1, height_ratio = 1;
        if (rect.width < 30)
            width_ratio = 0.7;
        else if (rect.width >= 30 && rect.width < 60)
            width_ratio = 0.3;
        else if (rect.width >= 60 && rect.width < 80)
            width_ratio = 0.25;
        else if (rect.width >= 80 && rect.width <120)
            width_ratio = 0.2;
        else if (rect.width >= 120)
            width_ratio = 0.1;

        if (rect.height < 30)
            height_ratio = 0.5;
        else if (rect.height >= 30 && rect.height <80)
            height_ratio = 0.3;
        else if (rect.height >= 80 && rect.height <120)
            height_ratio = 0.1;
        else if (rect.height >= 120)
            height_ratio = 0.06;

        rect2 = Rect(int(rect.x-rect.width*width_ratio),int(rect.y-rect.height*height_ratio),
                     int(rect.width*(1+2*width_ratio)),int(rect.height*(1+2*height_ratio)));

        if (!rectinimage(rect2,image_temp))
            return 0.0;

        Mat target2;
        image_temp(rect2).copyTo(target2);
        resize(target2,target2,Size(int(target2.cols*s),int(target2.rows*s)),0,0,INTER_LINEAR);
        cvtColor(target2,target2,CV_RGB2GRAY);
        equalizeHist(target2,target2);
        for (int j = 0; j < candicate.size(); ++j) {
            int result_cols = target2.cols - candicate[j].cols + 1;
            int result_rows = target2.rows - candicate[j].rows + 1;
            if (result_cols >= 1 && result_rows >= 1){
                Mat result(result_cols, result_rows, CV_32FC1,Scalar::all(0));
                matchTemplate(target2,candicate[j],result,TM_CCOEFF_NORMED);   //最好匹配为1,值越小匹配越差
                double minVal = -1;
                double maxVal;
                Point minLoc;
                Point maxLoc;
                minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
                if (maxVal > max_prob){
                    max_prob = maxVal;
                    max_prob_rect = Rect(int(maxLoc.x/s),int(maxLoc.y/s),int(candicate[j].cols/s),int(candicate[j].rows/s));
                }
            }
        }
        max_prob_rect.x += rect2.x;
        max_prob_rect.y += rect2.y;
        return max_prob;
    }
}

vector<Point2d > pnp(pair<int, int> armor,vector<Lightbar* > &lightbars,Camera &camera,Matrix4d Tw_c,double xmin, double ymin)
{
    vector<Point2f > camerapoints;
    camerapoints.push_back(lightbars[armor.first]->downcenter+Point2f(-20+xmin,-20+ymin));
    camerapoints.push_back(lightbars[armor.first]->upcenter+Point2f(-20+xmin,-20+ymin));
    camerapoints.push_back(lightbars[armor.second]->upcenter+Point2f(-20+xmin,-20+ymin));
    camerapoints.push_back(lightbars[armor.second]->downcenter+Point2f(-20+xmin,-20+ymin));
#ifdef _DEBUG
    cout << "p1p : " << camerapoints[0].x << "," << camerapoints[0].y << endl;
    cout << "      " << lightbars[armor.first]->downcenter.x << "," << lightbars[armor.first]->downcenter.y << endl;
    cout << "p2p : " << camerapoints[1].x << "," << camerapoints[1].y << endl;
    cout << "      " << lightbars[armor.first]->upcenter.x << "," << lightbars[armor.first]->upcenter.y << endl;
    cout << "p3p : " << camerapoints[2].x << "," << camerapoints[2].y << endl;
    cout << "      " << lightbars[armor.second]->upcenter.x << "," << lightbars[armor.second]->upcenter.y << endl;
    cout << "p4p : " << camerapoints[3].x << "," << camerapoints[3].y << endl;
    cout << "      " << lightbars[armor.second]->downcenter.x << "," << lightbars[armor.second]->downcenter.y << endl;
#endif

    vector<Point3f > objectpoints;
    objectpoints.emplace_back(Point3f(0.0675,-0.027f,0));
    objectpoints.emplace_back(Point3f(0.0675,0.027f,0));
    objectpoints.emplace_back(Point3f(-0.0675f,0.027f,0));
    objectpoints.emplace_back(Point3f(-0.0675f,-0.027f,0));
    Mat rvec;
    Mat tvec;
    Mat rotation = Mat_<double >(3,3);
    solvePnP(objectpoints, camerapoints, camera.matrix, camera.coeff, rvec, tvec);
    Rodrigues(rvec,rotation);
#ifdef _DEBUG
    cout << "rotation : " << rotation << endl;
    cout << "tvec : " << tvec << endl;
    cout << "rvec : " << rvec << endl;
#endif
    Matrix4d Tg_a; // Tgimbal_armor
    Tg_a << rotation.at<double >(0,0),rotation.at<double >(0,1),rotation.at<double >(0,2),tvec.at<double >(0,0),
            rotation.at<double >(1,0),rotation.at<double >(1,1),rotation.at<double >(1,2),tvec.at<double >(1,0),
            rotation.at<double >(2,0),rotation.at<double >(2,1),rotation.at<double >(2,2),tvec.at<double >(2,0),
                    0,0,0,1;
#ifdef _DEBUG
    cout << "Tg_a : " << Tg_a << endl;
#endif
    Matrix4d points;
    points << 0.0675,0.0675,-0.0675,-0.0675,-0.027,0.027,0.027,-0.027,0,0,0,0,1,1,1,1;
#ifdef _DEBUG
    cout << "points : " << points << endl;
#endif
    points = Tg_a*points;
#ifdef _DEBUG
    cout << "points : " << points << endl;
#endif
    points = Tw_c*points;
    vector<Point3d > p3d;
    for (int i = 0; i < 4; ++i) {
        p3d.emplace_back(Point3d(points(0, i), points(1, i), points(2, i)));
#ifdef _DEBUG
        cout << "pp : " << points(0, i) << "," << points(1, i) << "," << points(2, i) << endl;
#endif
    }

    vector<Point2d > new_position;
    new_position.emplace_back(Point2d((p3d[0].x+p3d[1].x)/2,(p3d[0].y+p3d[1].y)/2));
    new_position.emplace_back(Point2d((p3d[2].x+p3d[3].x)/2,(p3d[2].y+p3d[3].y)/2));
    return new_position;
}

// 根据机器人的正方向信息，将机器人角度归到统一的坐标系
double robotheta(const Enemy &enemy)
{
    if (enemy.positive_y_axis.y >= 0 && enemy.positive_y_axis.x > 0)
        return atan(enemy.positive_y_axis.y/enemy.positive_y_axis.x)*180/M_PI;
    else if (enemy.positive_y_axis.y >= 0 && enemy.positive_y_axis.x < 0)
        return 180+atan(enemy.positive_y_axis.y/enemy.positive_y_axis.x)*180/M_PI;
    else if (enemy.positive_y_axis.y < 0 && enemy.positive_y_axis.x > 0)
        return 360+atan(enemy.positive_y_axis.y/enemy.positive_y_axis.x)*180/M_PI;
    else if (enemy.positive_y_axis.y < 0 && enemy.positive_y_axis.x < 0)
        return 180+atan(enemy.positive_y_axis.y/enemy.positive_y_axis.x)*180/M_PI;
    else if (enemy.positive_y_axis.y >= 0 && enemy.positive_y_axis.x == 0)
        return 90;
    else if (enemy.positive_y_axis.y < 0 && enemy.positive_y_axis.x == 0)
        return 270;
}

// 返回两个点的距离
double dist(const Point2d &p1, const Point2d &p2)
{
    return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
}

double trans(double t)
{
    if (t < 0)
        t = -t;
    else
        t = 180 - t;
    return t;
}

void exchange(Point2d &p1, Point2d &p2)
{
    double y_temp;
    y_temp = p1.y;
    p1.y = p2.y;
    p2.y = y_temp;
}

// 根据识别出来的灯条位置判断哪两个灯条是同一个装甲板的
pair<int, int> find_pair(const vector<Point2d> &UV, const vector<Point2d > &lightbars_position, const vector<double > &lines)
{
    pair<int, int> mapping;
    bool success = false;
    for (int i = 0; i < lightbars_position.size(); ++i) {
        for (int j = 0; j < lightbars_position.size(); ++j) {
            if (i != j){
                if (dist(UV[i],UV[j]) > 15){
                    if (dist(lightbars_position[i],lightbars_position[j]) - 0.135 < 0.04){
                        int z = 3-i-j;
                        double t1,t2,t3;
                        t1 = trans(atan(lines[z])*180/M_PI);
                        t2 = trans(atan(lines[i])*180/M_PI);
                        t3 = trans(atan(lines[j])*180/M_PI);
                        if (ave_y < 3){
                            if (abs(t1-t2) > abs(t2-t3) && abs(t1-t3) > abs(t2-t3))
                            {
                                mapping.first = i;
                                mapping.second = j;
                                success = true;
                            }
                        }
                        else if (ave_y >= 3){
                            mapping.first = i;
                            mapping.second = j;
                            success = true;
                        }
                    }
                }
            }
        }
    }

    if (!success)
        return make_pair(0,0);

    return mapping;
}

int main() {
    unsigned long lo = 0;
    ofstream odometry(ODOMETRY);
    ofstream odometryop(ODOMETRYOP);
    ofstream pdfcenter(PDFCENTER);
    if (!odometry.is_open() || !odometryop.is_open()){
        cout << "odometry file is not opened correct" << endl;
        return -1;
    }

    int stateNum = 3;
    int measureNum = 1;
    double delta = 1;
    double alpha = 10;
    double alpha2 = 150;

    // 初始化滤波器，分别对三个方向进行滤波
    SKF KF(stateNum,measureNum,0.2,10);
    setIdentity(KF.processNoiseCov,Scalar::all(0.0001));
    setIdentity(KF.measurementNoiseCov,Scalar::all(0.01));
    KF.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
    KF.measurementMatrix.at<float>(0,0) = 1;
    randn(KF.statePost,Scalar::all(-1),Scalar::all(1));
    setIdentity(KF.errorCovPost,Scalar::all(1));

    KF.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
            0,1,(1-exp(-alpha*delta))/alpha,
            0,0,exp(-alpha*delta));
#ifdef _DEBUG
    cout << "KF.transitionMatrix : " << KF.transitionMatrix << endl;
    cout << "KF.statePost : " << KF.statePost << endl;
    cout << "KF.measure : " << KF.measurementMatrix << endl;
#endif
    SKF KF2(stateNum,measureNum,0.2,10);
    setIdentity(KF2.processNoiseCov,Scalar::all(0.001));
    setIdentity(KF2.measurementNoiseCov,Scalar::all(0.01));
    KF2.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
    KF2.measurementMatrix.at<float>(0,0) = 1;
    randn(KF2.statePost,Scalar::all(-1),Scalar::all(1));
    setIdentity(KF2.errorCovPost,Scalar::all(1));

    KF2.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
            0,1,(1-exp(-alpha*delta))/alpha,
            0,0,exp(-alpha*delta));

    SKF KF3(stateNum,measureNum,0.2,10);
    setIdentity(KF3.processNoiseCov,Scalar::all(0.001));
    setIdentity(KF3.measurementNoiseCov,Scalar::all(0.01));
    KF3.measurementMatrix = Mat(measureNum,stateNum,CV_32F,Scalar::all(0));
    KF3.measurementMatrix.at<float>(0,0) = 1;
    randn(KF3.statePost,Scalar::all(180),Scalar::all(1));
    setIdentity(KF3.errorCovPost,Scalar::all(1));

    KF3.transitionMatrix = (Mat_<float> (3,3) << 1,delta,(alpha*delta-1+exp(-alpha*delta))/(alpha*alpha),
            0,1,(1-exp(-alpha*delta))/alpha,
            0,0,exp(-alpha*delta));

    double delta1 = 6;
    Mat trans_matrix = (Mat_<float> (3,3) << 1,delta1,(alpha*delta1-1+exp(-alpha*delta1))/(alpha*alpha),
            0,1,(1-exp(-alpha*delta1))/alpha,
            0,0,exp(-alpha*delta1));
#ifdef _DEBUG
    cout << "KF.statePost : " << KF.statePost << endl;
#endif
    // 初始化各个类别
    read_data();
    optimize op;
    Enemy enemy;
    Camera camera(DAHUA2);
    camera.target_height = 0.128457;
    timespec tstart1,tend1,tstart2,tend2,tstart3,tend3,tstart4,tend4;

    Mat image,hsv;
    // 读取视频
    VideoCapture videoCapture(VIDEO_PATH);
    vector<Point3d > history_center; // 敌方机器人历史位置数据
    vector<Point3d > origin_center; // 敌方机器人呢
    vector<Point2d > tails; // 地方机器人尾部装甲板位置数据
    vector<Point3d > CENTER5,VARIANCE5,CENTER15,VARIANCE15; // 敌方机器人统计学参数、中心、方差、过去15帧平均中心位置数据、过去15帧方差平均数据
    vector<pair<double,double> > trust_prob; // front_prob > leap_prob + 0.15?
    vector<Point3d > predict; // 预测位置数据
    Rect last_rect; // 上一次机器人矩形框在图片中的位置，用来减少计算量
    if (!videoCapture.isOpened()){
        cout << "Video is not opened ! " << endl;
        return -1;
    }

    int count = 0;
    double s = 0.0;
    while (true){
        videoCapture >> image;
        bool f_trust = false;
        //lo++;
        //if (lo < 1000)
        //    continue;
        Mat em = Mat(800,800,CV_8UC3,Scalar::all(255));
        clock_gettime(CLOCK_REALTIME,&tstart1);
        Point2d center(em.rows/2.0,em.cols-10);
        if (image.empty()){
            cout << "Image is empty !" << endl;
            return -1;
        }

        // 将last_rect进行扩大，保证和当前图像的一致性
        refine(image,last_rect);
        Mat image_temp;
        image.copyTo(image_temp);
        // 利用上一次计算的机器人底盘边界框减小计算量
        if (!last_rect.empty())
            image = image(last_rect);
#ifdef _DEBUG
        cout << "imagesize : " << image.cols << "," << image.rows << endl;
        cout << "last : " << last_rect.x << "," << last_rect.y << "," << last_rect.width << "," << last_rect.height << endl;
#endif

        // 对图像进行二值化处理以及腐蚀膨胀
        cvtColor(image,image,CV_RGB2GRAY);
        threshold(image,hsv,210,255,THRESH_BINARY);
        Mat structureElement = getStructuringElement(MORPH_RECT, Size(2,2), Point(-1, -1));
        erode(hsv,hsv,structureElement);
        dilate(hsv,hsv,structureElement);

        //imshow("orbin",hsv);
        // 找到白色的最小区域，提取出来做后续处理，减小计算量
        int xmin = hsv.cols;
        int ymin = hsv.rows;
        int xmax = 0;
        int ymax = 0;
        for (int i = 0; i < hsv.rows; ++i) {
            for (int j = 0; j < hsv.cols; ++j) {
                if (hsv.ptr<uchar >(i)[j] > 150){
                    if (j<xmin)
                        xmin = j;
                    if (j>xmax)
                        xmax = j;
                    if (i<ymin)
                        ymin = i;
                    if (i>ymax)
                        ymax = i;
                }
            }
        }

        if (last_rect.width < 10 || last_rect.height < 10)
        {
            cout << "over" << endl;
            last_rect = Rect();
        }

        if (xmin == hsv.cols || ymin == hsv.rows || xmax == 0 || ymax == 0)
        {
            cout << "Return : No bright point detected!" << endl;
            continue;
        }

        // 将hsv的最小白色区域提取出来并在图像周围进行pad
        Mat hsv_temp(hsv.rows+40,hsv.cols+40,CV_8UC1,Scalar(0));
        hsv.copyTo(hsv_temp(Rect(20,20,hsv.cols,hsv.rows)));
        Rect rect(xmin,ymin,xmax-xmin+40,ymax-ymin+40);
        hsv = hsv_temp(rect);

        last_rect.x = xmin+last_rect.x;
        last_rect.y = ymin+last_rect.y;
        last_rect.width = xmax-xmin;
        last_rect.height = ymax-ymin;

        xmin = last_rect.x;
        ymin = last_rect.y;
        xmax = last_rect.x + last_rect.width;
        ymax = last_rect.y + last_rect.height;

        // 轮廓识别
        vector<vector<Point> > contours;
        vector<Vec4i > hierarchy;
        vector<Lightbar* > lightbars;
        findContours(hsv,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
        clock_gettime(CLOCK_REALTIME,&tend1);
        clock_gettime(CLOCK_REALTIME,&tstart2);
        if (contours.size() <= 1){
            last_rect = Rect();
            tails.clear();
            cout << "contour size less than 2" << endl;
            CENTER5.clear();
            CENTER15.clear();
            predict.clear();
            continue;
        }

        // 寻找出所有轮廓并且将面积小的轮廓加入无效轮廓
        // 正常的轮廓如果长宽比没达到要求或者过于水平，则剔除
        double valid_lightbar_y_sum = 0;
        double max_length = 0;
        unsigned long max_length_index = 0;
        for (int index = 0; index>=0 ; index=hierarchy[index][0]) {
            Scalar color(255,0,0);
            double area = cv::contourArea(contours[index]);
            if (area < 50){
                if (area > 20){
                    auto* lightbar = new Lightbar(blue,contours[index],hsv);
                    if (lightbar->ratio > 4){
                        Point vec = lightbar->boundingbox.boundingRect().br()-Point(lightbar->focus.x,lightbar->focus.y);
                        double angle = atan2(vec.y,vec.x)/M_PI*180;
                        if (abs(angle) > 45){
                            lightbars.push_back(lightbar);
                            valid_lightbar_y_sum += lightbar->focus.y;
                            if (lightbar->length > max_length){
                                max_length = lightbar->length;
                                max_length_index = lightbars.size()-1;
                            }
                        }
                        else
                            delete lightbar;
                    }
                }
            }
            else{
                auto* lightbar = new Lightbar(blue,contours[index],hsv);
                Point vec = lightbar->boundingbox.boundingRect().br()-Point(lightbar->focus.x,lightbar->focus.y);
                double angle = abs(atan(vec.y/vec.x)/M_PI*180);
                if (lightbar->ratio > 2 && abs(angle) > 45){
                    lightbars.push_back(lightbar);
                    valid_lightbar_y_sum += lightbar->focus.y;
                    if (lightbar->length > max_length){
                        max_length = lightbar->length;
                        max_length_index = lightbars.size()-1;
                    }
                }
                else
                    delete lightbar;

            }
        }

        valid_lightbar_y_sum = valid_lightbar_y_sum/lightbars.size();

        // 如果最大长度和最小长度差距过大，那么删除最小长度的lightbar
        for (int m = 0; m < lightbars.size(); ++m) {
            //rectangle(hsv,lightbars[m]->boundingbox.boundingRect(),Scalar(255),1);
            if (m != max_length_index){
                if (max_length > 1.4*lightbars[m]->length){
                    delete lightbars[m];
                    lightbars.erase(lightbars.begin()+m);
                    m--;
                }
            }
        }

        // 如果地面反光，那么将下面的lightbar删除
        for (int l = 0; l < lightbars.size(); ++l) {
            if (lightbars[l]->focus.y - valid_lightbar_y_sum > max_length*0.8){
                delete lightbars[l];
                lightbars.erase(lightbars.begin()+l);
                l--;
            }
        }

        // 判断lightbar是否有效
        if (lightbars.empty() || lightbars.size() == 1){
            cout << "Can't detect lightbar!" << endl;
            last_rect = Rect();
            tails.clear();
            CENTER5.clear();
            CENTER15.clear();
            predict.clear();
            continue;
        }

        // 设置相机外参矩阵(根据具体情况设定)
        Matrix4d Tw_c = Matrix4d::Identity();
        Tw_c(2,3) = 0.440; // 0.403 0.394 0.459
        Tw_c(2,2) = cos(105.3 / 180 * M_PI); // 105.3
        Tw_c(1,1) = cos(105.3 / 180 * M_PI);
        Tw_c(1,2) = sin(105.3/ 180 * M_PI);
        Tw_c(2,1) = -sin(105.3 / 180 * M_PI);
        //cout << "Tw_c : \n" << Tw_c.matrix() << endl;
        camera.Tw_c = Tw_c;

        vector<Point2d > UV;
        vector<Point2d > lightbars_position;
        ave_y = 0.0;

        // 计算每个灯条相对于机器人坐标系的坐标
        for (auto &l:lightbars) {
            //UV.emplace_back(Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin+image.rows));
            //Point2d focus = Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin+image.rows);
            UV.emplace_back(Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin));
            Point2d focus = Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin);
            lightbars_position.push_back(camera.back_project(focus));
            ave_y += lightbars_position.back().y;
#ifdef _DEBUG
            cout << "UV : " << l->focus.x-20+xmin << "," << l->focus.y-20+ymin << endl;
            cout << "lightbars_position : " << lightbars_position[lightbars_position.size()-1].x << "," <<
                 lightbars_position[lightbars_position.size()-1].y << endl;
#endif
        }
        ave_y = ave_y/lightbars_position.size();

        // 创建灯条斜率的容器，用来判断是否是同一个装甲板
        vector<double > lines;
        for (auto &l:lightbars){
            lines.push_back(l->k);
            //cout << "slope : " << l->k << endl;
        }

        // 将所有的图片lightbars数量都变为3，这种情况是比较稳定的，并且求出同一个armor对应的灯条index
        pair<int, int> armor;
        switch (lightbars.size()){
            case 5:{
                // 找到并删除两侧的不稳定的灯条
                double min_x = 2000;
                double max_x = 0;
                int max_x_index = 0;
                int min_x_index = 0;
                for (int i = 0; i < 5; ++i) {
                    if (lightbars[i]->focus.x < min_x){
                        min_x = lightbars[i]->focus.x;
                        min_x_index = i;
                    }
                    else if (lightbars[i]->focus.x > max_x){
                        max_x = lightbars[i]->focus.x;
                        max_x_index = i;
                    }
                }
#ifdef _DEBUG
                cout << "max : " << max_x_index << endl;
                cout << "min : " << min_x_index << endl;
#endif
                lightbars[min_x_index]->focus = Point2d(0,0);
                lightbars[max_x_index]->focus = Point2d(0,0);
                for (int i = 0; i < lightbars.size(); ++i) {
                    if (lightbars[i]->focus.x == 0 && lightbars[i]->focus.y == 0){
                        delete lightbars[i];
                        lightbars.erase(lightbars.begin()+i);
                        lines.erase(lines.begin()+i);
                        lightbars_position.erase(lightbars_position.begin()+i);
                        UV.erase(UV.begin()+i);
                        i--;
                    }
                }
                assert(lightbars.size() == 3);
                // 判断剩下灯条的匹配情况
                armor = find_pair(UV,lightbars_position,lines);
#ifdef _DEBUG
                cout << "armor : " << armor.first << ","<< armor.second << endl;
#endif
                break;
            }
            case 4:{
                double min_x = 2000;
                int min_x_index = 0;
                for (int i = 0; i < 4; ++i) {
                    if (lightbars[i]->focus.x < min_x){
                        min_x = lightbars[i]->focus.x;
                        min_x_index = i;
                    }
                }
                delete lightbars[min_x_index];
                lightbars.erase(lightbars.begin()+min_x_index);
                lightbars_position.erase(lightbars_position.begin()+min_x_index);
                UV.erase(UV.begin()+min_x_index);
                lines.erase(lines.begin()+min_x_index);
                assert(lightbars.size() == 3);
                armor = find_pair(UV,lightbars_position,lines);
                break;
            }
            case 3:{
                armor = find_pair(UV,lightbars_position,lines);
#ifdef _DEBUG
                cout << "armor : " << armor.first << ","<< armor.second << endl;
#endif
                break;
            }
            case 2:{
                if (dist(lightbars_position[0],lightbars_position[1]) - 0.135 < 0.04){
                    armor.first = 0;
                    armor.second = 1;
                }
                break;
            }
            default:
                break;
        }

        if (armor.first == armor.second){
            cout << "Can't detect an armor !" << endl;
            last_rect = Rect();
            tails.clear();
            CENTER5.clear();
            CENTER15.clear();
            predict.clear();
            continue;
        }

        // 直接计算得到敌方装甲板在机器人坐标系下的角度
        double ratio = (lightbars_position[armor.first].y-lightbars_position[armor.second].y)/
                       (lightbars_position[armor.first].x-lightbars_position[armor.second].x);
        double theta3 = abs(atan(ratio)/M_PI*180);

        // 处理在0度附近容易受误差干扰出现角度跳变的情况
        if (lines[armor.first] + lines[armor.second] < 0)
            theta3 = -theta3;

        vector<double > distance;
        distance.push_back(0.135);
        distance.push_back(0.235);

        // 按照从左到右的顺序调整灯条顺序
        if (UV[armor.first].x < UV[armor.second].x)
        {
            int temp = armor.second;
            armor.second = armor.first;
            armor.first = temp;
        }

        //vector<Point2d > ppoints = pnp(armor,lightbars,camera,Tw_c,xmin,ymin);
        //lightbars_position[armor.first] = ppoints[0];
        //lightbars_position[armor.second] = ppoints[1];

        // 得到装甲板侧边的灯条配对序号
        vector<pair<int, int>> reflect; // 容器第一项为装甲板灯条配对序号，第二项为装甲板侧边配对序号
        reflect.push_back(armor);
        for (int i1 = 0; i1 < lightbars.size(); ++i1) {
            if (i1 != armor.first && i1 != armor.second){
                if (abs(lightbars[i1]->focus.x-lightbars[armor.first]->focus.x) < abs(lightbars[i1]->focus.x-lightbars[armor.second]->focus.x))
                    reflect.emplace_back(make_pair(armor.first,i1));
                else
                    reflect.emplace_back(make_pair(armor.second,i1));
            }
        }

        if (reflect.size() == 1)
            distance.pop_back();

        vector<Point2d > lightbars_position2 = lightbars_position;

        // 可视化之前的lightbars_position
        for (auto &l:lightbars_position)
            circle(em,Point2d(center.x+l.x*150,center.y-l.y*150),2,Scalar(0,0,255),2);
        clock_gettime(CLOCK_REALTIME,&tend2);
        clock_gettime(CLOCK_REALTIME,&tstart3);
        // 如果距离过远的话计算得到的灯条位置不准确，优化灯条的位置
        if (ave_y > 3.5)
            lightbars_position = op.direct_optimize(3,UV,lightbars_position,armor,camera,distance,theta3,em,center);

        //如果优化得到的装甲板角度和图片方向不一致，那么取反
        double theta_act = atan((lightbars_position[armor.first].y - lightbars_position[armor.second].y)/
                                (lightbars_position[armor.first].x - lightbars_position[armor.second].x));
        if (theta3*theta_act < 0)
            exchange(lightbars_position[armor.first],lightbars_position[armor.second]);

        // 图片中识别到的装甲板可能是正装甲板也可能是侧面装甲板
        // 作出两种假设，判断两种假设下的概率，取概率最大者作为机器人的位姿
        Point2d center_front_assumpt; // 在前装甲板假设下的机器人中心
        Point2d center_leap_assumpt; // 在侧装甲板假设下的机器人呢中心
        Rect rect_front, rect_front2, rect_leap, rect_leap2;
        Rect front_max_rect, front_max_rect2, leap_max_rect, leap_max_rect2;
        Rect light_rect, light_max_rect;
        double front_prob, front_prob2 = 0.0, leap_prob, leap_prob2 = 0.0, light_prob = 0.0;
        center_front_assumpt = enemy.find_center_front(lightbars_position,armor,em,center);
        pair<Point2d, Point2d> front_cache, leap_cache; // 两种假设下的敌方机器人坐标系方量

        // 侧装甲板可以看到右前或左前轮子，假设面对的是正装甲板计算对应的轮子投影在当前图像下矩形框
        if (enemy.positive_x_axis.y < 0) {
            rect_front = enemy.wheel_world_coor(hitcrt::Enemy::RF, Tw_c, camera.K);
            negtive_flag = true;
        }
        else if (enemy.positive_x_axis.y >= 0) {
            rect_front = enemy.wheel_world_coor(hitcrt::Enemy::LF, Tw_c, camera.K);
            negtive_flag = false;
        }
        // 计算当前装甲板是正装甲板的概率
        front_prob = prob(image_temp,rect_front,negtive_flag,front_max_rect,1);
        front_cache.first = enemy.positive_y_axis;
        front_cache.second = enemy.positive_x_axis;

        // 侧装甲板可以看到右前或右后轮子，假设面对的是侧装甲板计算对应的轮子投影在当前图像下矩形框
        center_leap_assumpt = enemy.find_center_leap(lightbars_position,armor,em,center);
        if (enemy.positive_y_axis.y < 0) {
            rect_leap = enemy.wheel_world_coor(hitcrt::Enemy::RF, Tw_c, camera.K);
            negtive_flag = true;
        }
        else if (enemy.positive_y_axis.y >= 0) {
            rect_leap = enemy.wheel_world_coor(hitcrt::Enemy::RB, Tw_c, camera.K);
            negtive_flag = false;
        }
        // 计算当前装甲板是侧装甲板的概率
        leap_prob = prob(image_temp,rect_leap,negtive_flag,leap_max_rect,1);
        leap_cache.first = enemy.positive_y_axis;
        leap_cache.second = enemy.positive_x_axis;

        // 如果没有能够检测出来，那么历史记录清零
        if (leap_prob == 0 || front_prob == 0)
            history_center.clear();

        Point3d opcenter;
        int center_assupmt = 0;
        // 以大概率的假设作为基准，通过血量灯条，计算机器人正方向的朝向概率
        if (leap_prob > front_prob) {
            enemy.positive_x_axis = leap_cache.second;
            enemy.positive_y_axis = leap_cache.first;
            if (!history_center.empty()){
                opcenter = op.center_optimize(3, UV, lightbars_position, armor,
                                              Point3d(center_leap_assumpt.x, center_leap_assumpt.y, theta3), 1,
                                              camera, history_center, em, center,origin_center);
            }
            if (leap_prob - front_prob < 0.1){
                if (!trust_prob.empty()){
                    front_prob = trust_prob.back().first;
                    leap_prob = trust_prob.back().second;
                    if (front_prob > leap_prob)
                        enemy.exchange();
                }
            }
            else if (leap_prob - front_prob > 0.23) {
                trust_prob.emplace_back(make_pair(front_prob, leap_prob));
                f_trust = true;
            }

            enemy.robo_center = center_leap_assumpt;
            light_rect = enemy.direct_def(Tw_c,camera.K);
            if (enemy.positive_y_axis.x > 0)
                light_prob = prob(image_temp, light_rect, true, light_max_rect, 2);
            else if (enemy.positive_y_axis.x <= 0)
                light_prob = prob(image_temp, light_rect, false, light_max_rect, 2);
        } else {
            center_assupmt = 1;
            enemy.positive_x_axis = front_cache.second;
            enemy.positive_y_axis = front_cache.first;
            if (!history_center.empty()){
                opcenter = op.center_optimize(3, UV, lightbars_position, armor,
                                              Point3d(center_front_assumpt.x, center_front_assumpt.y, theta3), 2,
                                              camera, history_center, em, center,origin_center);
            }
            if (front_prob - leap_prob < 0.1) {
                if (!trust_prob.empty()){
                    front_prob = trust_prob.back().first;
                    leap_prob = trust_prob.back().second;
                    if (leap_prob > front_prob)
                        enemy.exchange();
                }
            }
            else if (front_prob - leap_prob > 0.23) {
                trust_prob.emplace_back(make_pair(front_prob, leap_prob));
                f_trust = true;
            }

            enemy.robo_center = center_front_assumpt;
            light_rect = enemy.direct_def(Tw_c,camera.K);
            if (enemy.positive_y_axis.x > 0)
                light_prob = prob(image_temp,light_rect,true,light_max_rect,2);
            else if (enemy.positive_y_axis.x <= 0)
                light_prob = prob(image_temp,light_rect,false,light_max_rect,2);
        }
        clock_gettime(CLOCK_REALTIME,&tend3);
        clock_gettime(CLOCK_REALTIME,&tstart4);
#ifdef _DEBUG
        cout << "light_prob : " << light_prob << endl;
#endif
        // 如果血量灯条边界框不能满足假设，那么直接归0，且概率值以0.3为上界
        if (prompt(image_temp,hsv,xmin,ymin,light_max_rect) == 1)
            light_prob = 0.0;

        else if (prompt(image_temp,hsv,xmin,ymin,light_max_rect) == 2)
            light_prob = 0.3;
#ifdef _DEBUG
        cout << "light_prob : " << light_prob << endl;
#endif
        // 如果概率过小，则血量灯条边界框在当前假设方向的对向
        if (light_prob < 0.1)
            enemy.inv();

        // 计算尾部装甲板的位置并返回计算成功与否
        bool iswrong = enemy.show_tail(em,center,tails);
        if (iswrong)
        {
            //tails.clear();
            if (f_trust)
                trust_prob.pop_back();
            predict.clear();
            continue;
        }
        else
            tails.push_back(enemy.tail);
#ifdef _DEBUG
        cout << "tailssss : " << tails.back().x << "," << tails.back().y << endl;
#endif
        /**********画图************/
        cvtColor(hsv,hsv,CV_GRAY2RGB);
        // 画出每个lightbar的拟合直线
        for (auto &l:lightbars)
            line(hsv,l->focus+Point2d(50,l->k*50),l->focus-Point2d(50,l->k*50),Scalar(255,0,0),1);

        // 框选出lightbar并显示中心
        for (int n = 0; n < lightbars.size(); ++n) {
            if (n == armor.first || n == armor.second){
                circle(hsv,lightbars[n]->focus,2,Scalar(0,0,255),2);
                rectangle(hsv,lightbars[n]->boundingbox.boundingRect(),Scalar(0,255,0),1);
            }
        }

        // 画出原始图片
        for (auto &l:lightbars) {
            //Point2d focus = Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin+image.rows);
            Point2d focus = Point2d(l->focus.x-20+xmin,l->focus.y-20+ymin);
            circle(image_temp,focus,2,Scalar(0,0,255),2);
        }
        rectangle(image_temp,rect_front,Scalar(0,255,0),2);
        rectangle(image_temp,rect_leap,Scalar(255,0,0),2);
        rectangle(image_temp,light_rect,Scalar(0,165,255),2);
        rectangle(image_temp,front_max_rect,Scalar(0,255,0),1);
        rectangle(image_temp,leap_max_rect,Scalar(255,0,0),1);
        rectangle(image_temp,light_max_rect,Scalar(0,165,255),1);
        putText(image_temp, "front_prob : "+to_string(front_prob), rect_front.tl()+Point(-60,-15), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,255,0),1);
        putText(image_temp, "leap_prob : "+to_string(leap_prob), rect_leap.br()+Point(10,-10), FONT_HERSHEY_COMPLEX, 0.5,Scalar(255,0,0),1);
        putText(image_temp, "light_prob : "+to_string(light_prob), light_rect.tl()+Point(-60,-15), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,165,255),1);

        /**********画图结束************/

        theta3 = robotheta(enemy);
        Mat x_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
        Mat y_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
        Mat theta_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
        Mat measure = Mat(1,1,CV_32F);

        vector<Point2d > assumpt_vec;
        assumpt_vec.push_back(center_leap_assumpt);
        assumpt_vec.push_back(center_front_assumpt);

        // 将计算结果收纳到历史信息并且保存计算结果
        if (front_prob > leap_prob) {
            origin_center.emplace_back(Point3d(center_front_assumpt.x, center_front_assumpt.y, theta3));
            rectangle(image_temp, Rect(1100, 100, 50, 50), Scalar(0, 255, 0), -1);
            //circle(image_temp,Point2d(front_max_rect.x+double(front_max_rect.width)/2,front_max_rect.y+double(front_max_rect.height)/2),3,Scalar(0,0,255));
            circle(image_temp,Point2d(rect_front.x+double(rect_front.width)/2,rect_front.y+double(rect_front.height)/2),3,Scalar(0,0,255));
            circle(em,Point2d(center.x+150*center_front_assumpt.x,center.y-150*center_front_assumpt.y),3,Scalar(0,100,0),2);
            odometry << center_front_assumpt.x << " " << center_front_assumpt.y << " " << theta3 << " " <<
                    enemy.positive_y_axis.x << " " << enemy.positive_y_axis.y << endl;
            //if (converge(history_center,Point3d(center_front_assumpt.x,center_front_assumpt.x,theta3)))
            //history_center.emplace_back(Point3d(center_front_assumpt.x, center_front_assumpt.y, theta3));
            if (history_center.empty()) {
                history_center.emplace_back(Point3d(center_front_assumpt.x, center_front_assumpt.y, theta3));
                //odometryop << opcenter.x << " " << opcenter.y << endl;
            }
            else {
                //history_center.emplace_back(Point3d(opcenter.x, opcenter.y, theta3));
                //odometryop << opcenter.x << " " << opcenter.y << " " << opcenter.z << endl;
            }
        }
        else {
            origin_center.emplace_back(Point3d(center_leap_assumpt.x, center_leap_assumpt.y, theta3));
            rectangle(image_temp, Rect(1100, 100, 50, 50), Scalar(255, 0, 0), -1);
            circle(em,Point2d(center.x+150*center_leap_assumpt.x,center.y-150*center_leap_assumpt.y),3,Scalar(0,100,0),2);
            circle(image_temp,Point2d(rect_leap.x+double(rect_leap.width)/2,rect_leap.y+double(rect_leap.height)/2),3,Scalar(0,0,255));
            odometry << center_leap_assumpt.x << " " << center_leap_assumpt.y << " "  <<  theta3 << " " <<
                    enemy.positive_y_axis.x << " " << enemy.positive_y_axis.y << endl;
            //if (converge(history_center,Point3d(center_leap_assumpt.x,center_leap_assumpt.x,theta3)))
            //history_center.emplace_back(Point3d(center_leap_assumpt.x, center_leap_assumpt.y, theta3));
            if (history_center.empty()) {
                history_center.emplace_back(Point3d(center_leap_assumpt.x, center_leap_assumpt.y, theta3));
                //odometryop << opcenter.x << " " << opcenter.y << " " << opcenter.z << endl;
            }
            else {
                //history_center.emplace_back(Point3d(opcenter.x, opcenter.y, theta3));
                //odometryop << opcenter.x << " " << opcenter.y << " " << opcenter.z << endl;
            }
        }
        Point3d center15(0,0,0), variance15(0,0,0);
        Point3d center5(0,0,0), variance5(0,0,0);

        // 统计机器人位置均值以及方差等参数，做数据处理用
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
            if (CENTER5.size() > 50)
                if (abs(center5.x-CENTER5.back().x) > 0.06 || abs(center5.y-CENTER5.back().y)>0.06)
                {
                    CENTER5.clear();
                    CENTER15.clear();
                    tails.clear();
                    predict.clear();
                    continue;
                }
        }

        /*********预测机器人接下来的位置************/
        measure.at<float>(0) = (float)center5.x;
#ifdef _DEBUG
        cout << "measure : " << measure << endl;
#endif
        x_pred = KF.predict(measure);
        measure.at<float>(0) = (float)center5.y;
        y_pred = KF2.predict(measure);
        measure.at<float>(0) = (float)center5.z;/*
        if (center5.z < 10 && KF3.statePost.at<float>(0,0) > 350)
            KF3.statePost.at<float>(0,0) -= 360;
        else if (center5.z > 350 && KF3.statePost.at<float>(0,0) < 10)
            KF3.statePost.at<float>(0,0) += 360;*/
        if (center5.z < 10 && KF3.statePost.at<float>(0,0) > 350) {
            measure.at<float>(0) = static_cast<float>(theta3);
            measure.at<float>(0) += 360;
        }
        else if (center5.z > 350 && KF3.statePost.at<float>(0,0) < 10) {
            measure.at<float>(0) = static_cast<float>(theta3);
            measure.at<float>(0) -= 360;
        }
        theta_pred = KF3.predict(measure);
        CENTER5.push_back(center5);
        CENTER15.push_back(center15);
        VARIANCE5.push_back(variance5);
        VARIANCE15.push_back(variance15);
        pdfcenter << center5.x << " " << center5.y << " " << center5.z << " "
                  << variance5.x << " " << variance5.y << " " << variance5.z << " "
                  << center15.x << " " << center15.y << " " << center15.z << " "
                  << variance15.x << " " << variance15.y << " " << variance15.z << endl;

        // 可视化优化后的灯条位置
        for (auto &l:lightbars_position)
            circle(em,Point2d(center.x+l.x*150,center.y-l.y*150),3,Scalar(0,0,0),2);
#ifdef _DEBUG
        cout << "x_pred : " << x_pred << endl;
        cout << "y_pred : " << y_pred << endl;
#endif
        Mat x_1 = trans_matrix*x_pred;
        Mat y_1 = trans_matrix*y_pred;
        Mat t_1 = trans_matrix*theta_pred;
        if (theta_pred.at<float>(0) < 0) {
            theta_pred.at<float>(0) += 360;
            KF3.statePost.at<float>(0,0) += 360;
        }
        else if (theta_pred.at<float>(0) > 360) {
            theta_pred.at<float>(0) -= 360;
            KF3.statePost.at<float>(0,0) -= 360;
        }

        if (t_1.at<float>(0) < 0)
            t_1.at<float>(0) += 360;
        else if (t_1.at<float>(0) > 360)
            t_1.at<float>(0) -= 360;

        predict.emplace_back(Point3d(x_1.at<float>(0,0),y_1.at<float>(0,0),t_1.at<float>(0,0)));
        if (predict.size() > 50){
            if (abs(predict.back().x - predict[predict.size()-2].x) > 0.06 ||
                    abs(predict.back().y - predict[predict.size()-2].y) > 0.06)
            {
                predict.clear();
                continue;
            }
        }
        /*********预测结束************/
        /*********可视化预测结果并显示图片************/
        circle(em,Point2d(center.x+x_pred.at<float>(0,0)*150,center.y-y_pred.at<float>(0,0)*150),5,Scalar(0,69,255),2);
        circle(em,Point2d(center.x+x_1.at<float>(0,0)*150,center.y-y_1.at<float>(0,0)*150),5,Scalar(255,111,131),2);
        odometryop << x_1.at<float>(0,0) << " " << y_1.at<float>(0,0) << " " << t_1.at<float>(0,0) << endl;
        // 显示distance的text
        double d = dist(lightbars_position[armor.first],lightbars_position[armor.second]);
        double d2 = dist(lightbars_position2[armor.first],lightbars_position2[armor.second]);
        putText(em, "theta3 : "+to_string(theta3), Point2d(center.x,center.y-165), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);
        putText(em, to_string(d), Point2d(center.x,center.y-150), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);
        putText(em, to_string(d2), Point2d(center.x,center.y-140), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);

        Point2d target = find_target(x_1.at<float>(0,0),y_1.at<float>(0,0),t_1.at<float>(0,0));
        circle(em,Point2d(center.x+target.x*150,center.y-target.y*150),3,Scalar(238,0,238),2);
        imshow("lightbars_position : ",em);
        imshow("binary",hsv);
        imshow("origin",image_temp);
        imshow("image",image);
        /*********可视化结束************/
        waitKey(1);
        clock_gettime(CLOCK_REALTIME,&tend4);
#ifdef _DEBUG
        cout<< "consume time1 : " << double(tend1.tv_nsec-tstart1.tv_nsec)/1000000 << "ms" <<endl;
        cout<< "consume time2 : " << double(tend2.tv_nsec-tstart2.tv_nsec)/1000000 << "ms" <<endl;
        cout<< "consume time3 : " << double(tend3.tv_nsec-tstart3.tv_nsec)/1000000 << "ms" <<endl;
        cout<< "consume time4 : " << double(tend4.tv_nsec-tstart4.tv_nsec)/1000000 << "ms" <<endl;
        cout<< "consume time : " << double(tend4.tv_nsec-tstart1.tv_nsec)/1000000 << "ms" <<endl;
#endif
        /*while (true){
            int k = waitKey(0);
            if (k == 27)
                break;
        }*/
    }
    odometry.close();
    odometryop.close();
    pdfcenter.close();
    return 0;
}

// 辅助计算可视化显示位置的函数
Point2d find_target(float x_1, float y_1, float t_1)
{
    double r1 = 0.268348, r2 = 0.187848;
    vector<Point2d > f_b_l1_l2;
    f_b_l1_l2.emplace_back(Point2d(x_1+r1*cos(t_1/180*M_PI),y_1+r1*sin(t_1/180*M_PI)));
    f_b_l1_l2.emplace_back(Point2d(x_1-r1*cos(t_1/180*M_PI),y_1-r1*sin(t_1/180*M_PI)));
    f_b_l1_l2.emplace_back(Point2d(x_1+r2*sin(t_1/180*M_PI),y_1-r2*cos(t_1/180*M_PI)));
    f_b_l1_l2.emplace_back(Point2d(x_1-r2*sin(t_1/180*M_PI),y_1+r2*cos(t_1/180*M_PI)));
    double fb_l = 0.135*abs(sin(t_1/180*M_PI));
    double l_l = 0.135*abs(cos(t_1/180*M_PI));
    vector<double > score;
    score.push_back(fb_l*2);
    score.push_back(fb_l*6);
    score.push_back(l_l*4);
    score.push_back(l_l*4);

    for (int j = 0; j < 4; ++j) {
        if (f_b_l1_l2[j].y >= y_1)
            score[j] = 0.0;
    }

    int index = 0;
    double max_score = 0.0;
    for (int i = 0; i < score.size(); ++i) {
        if (score[i] > max_score)
        {
            index = i;
            max_score = score[i];
        }
    }

    return f_b_l1_l2[index];
}

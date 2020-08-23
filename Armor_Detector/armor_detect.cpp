//
// Created by mrm on 19-12-23.
//

#include "armor_detect.h"

namespace hitcrt {

bool Armor_detector::rectinimage(const Rect &rect, const Mat &image_temp)
{
    if (rect.x < 0)
        return false;
    else if (rect.x > image_temp.cols)
        return false;

    if (rect.x + rect.width > image_temp.cols)
        return false;

    if (rect.y < 0)
        return false;
    else if (rect.y > image_temp.rows)
        return false;

    if (rect.y + rect.height > image_temp.rows)
        return false;

    return true;
}

void Armor_detector::refine(const Mat &image, Rect &rect)
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

int Armor_detector::prompt(const Mat &image_temp, const Mat &hsv,int xmin,int ymin, const Rect &light_max_rect)
{
    double white_count = 0.0;
    Rect hsv_rect;
    if (light_max_rect.height > 30){
        hsv_rect = Rect(light_max_rect.x-xmin+20,light_max_rect.y-ymin+20,light_max_rect.width,light_max_rect.height);
        if (hsv_rect.x > hsv.cols)
            return 1;
        if (hsv_rect.y > hsv.rows)
            return 1;
        if (hsv_rect.x + hsv_rect.width < 0)
            return 1;
        if (hsv_rect.y + hsv_rect.height < 0)
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
    cout << "white_count : " << white_count << endl;
    if (white_count < 5.0) {
        return 1;
    }
    else if (white_count/area > 0.3)
        return 2;

    return 3;
}

double Armor_detector::prob(const Mat &image_temp, const Rect &rect, bool negtive_flag, Rect &max_prob_rect, int state)
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

        double s = 60.0/rect.height;
        if (s > 1)
            s = 1.0;
        image_temp(rect).copyTo(target);
        resize(target,target,Size(int(s*rect.width),int(s*rect.height)),0,0,INTER_LINEAR);
        double ratio = double(rect.height)/rect.width;
        vector<Mat > candicate;
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

        cout << "candicate.size : " << candicate.size() << endl;
        if (candicate.empty())
            return 0.0;

        double max_prob = -1;
        Rect rect2;
        double width_ratio = 1, height_ratio = 1;
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
    else if (state == 2){
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

        cout << "candicate.size : " << candicate.size() << endl;
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

double Armor_detector::robotheta(const Enemy &enemy)
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

double Armor_detector::dist(const Point2d &p1, const Point2d &p2)
{
    return sqrt(pow(p1.x-p2.x,2)+pow(p1.y-p2.y,2));
}

double Armor_detector::trans(double t)
{
    if (t < 0)
        t = -t;
    else
        t = 180 - t;
    return t;
}

void Armor_detector::exchange(Point2d &p1, Point2d &p2)
{
    double y_temp;
    y_temp = p1.y;
    p1.y = p2.y;
    p2.y = y_temp;
}

pair<int, int> Armor_detector::find_pair(const vector<Point2d> &UV, const vector<Point2d > &lightbars_position, const vector<double > &lines)
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

int Armor_detector::run(Mat &image, Mat &EM, Mat &Image_temp, Matrix4d Tw_c)
{
    bool f_trust = false;
    Mat em = Mat(800,800,CV_8UC3,Scalar::all(255));
    clock_gettime(CLOCK_REALTIME,&tstart1);
    Point2d center(em.rows/2.0,em.cols-10);
    if (image.empty()){
        cout << "Return : Image is empty !" << endl;
        return 1;
    }

    refine(image,last_rect);
    Mat image_temp;
    image.copyTo(image_temp);
    if (!last_rect.empty())
        image = image(last_rect);
    cvtColor(image,image,CV_RGB2GRAY);
    threshold(image,hsv,210,255,THRESH_BINARY);
    Mat structureElement = getStructuringElement(MORPH_RECT, Size(2,2), Point(-1, -1));
    erode(hsv,hsv,structureElement);
    dilate(hsv,hsv,structureElement);

    // 找到白色的最小区域
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

    // 将hsv的最小白色区域提取出来，这样做是为了防止超出范围
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

    vector<vector<Point> > contours;
    vector<Vec4i > hierarchy;
    vector<Lightbar* > lightbars;
    findContours(hsv,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    clock_gettime(CLOCK_REALTIME,&tend1);
    clock_gettime(CLOCK_REALTIME,&tstart2);
    if (contours.size() <= 1){
        last_rect = Rect();
        tails.clear();
        cout << "Return : contour size less than 2" << endl;
        CENTER5.clear();
        CENTER15.clear();
        predict.clear();
        return 2;
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

    // 如果最大长度和最小长度差距过大，那么删除最小长度
    for (int m = 0; m < lightbars.size(); ++m) {
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
        cout << "Return : Can't detect lightbar!" << endl;
        last_rect = Rect();
        tails.clear();
        CENTER5.clear();
        CENTER15.clear();
        predict.clear();
        return 3;
    }

	// 设置相机外参矩阵
    camera.Tw_c = Tw_c;

    vector<Point2d > UV;
    vector<Point2d > lightbars_position;
    ave_y = 0.0;
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

    // 创建斜率的容器，用来判断是否是同一个装甲板
    vector<double > lines;
    for (auto &l:lightbars){
        lines.push_back(l->k);
        //cout << "slope : " << l->k << endl;
    }

    // 将所有的图片lightbars数量都变为3，并且求出同一个armor对应的index
    pair<int, int> armor;
    switch (lightbars.size()){
        case 5:{
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
        cout << "Return : Can't detect an armor !" << endl;
        last_rect = Rect();
        tails.clear();
        CENTER5.clear();
        CENTER15.clear();
        predict.clear();
        return 4;
    }

    // 直接计算得到theta
    double ratio = (lightbars_position[armor.first].y-lightbars_position[armor.second].y)/
                   (lightbars_position[armor.first].x-lightbars_position[armor.second].x);
    double theta3 = abs(atan(ratio)/M_PI*180);
    if (lines[armor.first] + lines[armor.second] < 0)
        theta3 = -theta3;

    vector<double > distance;
    distance.push_back(0.135);
    distance.push_back(0.235);

    if (UV[armor.first].x < UV[armor.second].x)
    {
        int temp = armor.second;
        armor.second = armor.first;
        armor.first = temp;
    }

    vector<pair<int, int>> reflect;
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
    //优化
    if (ave_y > 3.5)
        lightbars_position = op.direct_optimize(3,UV,lightbars_position,armor,camera,distance,theta3,em,center);

    //如果优化得到的armor角度和图片方向不一致，那么取反
    double theta_act = atan((lightbars_position[armor.first].y - lightbars_position[armor.second].y)/
                            (lightbars_position[armor.first].x - lightbars_position[armor.second].x));
    if (theta3*theta_act < 0)
        exchange(lightbars_position[armor.first],lightbars_position[armor.second]);

    Point2d center_front_assumpt;
    Point2d center_leap_assumpt;
    Rect rect_front, rect_front2, rect_leap, rect_leap2;
    Rect front_max_rect, front_max_rect2, leap_max_rect, leap_max_rect2;
    Rect light_rect, light_max_rect;
    double front_prob, front_prob2 = 0.0, leap_prob, leap_prob2 = 0.0, light_prob = 0.0;
    center_front_assumpt = enemy.find_center_front(lightbars_position,armor,em,center);
    pair<Point2d, Point2d> front_cache, leap_cache;

    if (enemy.positive_x_axis.y < 0) {
        rect_front = enemy.wheel_world_coor(hitcrt::Enemy::RF, Tw_c, camera.K);
        negtive_flag = true;
    }
    else if (enemy.positive_x_axis.y >= 0) {
        rect_front = enemy.wheel_world_coor(hitcrt::Enemy::LF, Tw_c, camera.K);
        negtive_flag = false;
    }
    front_prob = prob(image_temp,rect_front,negtive_flag,front_max_rect,1);
    front_cache.first = enemy.positive_y_axis;
    front_cache.second = enemy.positive_x_axis;

    // 计算leap假设的概率
    center_leap_assumpt = enemy.find_center_leap(lightbars_position,armor,em,center);
    if (enemy.positive_y_axis.y < 0) {
        rect_leap = enemy.wheel_world_coor(hitcrt::Enemy::RF, Tw_c, camera.K);
        negtive_flag = true;
    }
    else if (enemy.positive_y_axis.y >= 0) {
        rect_leap = enemy.wheel_world_coor(hitcrt::Enemy::RB, Tw_c, camera.K);
        negtive_flag = false;
    }
    leap_prob = prob(image_temp,rect_leap,negtive_flag,leap_max_rect,1);
    leap_cache.first = enemy.positive_y_axis;
    leap_cache.second = enemy.positive_x_axis;

    // 如果没有能够检测出来，那么historycenter清零
    if (leap_prob == 0 || front_prob == 0)
        history_center.clear();

    Point3d opcenter;
    int center_assupmt = 0;
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
    // 如果不能满足假设，那么直接归0
    if (prompt(image_temp,hsv,xmin,ymin,light_max_rect) == 1)
        light_prob = 0.0;

    else if (prompt(image_temp,hsv,xmin,ymin,light_max_rect) == 2)
        light_prob = 0.3;
#ifdef _DEBUG
    cout << "light_prob : " << light_prob << endl;
#endif

    if (light_prob < 0.1)
        enemy.inv();

    bool iswrong = enemy.show_tail(em,center,tails);
    if (iswrong)
    {
        //tails.clear();
        if (f_trust)
            trust_prob.pop_back();
        predict.clear();
        return 5;
    }
    else
        tails.push_back(enemy.tail);

#ifdef _DEBUG
    cout << "tailssss : " << tails.back().x << "," << tails.back().y << endl;
#endif

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

    theta3 = robotheta(enemy);
    Mat x_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
    Mat y_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
    Mat theta_pred = Mat(stateNum,1,CV_32F,Scalar::all(0));
    Mat measure = Mat(1,1,CV_32F);

    vector<Point2d > assumpt_vec;
    assumpt_vec.push_back(center_leap_assumpt);
    assumpt_vec.push_back(center_front_assumpt);

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

    // 自适应
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
                return 6;
            }
    }

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

    // 可视化优化后的lightbars_position
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
            return 7;
        }
    }

    circle(em,Point2d(center.x+x_pred.at<float>(0,0)*150,center.y-y_pred.at<float>(0,0)*150),5,Scalar(0,69,255),2);
    circle(em,Point2d(center.x+x_1.at<float>(0,0)*150,center.y-y_1.at<float>(0,0)*150),5,Scalar(255,111,131),2);
    odometryop << x_1.at<float>(0,0) << " " << y_1.at<float>(0,0) << " " << t_1.at<float>(0,0) << endl;
    // 显示distance的text
    double d = dist(lightbars_position[armor.first],lightbars_position[armor.second]);
    double d2 = dist(lightbars_position2[armor.first],lightbars_position2[armor.second]);
    putText(em, "theta3 : "+to_string(theta3), Point2d(center.x,center.y-165), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);
    putText(em, to_string(d), Point2d(center.x,center.y-150), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);
    putText(em, to_string(d2), Point2d(center.x,center.y-140), FONT_HERSHEY_COMPLEX, 0.5,Scalar(0,0,0),1);
    EM = em;
    Image_temp = image_temp;
    //imshow("lightbars_position : ",em);
    //imshow("binary",hsv);
    //imshow("origin",image_temp);
    //waitKey(1);
    clock_gettime(CLOCK_REALTIME,&tend4);

#ifdef _DEBUG
    cout<< "consume time1 : " << double(tend1.tv_nsec-tstart1.tv_nsec)/1000000 << "ms" <<endl;
    cout<< "consume time2 : " << double(tend2.tv_nsec-tstart2.tv_nsec)/1000000 << "ms" <<endl;
    cout<< "consume time3 : " << double(tend3.tv_nsec-tstart3.tv_nsec)/1000000 << "ms" <<endl;
    cout<< "consume time4 : " << double(tend4.tv_nsec-tstart4.tv_nsec)/1000000 << "ms" <<endl;
    cout<< "consume time : " << double(tend4.tv_nsec-tstart1.tv_nsec)/1000000 << "ms" <<endl;
#endif
    return 0;
}
}

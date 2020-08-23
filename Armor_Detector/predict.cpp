//
// Created by xiaoyang on 18-3-29.
//

#include "predict.h"
#include "opencv/cv.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/core/mat.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <Eigen/Core>

using namespace std;
using namespace cv;
using namespace Eigen;

const Mat& SKF::predict(const Mat& measurement,const Mat& control) {//
//        if(abs(measurement.at<float>(0)-statePost.at<float>(0))<= 15 )
//            return measurement;
    //cout << "statepost : " << statePost << endl;
    statePre = transitionMatrix*statePost;
    //cout << "statePre : " << statePre << endl;
    temp1 = measurement - measurementMatrix*statePre;
    //cout << "temp1 : " << temp1 << endl;
    if(k == 0)
    { S_0 = temp1*temp1.t();
        S_0pre = S_0;
        k++;
    }
    else
    {S_0 = (row*S_0pre+temp1*temp1.t())/(1+row);
        S_0pre = S_0;
        k++;
    }
    //cout << "S0 : " << S_0 << endl;
    temp2 = S_0-measurementMatrix*processNoiseCov*measurementMatrix.t()-beta*measurementNoiseCov;
    //cout << "temp2 : " << temp2 << endl;
    temp3 = measurementMatrix*transitionMatrix*errorCovPost*transitionMatrix.t()*measurementMatrix.t();
    //cout << "temp3 : " << temp3 << endl;
    traceN = trace(temp2);
    traceM = trace(temp3);
    lamda_0 = (float)traceN(0)/(float)traceM(0);
    if(lamda_0 >= 1 )
        lamda = lamda_0;
    else
        lamda = 1;
    errorCovPre = lamda*transitionMatrix*errorCovPost*transitionMatrix.t()+processNoiseCov;
    //cout << "errorCovPre : " << errorCovPre << endl;
    temp4 = measurementMatrix*errorCovPre*measurementMatrix.t()+measurementNoiseCov;
    //cout << "temp4 : " << temp4 << endl;
    gain = errorCovPre*measurementMatrix.t()*temp4.inv();
    //cout << "gain : " << gain << endl;
    statePost = statePre+gain*temp1;
//        statePre = transitionMatrix*statePost;
//        statePre = transitionMatrix*statePre;
//    statePre = transitionMatrix*statePre;
//    statePre = transitionMatrix*statePre;
    return statePost;
}
void SKF::recordArmorLocation(float X, float distance) {
    if(history_X.size()<=5)
    {
        history_X.push_back(X);
        history_distance.push_back(distance);
    }

    else
    {
        history_X.push_back(X);
        history_X.pop_front();
        history_distance.push_back(distance);
        history_distance.pop_front();
    }

}

void SKF::clear() {
    history_X.clear();
    history_distance.clear();
}

//double SKF::check(double value,double measurement) {
//    if (k==0)
//    {
//        SKF::recordArmorLocation(value);
//        } else{
//    double latest_value = history_X.back();
//    if (abs(latest_value-value)<=0.2)
//    {   cout<<"----------------------------------it's move slowly"<<endl;
//        return latest_value;}
//    else
//    {   SKF::recordArmorLocation(value);
//        return value;}
//}}

double SKF::searchTarget() {
    list<double>::const_iterator X_it = history_X.begin();
    list<double>::const_iterator distance_it = history_distance.begin();
    double X_num = 0,distance_num = 0;
    double direction = 0;
    for(;X_it != history_X.end(); X_it ++)
    {
        double X_value = *X_it;
        if(X_value<600)
            X_num++;
        else if(X_value>700)
            X_num--;
        if(X_num>0)
            direction = 2;
        else if(X_num<0)
            direction = -2;
        else
        {
            X_value = history_X.back();
            if(X_value<600)
                direction = -2;
            else if(X_value>700)
                direction = 2;
            else
                direction = 0;
        }
    }
    return direction;
}
SKF::~SKF() = default;


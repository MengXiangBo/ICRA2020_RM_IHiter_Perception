//
// Created by xiaoyang on 18-3-29.
//

#ifndef ARMOR_DETECTOR_PREDICT_H
#define ARMOR_DETECTOR_PREDICT_H

#include "stdlib.h"
#include <iostream>

#include <opencv2/core/mat.hpp>
#include <Eigen/Core>

#include "list"

using namespace std;
using namespace Eigen;
class SKF {

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    SKF( int _stateNum,int _measurementNum,double _beta,double _row)
    {
        cout << "successfully load!"<<endl;
        stateNum = _stateNum;
        measurementNum = _measurementNum;
        beta = _beta;
        row = _row;/*
        transitionMatrix.resize(stateNum,stateNum);
        measurementMatrix.resize(measurementNum,stateNum);
        processNoiseCov.resize(stateNum,stateNum);
        measurementNoiseCov.resize(measurementNum,measurementNum);
        errorCovPre.resize(stateNum,stateNum);
        statePre.resize(stateNum,1);
        statePost.resize(stateNum,1);
        gain.resize(stateNum,measurementNum);
        errorCovPost.resize(stateNum,stateNum);*/
    }
    ~SKF();
    int stateNum = 3;
    int measurementNum = 1;
    double row = 0.95;
    double beta = 30;
    int k=0;
    float lamda;/*
    Matrix<double, Dynamic, Dynamic> transitionMatrix;
    Matrix<double, Dynamic, Dynamic> measurementMatrix;
    Matrix<double, Dynamic, Dynamic> processNoiseCov;
    Matrix<double, Dynamic, Dynamic> measurementNoiseCov;
    Matrix<double, Dynamic, Dynamic> errorCovPre;
    Matrix<double, Dynamic, 1> statePre;
    Matrix<double, Dynamic, 1> statePost;
    Matrix<double, Dynamic, Dynamic> gain;
    Matrix<double, Dynamic, Dynamic> errorCovPost;*/

    cv::Mat transitionMatrix = cv::Mat (stateNum,stateNum,CV_32F);
    cv::Mat measurementMatrix = cv::Mat (measurementNum,stateNum,CV_32F);
    cv::Mat processNoiseCov = cv::Mat (stateNum,stateNum,CV_32F);
    cv::Mat measurementNoiseCov = cv::Mat (measurementNum,measurementNum,CV_32F);
    cv::Mat errorCovPre = cv::Mat (stateNum,stateNum,CV_32F);
    cv::Mat statePre = cv::Mat (stateNum,1,CV_32F);
    cv::Mat statePost = cv::Mat (stateNum,1,CV_32F);
    cv::Mat gain = cv::Mat (stateNum,measurementNum,CV_32F);
    cv::Mat errorCovPost = cv::Mat (stateNum,stateNum,CV_32F);

    void recordArmorLocation(float X, float distance);
    void clear();
    //double check(double value,double measurement = 0);
    const cv::Mat& predict(const cv::Mat& measurement,const cv::Mat& control = cv::Mat());    double searchTarget();
//    const cv::Mat& correct(const cv::Mat& measurement);


private:
    cv::Scalar traceN;
    cv::Scalar traceM;
    float lamda_0;
//    cv::Mat gama = cv::Mat (measurementNum,measurementNum,CV_64F);
    cv::Mat S_0 = cv::Mat(measurementNum,measurementNum,CV_32F);
    cv::Mat S_0pre = cv::Mat(measurementNum,measurementNum,CV_32F);
//    cv::Mat Zpre;
    cv::Mat temp1 = cv::Mat(measurementNum,1,CV_32F);    //ksi
    cv::Mat temp2 = cv::Mat(measurementNum,measurementNum,CV_32F);    //N
    cv::Mat temp3 = cv::Mat(measurementNum,measurementNum,CV_32F);    //M
    cv::Mat temp4 = cv::Mat(measurementNum,measurementNum,CV_32F);
    std::list<double> history_X;
    std::list<double> history_distance;
//    cv::Mat temp5;
//    cv::Mat temp6;


};

#endif //ARMOR_DETECTOR_PREDICT_H

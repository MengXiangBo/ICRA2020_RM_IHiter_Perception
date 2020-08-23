//
// Created by mrm on 19-11-13.
//

#include "lightbar.h"

using namespace std;
using namespace cv;

namespace hitcrt{

    void Lightbar::calc_parameter(const Mat &image) {
        Size2f size2f = boundingbox.size;
        length = max(size2f.height,size2f.width);
        width = min(size2f.height,size2f.width);
        ratio = length/width;
        // focus = boundingbox.center;
        float angle = boundingbox.angle;
        if (abs(angle) < 60)
            state = level;

        Rect rect = boundingbox.boundingRect();
        Point tl = rect.tl();
        Point br = rect.br();
        int xmin = tl.x;
        int ymin = tl.y;
        int xmax = br.x;
        int ymax = br.y;
        double white = 0;
        int x_min = 1000;
        int x_max = 0;
        int y_min = 1000;
        int y_max = 0;
        double x_sum = 0.0;
        double y_sum = 0.0;
        int total = (ymax-ymin)*(xmax-xmin);
        double A = 0.0,B = 0.0,C = 0.0,D = 0.0;
        int count = 0;
        vector<Point2d > middle;
        for (int i = ymin; i <= ymax; ++i) {
            int row_overlap = 0;
            double rowx_sum = 0.0;
            int row_count = 0;
            for (int j = xmin; j <= xmax; ++j) {
                if (image.ptr<uchar >(i)[j] != image.ptr<uchar >(i)[j+1])
                    row_overlap++;
                if (image.ptr<uchar >(i)[j] > 200){
                    white++;
                    row_count++;
                    rowx_sum += j;
                    A += j*j;
                    B += j;
                    C += i*j;
                    D += i;
                    x_sum += j;
                    y_sum += i;
                    if (i > y_max)
                        y_max = i;
                    if (i < y_min)
                        y_min = i;
                    if (j > x_max)
                        x_max = j;
                    if (j < x_min)
                        x_min = j;
                }
            }
            if (row_count != 0)
                middle.emplace_back(Point2d(rowx_sum/row_count,i));
            if (row_overlap > 3)
                count++;
        }

        if (count > 3)
            is_overlap = true;
        // is_overlap = false;
        //k = (C*white-B*D)/(A*white-B*B);
        //b = (A*D-C*B)/(A*white-B*B);
        cv::Vec4f line_para;
        fitLine(middle,line_para,cv::DIST_HUBER, 0, 1e-2, 1e-2);
        k = line_para[1] / line_para[0];
        b = 0;
        theta = abs(atan(k))*180/M_PI;
        downcenter = Point2d(line_para[2]+(ymax-line_para[3])/k,ymax);
        upcenter = Point2d(line_para[2]+(ymin-line_para[3])/k,ymin);
        L = sqrt(pow(down.x-up.x,2)+pow((down.y-up.y),2));
        //int c = 0;
        Moments mu = moments(contour,false);
#ifdef _DEBUG
        cout << "mu.x : " << mu.m10/(mu.m00+1e-5) << endl;
        cout << "mu.y : " << mu.m01/(mu.m00+1e-5) << endl;
#endif
        focus.x = mu.m10/(mu.m00+1e-5);
        focus.y = mu.m01/(mu.m00+1e-5);
/*
        focus.x = 0.0;
        focus.y = 0.0;

        for (int l = 0; l < middle.size(); ++l) {
            focus += middle[l];
            c++;
        }
        focus = focus/c;
        focus.x = boundingbox.center.x;
        focus.y = boundingbox.center.y;
        coloratio = white/(total+1);*/

    }

}

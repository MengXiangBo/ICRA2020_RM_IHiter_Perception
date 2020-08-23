#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string.h>

using namespace std;
using namespace cv;
string POSITIVE = "/home/mrm/ICRA/CODE/SIGN/P/";
string NEGTIVE = "/home/mrm/ICRA/CODE/SIGN/N/";
string POSITIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/P/config.txt";
string NEGTIVE_CONFIG = "/home/mrm/ICRA/CODE/SIGN/N/config.txt";
string VIDEO_PATH = "/home/mrm/ICRA/CODE/Dahuacam/video/down.avi";
bool select_flag = false;
bool has_rect = false;
int edge = -1; // 1,2,3,4分别代表上下左右
int relative_x,relative_y,origin_x,origin_y;
unsigned long skip_num = 0;
Mat image, showImg, backImg;
Rect area;

void on_Mouse(int event, int x, int y, int flags, void*param)
{
    if (!has_rect && event == EVENT_LBUTTONDOWN)
    {
        area.x = x;
        area.y = y;
        origin_x = x;
        origin_y = y;
        select_flag = true;
    }
    else if (!has_rect && select_flag && event == EVENT_MOUSEMOVE)
    {
        if (x > area.x)
            area.width = x - area.x;
        else{
            area.x = x;
            area.width = origin_x - area.x;
        }

        if (y > area.y)
            area.height = y - area.y;
        else{
            area.y = y;
            area.height = origin_y - area.y;
        }
    }
    else if (!has_rect && select_flag && event == EVENT_LBUTTONUP)
    {
        select_flag = false;
        has_rect = true;
    }
    else if (has_rect && event == EVENT_LBUTTONDOWN)
    {
        if (x > area.x && x < area.x + area.width){
            if (abs(y - area.y) < 6){
                select_flag =  true;
                relative_y = y;
                edge = 1;
            }
            else if (abs(y - (area.y+area.height)) < 6){
                select_flag = true;
                relative_y = y;
                edge = 2;
            }
        }
        else if (y > area.y && y < area.y + area.height){
            if (abs(x - area.x) < 6){
                select_flag =  true;
                relative_x = x;
                edge = 3;
            }
            else if (abs(x - (area.x+area.width)) < 6){
                select_flag = true;
                relative_x = x;
                edge = 4;
            }
        }
    }
    else if (has_rect && event == EVENT_MOUSEMOVE && select_flag)
    {
        switch (edge){
            case 1:{
                area.y += y - relative_y;
                area.height += relative_y - y;
                relative_y = y;
                break;
            }
            case 2:{
                area.height += y - relative_y;
                relative_y = y;
                break;
            }
            case 3:{
                area.x += x - relative_x;
                area.width += relative_x - x;
                relative_x = x;
                break;
            }
            case 4:{
                area.width += x - relative_x;
                relative_x = x;
                break;
            }
            default:
                break;
        }
    }
    else if (has_rect && event == EVENT_LBUTTONUP && select_flag)
    {
        select_flag = false;
        edge = -1;
    }
}

int main() {
    bool add_mode = false;
    unsigned long count = 0;
    unsigned long positive_count = 0;
    unsigned long negtive_count = 0;
    ofstream positive_writer;
    ofstream negtive_writer;

    if (add_mode){
        ifstream pw(POSITIVE_CONFIG);
        ifstream nw(NEGTIVE_CONFIG);
        while (!pw.eof()){
            string s;
            getline(pw,s);
            positive_count++;
        }

        while (!nw.eof()){
            string s;
            getline(nw,s);
            negtive_count++;
        }

        pw.close();
        nw.close();

        positive_writer.open(POSITIVE_CONFIG, ios::app | ios::in);
        negtive_writer.open(NEGTIVE_CONFIG, ios::app | ios::in);
    }
    else{
        positive_writer.open(POSITIVE_CONFIG, ios::in);
        negtive_writer.open(NEGTIVE_CONFIG, ios::in);
    }

    if(!positive_writer.is_open () || !negtive_writer.is_open()){
        cout << "Open config file failure !" << endl;
        return -1;
    }
    VideoCapture videoCapture(VIDEO_PATH);
    if (!videoCapture.isOpened()){
        cout << "Open video failure !" << endl;
        return -1;
    }


    bool close_flag = false;
    while (true) {
        videoCapture >> image;
        if (image.empty()){
            cout << "video end !" << endl;
            break;
        }

        count++;

        if (count < skip_num)
            continue;

        imshow(to_string(count),image);

        while (true)
        {
            image.copyTo(showImg);
            int key = waitKey(10);
            setMouseCallback(to_string(count), on_Mouse, nullptr);
            /*
            if (key == 'b'){
                if (backImg.empty()) {
                    cout << "Back Image is empty !" << endl;
                    return -1;
                }
                destroyWindow(to_string(count));
                imshow(to_string(count-1),backImg);
                while (true){
                    backImg.copyTo(showImg);
                    int key = waitKey(10);
                    setMouseCallback(to_string(count-1), on_Mouse, nullptr);
                    if (key == 'p'){

                    }
                }

            }*/
            if (select_flag || has_rect)
                rectangle(showImg,area,Scalar(0,255,0),2);
            imshow(to_string(count),showImg);
            if (key == 'p')
            {
                if (add_mode)
                    positive_count--;
                positive_count++;
                imwrite(POSITIVE+to_string(positive_count)+".png",image(area));
                // count_num width height ratio
                positive_writer << positive_count << " ";
                positive_writer << area.height << " " << area.width << " " << double(area.height)/area.width << "\n";
                cout << positive_count << " positive image has been saved !" << endl;
                has_rect = false;
                select_flag = false;
            }
            else if (key == 'n')
            {
                if (add_mode)
                    negtive_count--;
                negtive_count++;
                imwrite(NEGTIVE+to_string(negtive_count)+".png",image(area));
                // count_num width height ratio
                negtive_writer << negtive_count << " ";
                negtive_writer << area.height << " " << area.width << " " << double(area.height)/area.width << "\n";
                cout << negtive_count << " negtive image has been saved !" << endl;
                has_rect = false;
                select_flag = false;
            }
            else if (key == 'a'){
                cout << "next !" << endl;
                destroyWindow(to_string(count));
                break;
            }
            else if (key == 's'){
                skip_num = count + 5;
                cout << "next !" << endl;
                destroyWindow(to_string(count));
                break;
            }
            else if (key == 'd'){
                skip_num = count + 60;
                cout << "next !" << endl;
                destroyWindow(to_string(count));
                break;
            }
            else if (key == 'f'){
                cout << endl << "--------------------" << endl;
                cout << "help : \n";
                cout << "a : 跳过当前帧" << endl;
                cout << "s : 跳过5帧" << endl;
                cout << "d : 跳过60帧" << endl;
                cout << "p : 保存positive状态" << endl;
                cout << "n : 保存negtive状态" << endl;
                cout << "q : 退出" << endl;
                cout << "--------------------" << endl;
            }
            else if (key == 27||key=='q'){
                destroyWindow(to_string(count));
                close_flag = true;
                break;
            }

        }
        // image.copyTo(backImg);
        has_rect = false;
        select_flag = false;

        if (close_flag)
            break;
    }
    positive_writer.close();
    negtive_writer.close();
    return 0;
}
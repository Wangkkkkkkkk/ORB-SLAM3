/**
* This file is part of FORB-SLAM
*/

#ifndef ACCELERATE_H
#define ACCELERATE_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

namespace ORB_SLAM3
{

class Accelerate
{
public:

    Accelerate();

    ~Accelerate(){}

    void getImage(Mat images);
    void getGFpoints(vector<Point2f> GFpoints, int _level);
    vector<vector<int> > buildStat(int _nCols, int _nRows, int _wCell, int _hCell, int _minBorderX, int _minBorderY);  // 构建统计数组
    float getDensity();
    void addEdge();
    void getAllKeypoints(vector<vector<KeyPoint> > allkeypoints);
    void save();


    Mat mImages;                 // 当前图像帧
    vector<Point2f> vGFpoints;   // 优地图点
    vector<vector<KeyPoint> > vAllkeypoints;
    int nNumber;                     // 调试用，加速帧数
    Point2f pMove;                   // 像素移动方向
    vector<vector<int> > vStat;      // 统计矩阵
    int level;                       // 当前金字塔层级
    int nCols;
    int nRows;
    int wCell;
    int hCell;
    int minBorderX;
    int minBorderY;
    float density;

    // 金字塔第0层信息
    vector<Point2f> vGFpoints_origin;
    vector<vector<int> > vStat_origin;
    int nCols_origin;
    int nRows_origin;
    int wCell_origin;
    int hCell_origin;
    int minBorderX_origin;
    int minBorderY_origin;
};

} //namespace ORB_SLAM

#endif


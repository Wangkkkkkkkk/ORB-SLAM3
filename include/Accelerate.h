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

class Frame;

class Accelerate
{
public:

    Accelerate();

    ~Accelerate(){}

    void getImage(Mat images);
    void getPreframe(Frame* mPreframe);
    void getGFpoints(vector<Point2f> GFpoints, int _level);
    vector<vector<int> > buildStat(int _nCols, int _nRows, int _wCell, int _hCell, int _minBorderX, int _minBorderY);  // 构建统计数组
    float getDensity();
    void addEdge();
    void getAllKeypoints(vector<vector<KeyPoint> > allkeypoints);
    void saveExtractor();
    void save2Ddis();


    Mat mImages;                 // 当前图像帧
    // Frame* mPreframe;             // 前一帧信息

    Mat mPredictTcw_last;         // 上一帧预测的相机变换矩阵
    Mat mTcw_pre;                // 上一帧相机变换矩阵
    Mat mPredictTcw;             // 当前帧预测的相机变换矩阵
    
    Point2f pMove;                   // 像素移动方向


    // 金字塔第0层信息
    vector<Point2f> vGFpoints_origin;
    vector<vector<int> > vStat_origin;
    int nCols_origin;
    int nRows_origin;
    int wCell_origin;
    int hCell_origin;
    int minBorderX_origin;
    int minBorderY_origin;

    int level;                       // 当前金字塔层级
    vector<Point2f> vGFpoints;       // 优地图点
    vector<vector<int> > vStat;      // 统计矩阵
    int nCols;
    int nRows;
    int wCell;
    int hCell;
    int minBorderX;
    int minBorderY;

    float density;                 // 提取特征区域密度

    vector<vector<KeyPoint> > vAllkeypoints;  // 提取的所有特征点

    int nNumber;                     // 调试用，帧数

    vector<Point2f> vpDis;         // 恒速模型投影误差

};

} //namespace ORB_SLAM

#endif


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

    Accelerate(int _nlevels);

    ~Accelerate(){}

    void getImage(Mat images);
    void getPreframe(Frame* mPreframe);
    vector<vector<int> > buildStat(int _nCols, int _nRows, int _wCell, int _hCell, 
                                   int _minBorderX, int _minBorderY, int _maxBorderX, int _maxBorderY,
                                   int _level, float _W);  // 构建统计数组
    void computeMean();
    void addCellEdge(int n_x, int n_y, int _col, int _row);
    void addDensity();
    void addEdge();
    void getAllKeypoints(vector<vector<KeyPoint> > allkeypoints);
    void saveExtractor();
    void save2Ddis();

    int nlevels;
    Mat mImages;                 // 当前图像帧

    Mat mPredictTcw_last;         // 上一帧预测的相机变换矩阵
    Mat mTcw_pre;                // 上一帧相机变换矩阵
    Mat mPredictTcw;             // 当前帧预测的相机变换矩阵

    int nW;
    int level;                       // 当前金字塔层级
    vector<vector<Point2f> > vGFpoints;       // 优地图点
    vector<vector<vector<int> > > vStat;      // 统计矩阵
    vector<vector<vector<int> > > vStat_pre;      // 统计矩阵
    vector<vector<vector<int> > > vStat_out;      // 统计矩阵

    vector<int> nCols;
    vector<int> nRows;
    vector<int> wCell;
    vector<int> hCell;
    vector<int> minBorderX;
    vector<int> minBorderY;
    vector<int> maxBorderX;
    vector<int> maxBorderY;

    vector<Point2f> pMove;                   // 像素移动方向

    vector<vector<Point2f> > vpDis;         // 恒速模型投影误差
    vector<Point2f> pfCenter;                // 误差均值
    vector<float> fVariance;               // 误差方差
    vector<float> fDistance;

    vector<float> density;                 // 提取特征区域密度

    vector<vector<KeyPoint> > vAllkeypoints;  // 提取的所有特征点

    int nNumber;                     // 调试用，保存帧数

    int dis_parallel[4][2];
    int dis_tilted[4][2];

    

};

} //namespace ORB_SLAM

#endif


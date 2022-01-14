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

    Accelerate(int _nlevels, float _factor);

    ~Accelerate(){}

    void getImage(Mat images);
    void getFrame(Frame* mPreframe);
    vector<vector<int> > buildStat(int _nCols, int _nRows, int _wCell, int _hCell, 
                                   int _minBorderX, int _minBorderY, int _maxBorderX, int _maxBorderY,
                                   int _level, float _W, float _width, float _heights);  // 构建统计数组
    void computeProject();
    Point2f project(Mat _camera);
    void computeHomography();
    void computeMean();
    void addCellEdge(int n_x, int n_y, int _col, int _row);
    void addDensity();
    void computeDensity();
    void addEdge();
    vector<float> computeNode(Point2f p1, Point2f p2);
    void addTopEdge(vector<float> node);
    void addLeftEdge(vector<float> node);
    void addRightEdge(vector<float> node);
    void addDownEdge(vector<float> node);
    void saveExtractor(vector<vector<KeyPoint> > allkeypoints);
    void save2Ddis();

    bool isGetK;    // 判断是否得到内参矩阵

    int nlevels;      // 金字塔总层数
    float factor;     // 缩放因子 1.2

    Mat mImages;                 // 当前图像帧
    Mat mImagesLast;
    Frame* mLastFrame;           // 上一帧信息
    int nImages_width;
    int nImages_height;

    Mat mPredictTcw_last;         // 上一帧预测的相机变换矩阵
    Mat mTcw_pre;                // 上一帧相机变换矩阵
    Mat mPredictTcw;             // 当前帧预测的相机变换矩阵

    int nProjectNumber;

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
    vector<float> widths;
    vector<float> heights;

    vector<vector<Point2f> > vpDis;         // 恒速模型投影误差
    vector<Point2f> pfCenter;                // 误差均值
    vector<float> fVariance;               // 误差方差
    vector<float> fDistance;

    vector<float> density;                 // 提取特征区域密度

    int nNumberAll;                     // 调试用，保存帧数
    int nNumber;

    int dis_parallel[4][2];
    int dis_tilted[4][2];

    vector<int> vfResponses;

    vector<float> mvParameters;  // 相机内参信息
    Mat K;   // 相机内参矩阵
    Mat H;   // Homography
    float d;  // 平面距离

    // 平面投影的点的坐标
    Point2f pImage_lefttop;
    Point2f pImage_leftdown;
    Point2f pImage_righttop;
    Point2f pImage_rightdown;

};

} //namespace ORB_SLAM

#endif


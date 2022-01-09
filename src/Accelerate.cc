/**
* This file is part of FORB-SLAM
*/

#include <iostream>
#include <System.h>
#include <opencv2/opencv.hpp>
#include <Accelerate.h>
#include <Eigen/Dense>

using namespace Eigen; 
using namespace std;
using namespace cv;

namespace ORB_SLAM3
{

Accelerate::Accelerate(int _nlevels, float _factor):
    nlevels(_nlevels), factor(_factor)
{
    nNumber = -1;

    vGFpoints.resize(nlevels);
    vStat.resize(nlevels);
    vStat_pre.resize(nlevels);
    vStat_out.resize(nlevels);
    nCols.resize(nlevels);
    nRows.resize(nlevels);
    wCell.resize(nlevels);
    hCell.resize(nlevels);
    minBorderX.resize(nlevels);
    minBorderY.resize(nlevels);
    maxBorderX.resize(nlevels);
    maxBorderY.resize(nlevels);

    pMoves.resize(nlevels);
    pMove.resize(nlevels);

    vpDis.resize(nlevels);
    pfCenter.resize(nlevels);
    fVariance.resize(nlevels);
    fDistance.resize(nlevels);

    density.resize(nlevels);

    int _dis_parallel[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    int _dis_tilted[4][2] = {{1,1},{1,-1},{1,1},{-1,1}};

    memcpy(dis_parallel, _dis_parallel, sizeof(_dis_parallel));
    memcpy(dis_tilted, _dis_tilted, sizeof(_dis_tilted));
}

void Accelerate::getImage(Mat images) {
    nNumber++;
    mImages = images.clone();
}

void Accelerate::getPreframe(Frame* mPreframe) {
    // 得到相机变换矩阵信息
    mTcw_pre = mPreframe->mTcw;
    mPredictTcw = mPreframe->mPredictTcw;
    mPredictTcw_last = mPreframe->mPredictTcw_last;

    // 优特征点投影
    Mat Rcw = mTcw_pre.rowRange(0,3).colRange(0,3);
    Mat tcw = mTcw_pre.rowRange(0,3).col(3);

    Mat Rcw_last = mPredictTcw.rowRange(0,3).colRange(0,3);
    Mat tcw_last = mPredictTcw.rowRange(0,3).col(3);

    Mat Rcw_prelast = mPredictTcw_last.rowRange(0,3).colRange(0,3);
    Mat tcw_prelast = mPredictTcw_last.rowRange(0,3).col(3);

    vGFpoints[0].clear();
    vfResponses.clear();
    vpDis[0].clear();
    pMoves[0].clear();
    for(int i=0; i<mPreframe->N; i++)
    {
        if(mPreframe->mvpMapPoints[i])
        {
            if(!mPreframe->mvbOutlier[i])
            {
                mPreframe->mvpMapPoints[i]->IncreaseFound();
                if(mPreframe->mvpMapPoints[i]->Observations()>0)
                {
                    // 提前将优特征点投影到预测下一帧的图像上，下一帧位姿采用恒速模型
                    cv::Mat x3Dw = mPreframe->mvpMapPoints[i]->GetWorldPos();

                    cv::Mat x3Dc = Rcw * x3Dw + tcw;
                    Point2f uv = mPreframe->mpCamera->project(x3Dc);
                    
                    cv::Mat pc = Rcw_last * x3Dw + tcw_last;
                    cv::Point2f px = mPreframe->mpCamera->project(pc);

                    cv::Mat x3Dc_pre = Rcw_prelast * x3Dw + tcw_prelast;
                    cv::Point2f uv_pre = mPreframe->mpCamera->project(x3Dc_pre);

                    vGFpoints[0].push_back(px);
                    vfResponses.push_back(mPreframe->mvpMapPoints[i]->response);
                    vpDis[0].push_back(uv - uv_pre);

                    // 像素的移动方向，上一帧像素坐标 - 预测帧像素坐标
                    pMoves[0].push_back(uv - px);
                }
            }
        }
    }
}

vector<vector<int> > Accelerate::buildStat(int _nCols, int _nRows, int _wCell, int _hCell,
                                           int _minBorderX, int _minBorderY, int _maxBorderX, int _maxBorderY,
                                           int _level, float _W) {
    level = _level;
    nW = _W;
    nCols[level] = _nCols;
    nRows[level] = _nRows;
    wCell[level] = _wCell;
    hCell[level] = _hCell;
    minBorderX[level] = _minBorderX;
    minBorderY[level] = _minBorderY;
    maxBorderX[level] = _maxBorderX;
    maxBorderY[level] = _maxBorderY;

    vStat[level].clear();
    vector<int> vstat(nCols[level]);
    vStat[level].resize(nRows[level],vstat);

    computeMean();

    // 根据投影点选择提取特征区域
    for (int i=0;i<vGFpoints[level].size();i++) {
        int _col = vGFpoints[level][i].x;
        int _row = vGFpoints[level][i].y;
        int n_x = (_col - minBorderX[level]) / wCell[level];
        int n_y = (_row - minBorderY[level]) / hCell[level];
        if (n_x < 0 || n_y < 0 || n_x > nCols[level]-1 || n_y > nRows[level]-1) {
            continue;
        }
        vStat[level][n_y][n_x]++;

        // 添加网格边缘点
        addCellEdge(n_x, n_y, _col, _row);
    }

    // 添加图像边缘特征提取
    addEdge();

    // 添加特征提取区域的密度
    addDensity();

    return vStat_out[level];
}

// 计算均值及方差
void Accelerate::computeMean() {
    if (level == 0) {
        int num = vpDis[level].size();
        Eigen::MatrixXd mMat(num,2);
        for (int i=0;i<num;i++) {
            mMat(i, 0) = vpDis[level][i].x;
            mMat(i, 1) = vpDis[level][i].y;
        }

        // mean
        Eigen::MatrixXd _center = mMat.colwise().mean();
        float x0 = _center(0);
        float y0 = _center(1);
        pfCenter[level] = cvPoint2D32f(x0, y0);

        // 方差
        for (int i=0;i<num;i++) {
            mMat(i, 0) -= _center(0);
            mMat(i, 1) -= _center(1);
            mMat(i, 0) *= mMat(i, 0);
            mMat(i, 1) *= mMat(i, 1);
        }
        Eigen::MatrixXd disSum = mMat.colwise().sum();
        disSum(0) = disSum(0) / (num - 1);
        disSum(1) = disSum(1) / (num - 1);
        fVariance[level] = disSum(0) + disSum(1);

        fDistance[level] = sqrt(pow(pfCenter[level].x, 2) + pow(pfCenter[level].y, 2)) + fVariance[level];
        if (fDistance[level] > nW) {
            fDistance[level] = nW;
        }
    }
    else {
        pfCenter[level] = pfCenter[level-1] / factor;
        fVariance[level] = fVariance[level-1] / factor;
        fDistance[level] = fDistance[level-1] / factor;
    }
}


void Accelerate::addCellEdge(int n_x, int n_y, int _col, int _row) {
    float c0_x = n_x * wCell[level] + minBorderX[level]; // 网格左上角
    float c0_y = n_y * hCell[level] + minBorderY[level];
    float c1_x = n_x * wCell[level] + wCell[level] + minBorderX[level]; // 网格右下角
    float c1_y = n_y * hCell[level] + hCell[level] + minBorderY[level];

    // 判断上下左右是否超出
    for (int i=0;i<4;i++) {
        int _x = _col + dis_parallel[i][0] * fDistance[level];
        int _y = _row + dis_parallel[i][1] * fDistance[level];

        if (_x > c1_x && n_x + 1 < nCols[level]) {
            vStat[level][n_y][n_x + 1]++;
        }
        if (_x < c0_x && n_x - 1 >= 0) {
            vStat[level][n_y][n_x - 1]++;
        }
        if (_y > c1_y && n_y + 1 < nRows[level]) {
            vStat[level][n_y+1][n_x]++;
        }
        if (_y < c0_y && n_y - 1 >= 0) {
            vStat[level][n_y-1][n_x]++;
        }
    }

    // 判断斜方向的上下左右是否超出
    for (int i=0;i<4;i++) {
        int _x = _col + dis_tilted[i][0] * fDistance[level];
        int _y = _row + dis_tilted[i][1] * fDistance[level];

        if (_x > c1_x && _y > c1_y && n_x + 1 < nCols[level] && n_y + 1 < nRows[level]) {
            vStat[level][n_y+1][n_x+1]++;
        }
        if (_x < c0_x && _y > c1_y && n_x - 1 >= 0 && n_y + 1 < nRows[level]) {
            vStat[level][n_y+1][n_x-1]++;
        }
        if (_x > c1_x && _y < c0_y && n_x + 1 < nCols[level] && n_y - 1 >= 0) {
            vStat[level][n_y-1][n_x+1]++;
        }
        if (_x < c0_x && _y < c0_y && n_x - 1 >= 0 && n_y - 1 >= 0) {
            vStat[level][n_y-1][n_x-1]++;
        }
    }
}

void Accelerate::addEdge() {

    Point2f move = computeMove();

    int _move = 0;
    if (move.x < -1) {
        _move = ceil(-move.x/ nW);
        for (int c=0;c<_move;c++) {
            for (int i=0;i<nRows[level];i++) {
                vStat[level][i][c]++;
            }
        }
    }
    else if (move.x > 1) {
        _move = ceil(move.x / nW);
        for (int c=0;c<_move;c++) {
            for (int i=0;i<nRows[level];i++) {
                vStat[level][i][nCols[level]-c-1]++;
            }
        }
    }
    if (move.y < -1) {
        _move = ceil(-move.y / nW);
        for (int r=0;r<_move;r++) {
            for (int i=0;i<nCols[level];i++) {
                vStat[level][r][i]++;
            }
        }
    }
    else if (move.y > 1) {
        _move = ceil(move.y / nW);
        for (int r=0;r<_move;r++) {
            for (int i=0;i<nCols[level];i++) {
                vStat[level][nRows[level]-r-1][i]++;
            }
        }
    }
}

void Accelerate::addDensity() {
    vStat_out[level].clear();
    vStat_out[level].resize(vStat[level].size());
    for (int i=0;i<vStat[level].size();i++) {
        vStat_out[level][i].assign(vStat[level][i].begin(), vStat[level][i].end());
    }

    if (nNumber != 0) {
        for (int r=0;r<vStat_out[level].size();r++) {
            for (int c=0;c<vStat_out[level][r].size();c++) {
                vStat_out[level][r][c] += vStat_pre[level][r][c];
            }
        }
    }

    int nEar = 0;
    for (int r=0;r<nRows[level];r++) {
        for (int i=0;i<nCols[level];i++) {
            if (vStat_out[level][r][i] > 0) {
                nEar++;
            }
        }
    }
    density[level] = float(nEar) / (nRows[level] * nCols[level]);

    vStat_pre[level].clear();
    vStat_pre[level].resize(vStat[level].size());
    for (int i=0;i<vStat[level].size();i++) {
        vStat_pre[level][i].assign(vStat[level][i].begin(), vStat[level][i].end());
    }

}

Point2f Accelerate::computeMove() {
    if (level == 0) {
        int num = pMoves[level].size();
        Point2f _move = cvPoint(0, 0);
        for (int i=0;i<pMoves[level].size();i++) {
            _move += pMoves[level][i];
        }
        pMove[level] = _move / num;
        return pMove[level];
    }
    else {
        pMove[level] = pMove[level-1] / factor;
        return pMove[level];
    }
}

// save project points info and extractor points info
void Accelerate::saveExtractor(vector<vector<KeyPoint> > allkeypoints) {
    Mat _images;
    cvtColor(mImages, _images, COLOR_GRAY2BGR);
    string numb = to_string(nNumber);
    string filename = "/home/kai/file/VO_SpeedUp/Dataset/feature_extractor/" + numb + ".png";
    // 投影特征点
    for (int i=0;i<vGFpoints[0].size();i++) {
        int x = vGFpoints[0][i].x;
        int y = vGFpoints[0][i].y;
        int response = vfResponses[i];
        // putText(_images, to_string(response), cvPoint(x, y-10), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1);
        circle(_images, cvPoint(x,y), 5, Scalar(0, 0, 255), 2, 4, 0);
    }
    // 提取特征点
    for (int j=0;j<allkeypoints[0].size();j++) {
        int x = allkeypoints[0][j].pt.x;
        int y = allkeypoints[0][j].pt.y;
        int response = allkeypoints[0][j].response;
        // putText(_images, to_string(response), cvPoint(x, y-10), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1);
        circle(_images, cvPoint(x,y), 5, Scalar(0, 255, 0), 2, 4, 0);
    }
    // 特征提取区域
    for (int i=0;i<nRows[0];i++) {
        float iniY = minBorderY[0] + i * hCell[0];
        float maxY = iniY + hCell[0];
        if(maxY>maxBorderY[0])
            maxY = maxBorderY[0];
        for (int j=0;j<nCols[0];j++) {
            float iniX = minBorderX[0] + j * wCell[0];
            float maxX = iniX + wCell[0];
            if(maxX>maxBorderX[0])
                maxX = maxBorderX[0];
            if (vStat_out[0][i][j] > 0) {
                Point2f pt1;
                Point2f pt2;
                pt1.x = iniX;
                pt1.y = iniY;
                pt2.x = maxX;
                pt2.y = maxY;
                rectangle(_images, pt1, pt2, cvScalar(255, 0, 0), 1, 4, 0);
            }
        }
    }
    int dens = density[0] * 100;
    putText(_images, to_string(dens) + "%", cvPoint(0, 470), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(255, 0, 0), 1);
    imwrite(filename, _images);
}

void Accelerate::save2Ddis() {
    Mat img(1000, 1000, CV_8UC3, Scalar(255,255,255));

    string numb = to_string(nNumber);
    string filename = "/home/kai/file/VO_SpeedUp/Dataset/feature_projectDis/" + numb + ".png";

    arrowedLine(img, Point(0, 500), Point(1000, 500), Scalar(0, 0, 0), 2, 8, 0, 0.02);
    arrowedLine(img, Point(500, 0), Point(500, 1000), Scalar(0, 0, 0), 2, 8, 0, 0.02);

    // X 轴正向
    line(img, Point(550, 500), Point(550, 490), Scalar(0, 0, 0), 2);
    putText(img, "1", cvPoint(545, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(600, 500), Point(600, 490), Scalar(0, 0, 0), 2);
    putText(img, "2", cvPoint(595, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(650, 500), Point(650, 490), Scalar(0, 0, 0), 2);
    putText(img, "3", cvPoint(645, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(700, 500), Point(700, 490), Scalar(0, 0, 0), 2);
    putText(img, "4", cvPoint(695, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(750, 500), Point(750, 490), Scalar(0, 0, 0), 2);
    putText(img, "5", cvPoint(745, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(800, 500), Point(800, 490), Scalar(0, 0, 0), 2);
    putText(img, "6", cvPoint(795, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(850, 500), Point(850, 490), Scalar(0, 0, 0), 2);
    putText(img, "7", cvPoint(845, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(900, 500), Point(900, 490), Scalar(0, 0, 0), 2);
    putText(img, "8", cvPoint(895, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(950, 500), Point(950, 490), Scalar(0, 0, 0), 2);
    putText(img, "9", cvPoint(945, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);

    // X 轴负向
    line(img, Point(450, 500), Point(450, 490), Scalar(0, 0, 0), 2);
    putText(img, "-1", cvPoint(440, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(400, 500), Point(400, 490), Scalar(0, 0, 0), 2);
    putText(img, "-2", cvPoint(390, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(350, 500), Point(350, 490), Scalar(0, 0, 0), 2);
    putText(img, "-3", cvPoint(340, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(300, 500), Point(300, 490), Scalar(0, 0, 0), 2);
    putText(img, "-4", cvPoint(290, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(250, 500), Point(250, 490), Scalar(0, 0, 0), 2);
    putText(img, "-5", cvPoint(240, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(200, 500), Point(200, 490), Scalar(0, 0, 0), 2);
    putText(img, "-6", cvPoint(190, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(150, 500), Point(150, 490), Scalar(0, 0, 0), 2);
    putText(img, "-7", cvPoint(140, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(100, 500), Point(100, 490), Scalar(0, 0, 0), 2);
    putText(img, "-8", cvPoint(90, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);
    line(img, Point(50, 500), Point(50, 490), Scalar(0, 0, 0), 2);
    putText(img, "-9", cvPoint(40, 525), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0), 1);

    int num = vpDis[0].size();
    for (int i=0;i<num;i++) {
        int x = vpDis[0][i].x * 50 + 500;
        int y = vpDis[0][i].y * 50 + 500;
        circle(img, cvPoint(x,y), 3, Scalar(0, 255, 0), 2, 4, 0);
    }

    circle(img, cvPoint(500, 500), fDistance[0]*50, Scalar(0, 0, 255), 2, 4, 0);

    imwrite(filename, img);
}

}   //namespace ORB_SLAM
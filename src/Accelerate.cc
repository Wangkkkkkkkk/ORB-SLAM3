/**
* This file is part of FORB-SLAM
*/


#include<iostream>
#include<System.h>
#include <opencv2/opencv.hpp>
#include<Accelerate.h>

using namespace std;
using namespace cv;

namespace ORB_SLAM3
{

Accelerate::Accelerate() {
    nNumber = 0;
}

void Accelerate::getImage(Mat images) {
    mImages = images.clone();
}

void Accelerate::getGFpoints(vector<Point2f> GFpoints, int _level) {
    level = _level;
    vGFpoints.assign(GFpoints.begin(), GFpoints.end());
    if (level == 0) {
        vGFpoints_origin.assign(GFpoints.begin(), GFpoints.end());
        pMove = GFpoints[GFpoints.size()-1];
    }
}

vector<vector<int> > Accelerate::buildStat(int _nCols, int _nRows, int _wCell, int _hCell, int _minBorderX, int _minBorderY) {
    nCols = _nCols;
    nRows = _nRows;
    wCell = _wCell;
    hCell = _hCell;
    minBorderX = _minBorderX;
    minBorderY = _minBorderY;
    vector<int> vstat(nCols);
    vStat.resize(nRows,vstat);

    // 根据投影点选择提取特征区域
    for (int i=0;i<vGFpoints.size()-1;i++) {
        int _col = vGFpoints[i].x;
        int _row = vGFpoints[i].y;
        int n_x = (_col - minBorderX) / wCell;
        int n_y = (_row - minBorderY) / hCell;
        if (n_x < 0 || n_y < 0 || n_x > nCols-1 || n_y > nRows-1) {
            continue;
        }
        vStat[n_y][n_x]++;

        // 计算图像网格边缘点
        float c_x = n_x * wCell + 0.5 * wCell + minBorderX; // 网格中心点
        float c_y = n_y * hCell + 0.5 * hCell + minBorderY;
        float _x = _col - c_x; // 投影点与网格中心点的偏差
        float _y = _row - c_y;
        if (_x > 0 && n_x + 1 < nCols) {
            vStat[n_y][n_x + 1]++;
        }
        if (_x < 0 && n_x - 1 >= 0) {
            vStat[n_y][n_x - 1]++;
        }
        if (_y > 0 && n_y + 1 < nRows) {
            vStat[n_y+1][n_x]++;
        }
        if (_y < 0 && n_y - 1 >= 0) {
            vStat[n_y-1][n_x]++;
        }
        if (_x > 0 && _y > 0 && n_x + 1 < nCols && n_y + 1 < nRows) {
            vStat[n_y+1][n_x+1]++;
        }
        if (_x < 0 && _y > 0 && n_x - 1 >= 0 && n_y + 1 < nRows) {
            vStat[n_y+1][n_x-1]++;
        }
        if (_x > 0 && _y < 0 && n_x + 1 < nCols && n_y - 1 >= 0) {
            vStat[n_y-1][n_x+1]++;
        }
        if (_x < 0 && _y < 0 && n_x - 1 >= 0 && n_y - 1 >= 0) {
            vStat[n_y-1][n_x-1]++;
        }
    }
    addEdge();
    if (level == 0) {
        nCols_origin = _nCols;
        nRows_origin = _nRows;
        wCell_origin = _wCell;
        hCell_origin = _hCell;
        minBorderX_origin = _minBorderX;
        minBorderY_origin = _minBorderY;
        vStat_origin.resize(vStat.size());
        for (int i=0;i<vStat.size();i++) {
            vStat_origin[i].assign(vStat[i].begin(), vStat[i].end());
        }
    }
    return vStat;
}

void Accelerate::addEdge() {
    int col = 0;
    int row = 0;
    if (pMove.x > 1) {
        col = ceil(pMove.x / 35);
        if (col > 0) {
            for (int c=0;c<col;c++) {
                for (int i=0;i<nRows;i++) {
                    vStat[i][c]++;
                }
            }
        }
    }
    else if (pMove.x < -1) {
        col = ceil(-pMove.x / 35);
        if (col > 0) {
            for (int c=0;c<col;c++) {
                for (int i=0;i<nRows;i++) {
                    vStat[i][nCols-c-1]++;
                }
            }
        }
    }
    if (pMove.y > 1) {
        row = ceil(pMove.y / 35);
        if (row > 0) {
            for (int r=0;r<row;r++) {
                for (int i=0;i<nCols;i++) {
                    vStat[r][i]++;
                }
            }
        }
    }
    else if (pMove.y < -1) {
        row = ceil(-pMove.y / 35);
        if (row > 0) {
            for (int r=0;r<row;r++) {
                for (int i=0;i<nCols;i++) {
                    vStat[nRows-r-1][i]++;
                }
            }
        }
    }
}

float Accelerate::getDensity() {
    int nEar = 0;
    if (level == 0) {
        for (int r=0;r<nRows;r++) {
            for (int i=0;i<nCols;i++) {
                if (vStat[r][i] > 0) {
                    nEar++;
                }
            }
        }
        density = float(nEar) / (nRows * nCols);

        // mnFeaturesPerLevel.clear();
        // mnFeaturesPerLevel.resize(nlevels);
        // nfeatures = 1000; //300 + density * nfeatures;
        // float factor = 1.0f / scaleFactor;
        // float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));
        // int sumFeatures = 0;
        // for( int level = 0; level < nlevels-1; level++ )
        // {
        //     mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        //     sumFeatures += mnFeaturesPerLevel[level];
        //     nDesiredFeaturesPerScale *= factor;
        // }
        // mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);
    }
    return density;
}

void Accelerate::getAllKeypoints(vector<vector<KeyPoint> > allkeypoints){
    vAllkeypoints.resize(allkeypoints.size());
    for (int i=0;i<allkeypoints.size();i++) {
        vAllkeypoints[i].assign(allkeypoints[i].begin(), allkeypoints[i].end());
    }
}

void Accelerate::save() {
    Mat _images;
    cvtColor(mImages, _images, COLOR_GRAY2BGR);
    string numb = to_string(nNumber);
    string filename = "/home/kai/file/VO_SpeedUp/Dataset/feature_extractor/" + numb + ".png";
    // 投影特征点
    for (int i=0;i<vGFpoints_origin.size()-1;i++) {
        int x = vGFpoints_origin[i].x;
        int y = vGFpoints_origin[i].y;
        circle(_images, cvPoint(x,y), 5, Scalar(0, 0, 255), 2, 4, 0);
    }
    // 提取特征点
    for (int j=0;j<vAllkeypoints[0].size();j++) {
        int x = vAllkeypoints[0][j].pt.x;
        int y = vAllkeypoints[0][j].pt.y;
        circle(_images, cvPoint(x,y), 5, Scalar(0, 255, 0), 2, 4, 0);
    }
    // 特征提取区域
    for (int i=0;i<nRows_origin;i++) {
        float iniY = minBorderY_origin + i * hCell_origin;
        float maxY = iniY + hCell_origin;
        for (int j=0;j<nCols_origin;j++) {
            float iniX = minBorderX_origin + j * wCell_origin;
            float maxX = iniX + wCell_origin;
            if (vStat_origin[i][j] > 0) {
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
    int dens = density * 100;
    putText(_images, to_string(dens) + "%", cvPoint(0, 470), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(255, 0, 0), 1);
    imwrite(filename, _images);
    nNumber++;
}
}   //namespace ORB_SLAM
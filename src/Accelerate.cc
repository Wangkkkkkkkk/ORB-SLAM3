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
    nNumberAll = -1;
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
    widths.resize(nlevels);
    heights.resize(nlevels);

    vpDis.resize(nlevels);
    pfCenter.resize(nlevels);
    fVariance.resize(nlevels);
    fDistance.resize(nlevels);

    density.resize(nlevels);

    int _dis_parallel[4][2] = {{0,1},{0,-1},{1,0},{-1,0}};
    int _dis_tilted[4][2] = {{1,1},{1,-1},{1,1},{-1,1}};

    memcpy(dis_parallel, _dis_parallel, sizeof(_dis_parallel));
    memcpy(dis_tilted, _dis_tilted, sizeof(_dis_tilted));

    isGetK = false;
}

void Accelerate::getImage(Mat images) {
    nNumberAll++;
    if (nNumberAll > 0) {
        mImagesLast = mImages.clone();
    }
    mImages = images.clone();
    nImages_width = mImages.size().width;
    nImages_height = mImages.size().height;
}

void Accelerate::getFrame(Frame* mPreframe) {
    nNumber++;
    // cout<< "--- nNumber:" << nNumber << " ---" <<endl;
    mLastFrame = mPreframe;

    if (isGetK == false) {
        mvParameters.clear();
        int _n = mLastFrame->mpCamera->size();
        for (int j=0;j<_n;j++) {
            mvParameters.push_back(mLastFrame->mpCamera->getParameter(j));
        }
        K = (Mat_<float>(3, 3) << mvParameters[0], 0.f, mvParameters[2],
                                      0.f, mvParameters[1], mvParameters[3], 
                                      0.f, 0.f, 1.f);
        isGetK = true;
    }
}

vector<vector<int> > Accelerate::buildStat(int _nCols, int _nRows, int _wCell, int _hCell,
                                           int _minBorderX, int _minBorderY, int _maxBorderX, int _maxBorderY,
                                           int _level, float _W, float _width, float _height) {
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
    widths[level] = _width;
    heights[level] = _height;

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

    computeDensity();

    // 添加图像边缘特征提取
    addEdge();

    // 添加特征提取区域的密度
    addDensity();

    return vStat_out[level];
}

// 计算投影
void Accelerate::computeProject() {
    // 得到相机变换矩阵信息
    mTcw_pre = mLastFrame->mTcw;
    mPredictTcw = mLastFrame->mPredictTcw;
    mPredictTcw_last = mLastFrame->mPredictTcw_last;

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
    int num=0;
    d = 0;
    for(int i=0; i<mLastFrame->N; i++)
    {
        if(mLastFrame->mvpMapPoints[i])
        {
            if(!mLastFrame->mvbOutlier[i])
            {
                // 优特征点判断
                int _nProject = mLastFrame->mvpMapPoints[i]->nProjects;
                int _nMatch = mLastFrame->mvpMapPoints[i]->nMatchs;
                int _nIner = mLastFrame->mvpMapPoints[i]->nIners;
                int _nOuter = mLastFrame->mvpMapPoints[i]->nOuters;
                // cout<< "_nProject:" << _nProject << " _nMatch:" << _nMatch << " _nIner:" << _nIner << " _nOuter:" << _nOuter <<endl;

                if (_nProject > 10 && (_nIner + _nOuter) > 0) {
                    // cout<< "_nProject:" << _nProject << " _nMatch:" << _nMatch << " _nIner:" << _nIner << " _nOuter:" << _nOuter <<endl;
                    float _recall = float(_nMatch) / float(_nProject);
                    float _precision = float(_nIner) / float(_nIner + _nOuter);
                    // cout<< "_recall:" << _recall << "  _precision:" << _precision <<endl;

                    if (_recall < 0.8 || _precision < 0.8) {
                        continue;
                    }
                }
                

                // 提前将优特征点投影到预测下一帧的图像上，下一帧位姿采用恒速模型
                Mat _point_world = mLastFrame->mvpMapPoints[i]->GetWorldPos();

                Mat _predict_camera = Rcw_last * _point_world + tcw_last;
                Point2f _predict_uv = project(_predict_camera);

                if (_predict_uv.x < 0 || _predict_uv.y < 0 || _predict_uv.x > nImages_width || _predict_uv.y > nImages_height) {
                    continue;
                }

                Mat x3Dc = Rcw * _point_world + tcw;
                d += x3Dc.at<float>(2, 0);
                Point2f uv = project(x3Dc);

                Mat x3Dc_pre = Rcw_prelast * _point_world + tcw_prelast;
                Point2f uv_pre = project(x3Dc_pre);

                vGFpoints[0].push_back(_predict_uv);
                vfResponses.push_back(mLastFrame->mvpMapPoints[i]->response);
                vpDis[0].push_back(uv - uv_pre);

                num++;
            }
        }
    }
    d = d / num;

    if (nNumber > 0) {
       computeHomography();
    }
}

/** 
 * @brief 投影
 * xc​ = Xc/Zc, yc = Yc/Zc
 * r^2 = xc^2 + yc^2
 * θ = arctan(r)
 * θd = k0*θ + k1*θ^3 + k2*θ^5 + k3*θ^7 + k4*θ^9
 * xd = θd/r * xc   yd = θd/r * yc
 * u = fx*xd + cx  v = fy*yd + cy
 * @param p3D 三维点
 * @return 像素坐标
 */
Point2f Accelerate::project(Mat _camera) {
    Point3f p3D = Point3f(_camera.at<float>(0, 0), _camera.at<float>(1, 0), _camera.at<float>(2, 0));

    const float x2_plus_y2 = p3D.x * p3D.x + p3D.y * p3D.y;
    const float theta = atan2f(sqrtf(x2_plus_y2), p3D.z);
    const float psi = atan2f(p3D.y, p3D.x);

    const float theta2 = theta * theta;
    const float theta3 = theta * theta2;
    const float theta5 = theta3 * theta2;
    const float theta7 = theta5 * theta2;
    const float theta9 = theta7 * theta2;
    const float r = theta + mvParameters[4] * theta3 + mvParameters[5] * theta5 + mvParameters[6] * theta7 + mvParameters[7] * theta9;

    return Point2f(mvParameters[0] * r * cos(psi) + mvParameters[2],
                   mvParameters[1] * r * sin(psi) + mvParameters[3]);
}

// 计算单应矩阵
void Accelerate::computeHomography() {
    Mat Tcc = mPredictTcw * mTcw_pre.inv();   // 计算预测位姿相机和上一帧相机位姿变换

    Mat Rcc = Tcc.rowRange(0,3).colRange(0,3);  // 取出旋转矩阵
    Mat tcc = Tcc.rowRange(0,3).col(3);

    Mat _N = Mat::zeros(3, 3, CV_32F);
    Mat N = _N.rowRange(0,3).col(0);
    N.at<float>(2,0) = 1;

    Mat h = Rcc + tcc * N.t() / d;

    H = K * h * K.inv();

    Mat Image_lefttop = _N.rowRange(0, 3).col(0).clone();
    Image_lefttop.at<float>(0, 0) = 16;
    Image_lefttop.at<float>(1, 0) = 16;
    Image_lefttop.at<float>(2, 0) = 1;

    Mat Image_leftdown = _N.rowRange(0, 3).col(0).clone();
    Image_leftdown.at<float>(0, 0) = 16;
    Image_leftdown.at<float>(1, 0) = 464;
    Image_leftdown.at<float>(2, 0) = 1;

    Mat Image_righttop = _N.rowRange(0, 3).col(0).clone();
    Image_righttop.at<float>(0, 0) = 736;
    Image_righttop.at<float>(1, 0) = 16;
    Image_righttop.at<float>(2, 0) = 1;

    Mat Image_rightdown = _N.rowRange(0, 3).col(0).clone();
    Image_rightdown.at<float>(0, 0) = 736;
    Image_rightdown.at<float>(1, 0) = 464;
    Image_rightdown.at<float>(2, 0) = 1;

    Mat Image_lefttop_project = H * Image_lefttop;
    Mat Image_leftdown_project = H * Image_leftdown;
    Mat Image_righttop_project = H * Image_righttop;
    Mat Image_rightdown_project = H * Image_rightdown;

    pImage_lefttop = cvPoint2D32f(Image_lefttop_project.at<float>(0, 0), Image_lefttop_project.at<float>(1, 0));
    pImage_leftdown = cvPoint2D32f(Image_leftdown_project.at<float>(0, 0), Image_leftdown_project.at<float>(1, 0));
    pImage_righttop = cvPoint2D32f(Image_righttop_project.at<float>(0, 0), Image_righttop_project.at<float>(1, 0));
    pImage_rightdown = cvPoint2D32f(Image_rightdown_project.at<float>(0, 0), Image_rightdown_project.at<float>(1, 0));

    // 整副图像投影
    // Mat mProject;
    // warpPerspective(mImagesLast, mProject, H, cv::Size(752, 480), cv::INTER_LINEAR);
    
    // 单点投影
    // Mat mProject(480, 752, CV_8UC3, Scalar(255,255,255));

    // line(mProject, cvPoint(16,16), cvPoint(736,16), Scalar(0, 0, 255), 2);
    // line(mProject, cvPoint(16,16), cvPoint(16,464), Scalar(0, 0, 255), 2);
    // line(mProject, cvPoint(16,464), cvPoint(736,464), Scalar(0, 0, 255), 2);
    // line(mProject, cvPoint(736,16), cvPoint(736,464), Scalar(0, 0, 255), 2);

    // line(mProject, pImage_lefttop, pImage_leftdown, Scalar(255, 0, 0), 2);
    // line(mProject, pImage_leftdown, pImage_rightdown, Scalar(255, 0, 0), 2);
    // line(mProject, pImage_righttop, pImage_lefttop, Scalar(255, 0, 0), 2);
    // line(mProject, pImage_rightdown, pImage_righttop, Scalar(255, 0, 0), 2);

    // string numb = to_string(nNumber);
    // string filename = "/home/kai/file/VO_SpeedUp/Dataset/image_project/" + numb + ".png";
    // imwrite(filename, mProject);
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

        // 标准差
        for (int i=0;i<num;i++) {
            mMat(i, 0) -= _center(0);
            mMat(i, 1) -= _center(1);
            mMat(i, 0) *= mMat(i, 0);
            mMat(i, 1) *= mMat(i, 1);
        }
        Eigen::MatrixXd disSum = mMat.colwise().sum();
        disSum(0) = sqrt(disSum(0) / num);
        disSum(1) = sqrt(disSum(1) / num);
        fVariance[level] = sqrt(pow(disSum(0), 2) + pow(disSum(1), 2));

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

void Accelerate::computeDensity() {
    int nEar = 0;
    for (int r=0;r<nRows[level];r++) {
        for (int i=0;i<nCols[level];i++) {
            if (vStat[level][r][i] > 0) {
                nEar++;
            }
        }
    }
    density[level] = float(nEar) / (nRows[level] * nCols[level]);

    if (density[level] < 0.15) {
        for (int r=0;r<nRows[level];r++) {
            for (int i=0;i<nCols[level];i++) {
                vStat[level][r][i] = 1;
            }
        }
    }
}

void Accelerate::addEdge() {
    vector<float> top_node;
    vector<float> left_node;
    vector<float> right_node;
    vector<float> down_node;

    top_node.reserve(4);
    left_node.reserve(4);
    right_node.reserve(4);
    down_node.reserve(4);

    top_node = computeNode(pImage_lefttop, pImage_righttop);
    left_node = computeNode(pImage_lefttop, pImage_leftdown);
    right_node = computeNode(pImage_righttop, pImage_rightdown);
    down_node = computeNode(pImage_leftdown, pImage_rightdown);

    addTopEdge(top_node);
    addLeftEdge(left_node);
    addRightEdge(right_node);
    addDownEdge(down_node);
}

// 返回的四维数组：top交点、left交点、right交点、down交点
vector<float> Accelerate::computeNode(Point2f p1, Point2f p2) {
    vector<float> output;
    // 求边的直线方程 
    //     Ax + By + C = 0; 
    //     A = y2 - y1; 
    //     B = x1 - x2; 
    //     C = x2 * y1 - x1 * y2;

    float A = p2.y - p1.y;
    float B = p1.x - p2.x;
    float C = p2.x * p1.y - p1.x * p2.y;

    // 求与四个边界交点
    // top: y = 16;
    float top_x = (-C - B * 16) / A;
    top_x = top_x / pow(factor, level) - 16;
    output.push_back(top_x);

    // left: x = 16;
    float left_y = (-C - A * 16) / B;
    left_y = left_y / pow(factor, level) - 16;
    output.push_back(left_y);

    // right: x = width-16;
    float right_y = (-C - A  * (nImages_width-16)) / B;
    right_y = right_y / pow(factor, level) - 16;
    output.push_back(right_y);

    // down: y = height-16;
    float down_x = (-C - B * (nImages_height-16)) / A;
    down_x = down_x / pow(factor, level) - 16;
    output.push_back(down_x);

    return output;
}

void Accelerate::addTopEdge(vector<float> node) {
    float top_x = node[0];
    float left_y = node[1];
    float right_y = node[2];
    float down_x = node[3];
    // cout<< "Top:" << " top_x:" << top_x << " left_y:" << left_y << " right_x:" << right_y << " down_x:" << down_x <<endl;
    if (left_y > 1 && right_y > 1) {
        float _y = max(left_y, right_y);
        int _rows = ceil( _y  / (nW/2));
        if (_rows > nRows[level]) {_rows = nRows[level];}
        // cout<< "nRows:" << nRows[level] << " _rows:" << _rows <<endl;
        for (int r=0;r<_rows;r++) {
            for (int c=0;c<nCols[level];c++) {
                vStat[level][r][c]++;
            }
        }
    }
    else if ((left_y > 1 && right_y < -1) || (left_y < -1 && right_y > 1)) {
        if (left_y > right_y) {
            int _rows = ceil(left_y / (nW/2));
            int _cols = ceil(top_x / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            // cout<< "nRows:" << nRows[level] << " nCols:" << nCols[level] << " _rows:" << _rows << " _cols:" <<endl;
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][r][c]++;
                }
            }
        }
        else {
            int _rows = ceil(right_y / (nW/2));
            int _cols = int((widths[level] - top_x) / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            // cout<< "nRows:" << nRows[level] << " nCols:" << nCols[level] << " _rows:" << _rows << " _cols:" <<endl;
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][r][nCols[level]-c-1]++;
                }
            }
        }
    }
}

void Accelerate::addLeftEdge(vector<float> node) {
    float top_x = node[0];
    float left_y = node[1];
    float right_y = node[2];
    float down_x = node[3];
    if (top_x > 1 && down_x > 1) {
        float _x = max(top_x, down_x);
        int _cols = ceil( _x  / (nW/2));
        if (_cols > nCols[level]) {_cols = nCols[level];}
        for (int r=0;r<nRows[level];r++) {
            for (int c=0;c<_cols;c++) {
                vStat[level][r][c]++;
            }
        }
    }
    else if ((top_x > 1 && down_x < -1) || (top_x < -1 && down_x > 1)) {
        if (top_x > down_x) {
            int _rows = ceil(left_y / (nW/2));   // 采用向下取整，防止超出
            int _cols = ceil(top_x / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][r][c]++;
                }
            }
        }
        else {
            int _rows = ceil((heights[level] - left_y) / (nW/2));
            int _cols = ceil(top_x / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][nRows[level]-r-1][c]++;
                }
            }
        }
    }
}

void Accelerate::addRightEdge(vector<float> node) {
    float top_x = node[0];
    float left_y = node[1];
    float right_y = node[2];
    float down_x = node[3];
    // cout<< "Right:" << " top_x:" << top_x << " left_y:" << left_y << " right_y:" << right_y << " down_x:" << down_x <<endl;
    if (top_x < widths[level] && down_x < widths[level]) {
        float _x = min(top_x, down_x);
        int _cols = ceil((widths[level] - _x)  / (nW/2));
        if (_cols > nCols[level]) {_cols = nCols[level];}
        for (int r=0;r<nRows[level];r++) {
            for (int c=0;c<_cols;c++) {
                vStat[level][r][nCols[level]-c-1]++;
            }
        }
    }
    else if ((top_x < widths[level] && down_x > widths[level]) ||
             (top_x > widths[level] && down_x < widths[level])) {
        if (top_x < down_x) {
            int _rows = ceil(right_y / (nW/2));   // 采用向下取整，防止超出
            int _cols = ceil((widths[level] - top_x) / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][r][nCols[level]-c-1]++;
                }
            }
        }
        else {
            int _rows = ceil((heights[level] - right_y) / (nW/2));
            int _cols = ceil((widths[level] - down_x) / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][nRows[level]-r-1][nCols[level]-c-1]++;
                }
            }
        }
    }
}

void Accelerate::addDownEdge(vector<float> node) {
    float top_x = node[0];
    float left_y = node[1];
    float right_y = node[2];
    float down_x = node[3];
    // cout<< "Down:" << " top_x:" << top_x << " left_y:" << left_y << " right_y:" << right_y << " down_x:" << down_x <<endl;
    if (left_y < heights[level] && right_y < heights[level]) {
        float _y = min(left_y, right_y);
        int _rows = ceil((heights[level] - _y)  / (nW/2));
        if (_rows > nRows[level]) {_rows = nRows[level];}
        for (int r=0;r<_rows;r++) {
            for (int c=0;c<nCols[level];c++) {
                vStat[level][nRows[level]-r-1][c]++;
            }
        }
    }
    else if ((left_y > heights[level] && right_y < heights[level]) ||
             (left_y < heights[level] && right_y > heights[level])) {
        if (left_y < right_y) {
            int _rows = ceil((heights[level] - left_y) / (nW/2));
            int _cols = ceil(down_x / (nW/2));   // 采用向下取整，防止超出
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][nRows[level]-r-1][c]++;
                }
            }
        }
        else {
            int _rows = ceil((heights[level] - right_y) / (nW/2));
            int _cols = ceil((widths[level] - down_x) / (nW/2));
            if (_rows > nRows[level]) {_rows = nRows[level];}
            if (_cols > nCols[level]) {_cols = nCols[level];}
            for (int r=0;r<_rows;r++) {
                for (int c=0;c<_cols;c++) {
                    vStat[level][nRows[level]-r-1][nCols[level]-c-1]++;
                }
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

    // if (nNumber != 0) {
    //     for (int r=0;r<vStat_out[level].size();r++) {
    //         for (int c=0;c<vStat_out[level][r].size();c++) {
    //             vStat_out[level][r][c] += vStat_pre[level][r][c];
    //         }
    //     }
    // }

    // vStat_pre[level].clear();
    // vStat_pre[level].resize(vStat[level].size());
    // for (int i=0;i<vStat[level].size();i++) {
    //     vStat_pre[level][i].assign(vStat[level][i].begin(), vStat[level][i].end());
    // }
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

    // 偏差中心点
    Point2f _center = cvPoint2D32f(pfCenter[0].x*50+500, pfCenter[0].y*50+500);
    circle(img, _center, 3, Scalar(0, 0, 255), 2, 4, 0);

    // 方差域
    circle(img, _center, fVariance[0]*50, Scalar(255, 0, 255), 2, 4, 0);

    // 偏差域
    circle(img, cvPoint(500, 500), fDistance[0]*50, Scalar(0, 0, 255), 2, 4, 0);

    imwrite(filename, img);
}

}   //namespace ORB_SLAM
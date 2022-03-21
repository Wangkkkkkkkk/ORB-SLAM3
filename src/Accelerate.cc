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

    vPoints.resize(nlevels);
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

    last_GF_number = 1;
    last_nProjectNumber = 1;
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

    if (GF_number < 100) {
        for (int r=0;r<nRows[level];r++) {
            for (int i=0;i<nCols[level];i++) {
                vStat[level][r][i] = 1;
            }
        }
        density[level] = 1;
        return vStat[level];
    }

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

    // cout<< "vStat:" << endl;
    // for (int r=0;r<nRows[level];r++) {
    //     for (int i=0;i<nCols[level];i++) {
    //         if (vStat[level][r][i] > 0) {
    //             cout<< "1 ";
    //         }
    //         else {
    //             cout<< "0 ";
    //         }
    //     }
    //     cout<<endl;
    // }
    return vStat[level];
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

    vPoints[0].clear();
    vGFpoints[0].clear();
    vpDis[0].clear();

    vector<int> Point_index;
    int nAllProject = 0;
    nProjectNumber = 0;
    d = 0;

#ifdef ACCELERATE_TIME
    double compute_GF = 0;
    double compute_GF_H = 0;
    double compute_GF_RW = 0;
    double compute_project = 0;
#endif

    for(int i=0; i<mLastFrame->N; i++)
    {
        if(mLastFrame->mvpMapPoints[i])
        {
            if(!mLastFrame->mvbOutlier[i])
            {
                // 提前将优特征点投影到预测下一帧的图像上，下一帧位姿采用恒速模型
                Mat _point_world = mLastFrame->mvpMapPoints[i]->GetWorldPos();

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_StartComputeProject1 = chrono::steady_clock::now();
#endif

                Mat _predict_camera = Rcw_last * _point_world + tcw_last;
                Point2f _predict_uv = project(_predict_camera);

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_EndComputeProject1 = chrono::steady_clock::now();
                double mTimeComputeProject1 = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeProject1 - time_StartComputeProject1).count();
                compute_project += mTimeComputeProject1;
#endif

                if (_predict_uv.x < 0 || _predict_uv.y < 0 || _predict_uv.x > nImages_width || _predict_uv.y > nImages_height) {
                    continue;
                }

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_StartComputeProject2 = chrono::steady_clock::now();
#endif

                Mat x3Dc = Rcw * _point_world + tcw;
                d += x3Dc.at<float>(2, 0);
                Point2f uv = project(x3Dc);

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_EndComputeProject2 = chrono::steady_clock::now();
                double mTimeComputeProject2 = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeProject2 - time_StartComputeProject2).count();
                compute_project += mTimeComputeProject2;
#endif

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_StartComputeGF = chrono::steady_clock::now();
#endif
                // 优特征点判断
                arma::mat H_meas, H_proj;
                H_meas = compute_H_subblock_simplied(_point_world, Rcw_last, tcw_last, H_proj);
                // cout<< "H_meas:" <<endl<< H_meas <<endl;
                
#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_EndComputeGF_H = chrono::steady_clock::now();
                double mTimeComputeGF_H = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeGF_H - time_StartComputeGF).count();
                compute_GF_H += mTimeComputeGF_H;
#endif

                float res_u = mLastFrame->mvKeys[i].pt.x - uv.x,
                      res_v = mLastFrame->mvKeys[i].pt.y - uv.y;  // res_u, res_v 是投影误差
                // cout<< "res_u:" << res_u << " res_v:" << res_v <<endl;

                arma::mat H_rw;
                ReWeightInfoMat(mLastFrame, i, mLastFrame->mvpMapPoints[i],
                                H_meas, res_u, res_v,
                                H_proj, H_rw);
                // cout<< "curMat:" <<endl<< curMat <<endl;

                arma::mat curMat = H_rw.t() * H_rw;
                double curDet = _logDet(curMat);
                // cout<< "curDet:" << curDet <<endl;

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_EndComputeGF = chrono::steady_clock::now();
                double mTimeComputeGF = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeGF - time_StartComputeGF).count();
                compute_GF += mTimeComputeGF;

                double mTimeComputeGF_RW = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeGF - time_EndComputeGF_H).count();
                compute_GF_RW += mTimeComputeGF_RW;
#endif

                pair<double, int> p(curDet, nProjectNumber);
                curDet_que.push(p);
                Point_index.push_back(i);

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_StartComputeProject3 = chrono::steady_clock::now();
#endif

                Mat x3Dc_pre = Rcw_prelast * _point_world + tcw_prelast;
                Point2f uv_pre = project(x3Dc_pre);

#ifdef ACCELERATE_TIME
                chrono::steady_clock::time_point time_EndComputeProject3 = chrono::steady_clock::now();
                double mTimeComputeProject3 = chrono::duration_cast<chrono::duration<double, milli> >(time_EndComputeProject3 - time_StartComputeProject3).count();
                compute_project += mTimeComputeProject3;
#endif

                vPoints[0].push_back(_predict_uv);
                vpDis[0].push_back(uv - uv_pre);

                nProjectNumber++;
            }
        }
    }

#ifdef ACCELERATE_TIME
    cout<< "compute GF:" << compute_GF <<endl;
    cout<< "  compute GF H:" << compute_GF_H <<endl;
    cout<< "  compute GF RW:" << compute_GF_RW <<endl;
    cout<< "compute project:" << compute_project <<endl;
#endif

    last_nProjectNumber = nProjectNumber;

    d = d / nProjectNumber;

    // cout<< "curDet_que:" <<endl;
    pair<double, int> x = curDet_que.top();
    double max_logdet = x.first;
    double logdet_th1 = max_logdet * 1.2;
    double logdet_th2 = max_logdet * 1.5;
    curDet_que.pop();

    GF_number = 1;
    vector<pair<double, int> > _curDet;
    for (int i=1;i<nProjectNumber;i++) {
        pair<double, int> _x = curDet_que.top();
        // cout<< "curdet:" << x.first << " index:" << x.second <<endl;

        if (_x.first > logdet_th1) {
            vGFpoints[0].push_back(vPoints[0][_x.second]);
            GF_number++;
        }
        else {
            _curDet.push_back(_x);
        }
        curDet_que.pop();
    }
    
    if (GF_number < 100) {
        for (int i=0;i<_curDet.size();i++) {
            if (_curDet[i].first > logdet_th2) {
                vGFpoints[0].push_back(vPoints[0][_curDet[i].second]);
                GF_number++;
            }
            if (GF_number > 150) {
                break;
            }
        }
    }
    last_GF_number = GF_number;
    // cout<< "GF_number:" << GF_number <<endl;

    if (nNumber > 0) {
       computeHomography();
    }
}

arma::mat Accelerate::compute_H_subblock_simplied(Mat point_world, Mat Rcw, Mat t, arma::mat & dhu_dhrl) {
    arma::mat H13, H47, H_meas;
    
    dhu_dhrl = { {mvParameters[0]/(point_world.at<float>(2,0)), 0.0, -point_world.at<float>(0,0)*mvParameters[0]/( pow(point_world.at<float>(2,0), 2.0))},
                 {0.0, mvParameters[1]/(point_world.at<float>(2,0)), -point_world.at<float>(1,0)*mvParameters[1]/( pow(point_world.at<float>(2,0), 2.0))} };  // 雅可比矩阵
    // cout<< "dhu_dhrl:" <<endl<< dhu_dhrl <<endl;
    arma::mat dqbar_by_dq;
    arma::colvec v1;
    v1 << 1.0 << -1.0 << -1.0 << -1.0 << arma::endr;
    dqbar_by_dq = arma::diagmat(v1);

    // arma::mat dhu_dhrl = { {4.3610e+2, 0, -1.1766e+02},
    //                        {0, 4.3481e+02, -2.0851e+02} };
    // arma::rowvec q_wr = {0.9986, 0.0178, 0.0463, -0.0157};
    // arma::mat R_rw = { {0.9952, -0.0298, -0.0931},
    //                    {0.0331, 0.9989, 0.0341},
    //                    {0.0920, -0.0370, 0.9951} };
    // arma::rowvec t_rw = {0.3958, 0.4564, 1.0373};

    arma::mat R_rw = { {Rcw.at<float>(0, 0), Rcw.at<float>(0, 1), Rcw.at<float>(0, 2)},
                       {Rcw.at<float>(1, 0), Rcw.at<float>(1, 1), Rcw.at<float>(1, 2)},
                       {Rcw.at<float>(2, 0), Rcw.at<float>(2, 1), Rcw.at<float>(2, 2)} };
    arma::mat R_wr = arma::inv(R_rw);
    arma::rowvec q_wr = r2q(R_wr);

    arma::rowvec _t = {t.at<float>(0, 0), t.at<float>(1, 0), t.at<float>(2, 0)};
    arma::rowvec t_rw = (- R_wr * _t.t()).t();

    arma::rowvec qwr_conj = qreset(q_wr);

    H13 = -1.0 * (dhu_dhrl *  R_rw);

    H47 = dhu_dhrl * (dRq_times_by_dq( qwr_conj ,  t_rw) * dqbar_by_dq);

    H_meas = arma::join_horiz(H13, H47);  // 按照水平方向连接两个矩阵
    return H_meas;
}

arma::rowvec Accelerate::r2q(arma::mat R) {
    double trace = R(0, 0) + R(1, 1) + R(2, 2);
    arma::rowvec Q = {0, 0, 0, 0};
	if (trace > 0.0)
	{
		double s = sqrt(trace + 1.0);
		Q[3] = (s * 0.5);
		s = 0.5 / s;

		//主要区别在此，即减数与被减数顺序
		Q[0] = (((R(2, 1) - R(1, 2))) * s);
		Q[1] = (((R(0, 2) - R(2, 0))) * s);
		Q[2] = (((R(1, 0) - R(0, 1))) * s);
	}
	else
	{
		int i = R(0, 0) < R(1, 1) ? (R(1, 1) < R(2, 2) ? 2 : 1) : (R(0, 0) < R(2, 2) ? 2 : 0);
		int j = (i + 1) % 3;
		int k = (i + 2) % 3;

		double s = sqrt(R(i, i) - R(j, j) - R(k, k) + 1.0);
		Q[i] = s * 0.5;
		s = 0.5 / s;

		Q[3] = ((R(k, j) - R(j, k))) * s;
		Q[j] = ((R(j, i) + R(i, j))) * s;
		Q[k] = ((R(k, i) + R(i, k))) * s;
	}
    double x = Q[0];
    double y = Q[1];
    double z = Q[2];
    double r = Q[3];
    Q[0] = r;
    Q[1] = x;
    Q[2] = y;
    Q[3] = z;
    return Q;
}

arma::rowvec Accelerate::qreset(arma::rowvec q) {
    q = -1.0 * q;
    q[0] = -q[0];
    return q;
}

arma::mat Accelerate::dRq_times_by_dq(arma::rowvec & q,
                          arma::rowvec & aMat) {
    double q0 = q[0];
    double qx = q[1];
    double qy = q[2];
    double qz = q[3];

    arma::mat dR_by_dq0(3,3), dR_by_dqx(3,3), dR_by_dqy(3,3), dR_by_dqz(3,3);
    dR_by_dq0 = { {2.0*q0, -2.0*qz, 2.0*qy},
                  {2.0*qz, 2.0*q0, -2.0*qx},
                  {-2.0*qy, 2.0*qx, 2.0*q0} };

    dR_by_dqx = { {2.0*qx, 2.0*qy, 2.0*qz},
                  {2.0*qy, -2.0*qx, -2.0*q0},
                  {2.0*qz, 2.0*q0, -2.0*qx} };

    dR_by_dqy = { {-2.0*qy, 2.0*qx, 2.0*q0},
                  {2.0*qx, 2.0*qy, 2.0*qz},
                  {-2.0*q0, 2.0*qz, -2.0*qy} };

    dR_by_dqz = { {-2.0*qz, -2.0*q0, 2.0*qx},
                  {2.0*q0, -2.0*qz, 2.0*qy},
                  {2.0*qx, 2.0*qy, 2.0*qz} };

    arma::mat RES = arma::zeros<arma::mat>(3,4);
    RES(arma::span(0,2), arma::span(0,0)) = dR_by_dq0 * aMat.t();
    RES(arma::span(0,2), arma::span(1,1)) = dR_by_dqx * aMat.t();
    RES(arma::span(0,2), arma::span(2,2)) = dR_by_dqy * aMat.t();
    RES(arma::span(0,2), arma::span(3,3)) = dR_by_dqz * aMat.t();

    return RES;
}

double Accelerate::_logDet(arma::mat M) {
    arma::cx_double ld_ = arma::log_det(M);
    return ld_.real();
}

void Accelerate::ReWeightInfoMat(Frame * F, int & kptIdx, MapPoint * pMP,
                     arma::mat & H_meas, float & res_u, float & res_v,
                     arma::mat & H_proj, arma::mat & H_rw) {
    int measSz = H_meas.n_rows;
    arma::mat Sigma_r(measSz, measSz), W_r(measSz, measSz);
    Sigma_r.eye();
    
    float Sigma2 = F->mvLevelSigma2[F->mvKeys[kptIdx].octave];
    Sigma_r = Sigma_r * Sigma2;
    // cout<< "Sigma_r:" <<endl<< Sigma_r <<endl;

    // double stdMapErr = -exp(double(pMP->mnVisible - 1)) * 0.01;
    double stdMapErr = exp(min(fabs(res_u) + fabs(res_v), float(10.0)));  // 地图点投影误差
    // cout<< "stdMapErr:" << stdMapErr <<endl;
    // cout<< "H_proj:" <<endl<< H_proj <<endl;
    Sigma_r = Sigma_r + H_proj * H_proj.t() * pow(stdMapErr, 2.0);
    // cout<< "Sigma_r:" <<endl<< Sigma_r <<endl;

    if (arma::chol(W_r, Sigma_r, "lower") == true) {
        // scale the meas. Jacobian with the scaling block W_r
        // cout<< "W_r:" <<endl<< W_r <<endl;
        H_rw = arma::inv(W_r) * H_meas;
    }
    else {
        // cout<< "reweight chol fail" <<endl;
    }
}

void Accelerate::compute_Huber_Weight (float residual_, float & weight_) {
    if (fabs(residual_) > 0.001) {
        float loss_;
        compute_Huber_Loss(residual_, loss_);
        weight_ = sqrt( loss_ ) / residual_;
    }
    else {
        weight_ = 1.0;
    }
}

void Accelerate::compute_Huber_Loss (float residual_, float & loss_) {
    float delta_ = sqrt(5.991);

    if (fabs(residual_) < delta_) {
        loss_ = pow(residual_, 2);
    }
    else {
        loss_ = 2 * delta_ * fabs(residual_) - pow(delta_, 2);
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
    Image_leftdown.at<float>(1, 0) = nImages_height-16;
    Image_leftdown.at<float>(2, 0) = 1;

    Mat Image_righttop = _N.rowRange(0, 3).col(0).clone();
    Image_righttop.at<float>(0, 0) = nImages_width-16;
    Image_righttop.at<float>(1, 0) = 16;
    Image_righttop.at<float>(2, 0) = 1;

    Mat Image_rightdown = _N.rowRange(0, 3).col(0).clone();
    Image_rightdown.at<float>(0, 0) = nImages_width-16;
    Image_rightdown.at<float>(1, 0) = nImages_height-16;
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
    // string filename_real = "/home/kai/file/VO_SpeedUp/Dataset/image_project/real_" + numb + ".png";
    // imwrite(filename, mProject);
    // imwrite(filename_real, mImages);
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

        fDistance[level] = sqrt(pow(pfCenter[level].x, 2) + pow(pfCenter[level].y, 2)) + 2 * fVariance[level];
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
    // cout<< "level:" << level << " density:" << density[level] <<endl;
    // if (density[level] < 0.15) {
    //     for (int r=0;r<nRows[level];r++) {
    //         for (int i=0;i<nCols[level];i++) {
    //             vStat[level][r][i] = 1;
    //         }
    //     }
    // }
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


// save project points info and extractor points info
void Accelerate::saveExtractor(vector<vector<KeyPoint> > allkeypoints) {
    Mat _images;
    cvtColor(mImages, _images, COLOR_GRAY2BGR);
    string numb = to_string(nNumberAll);
    string filename = "/home/kai/file/VO_SpeedUp/Dataset/feature_extractor/" + numb + ".png";

    // 投影特征点
    // for (int i=0;i<vPoints[0].size();i++) {
    //     int x = vPoints[0][i].x;
    //     int y = vPoints[0][i].y;
    //     circle(_images, cvPoint(x,y), 4, Scalar(0, 255, 0), 2, 4, 0);  // 绿色
    // }

    // 优特征点
    for (int i=0;i<vGFpoints[0].size();i++) {
        int x = vGFpoints[0][i].x;
        int y = vGFpoints[0][i].y;
        circle(_images, cvPoint(x,y), 4, Scalar(0, 0, 255), 2, 4, 0);  // 红色
    }

    // 提取特征点
    // for (int j=0;j<allkeypoints[0].size();j++) {
    //     int x = allkeypoints[0][j].pt.x;
    //     int y = allkeypoints[0][j].pt.y;
    //     circle(_images, cvPoint(x,y), 6, Scalar(0, 255, 255), 2, 4, 0);  // 黄色
    // }

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
            if (vStat[0][i][j] > 0) {
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
    // putText(_images, to_string(dens) + "%", cvPoint(0, 470), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(0, 255, 0), 1);
    // putText(_images, "All number:" + to_string(nProjectNumber), cvPoint(105, 475), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(0, 255, 0), 1);  // 总数量
    // putText(_images, "GF number:" + to_string(GF_number), cvPoint(305, 475), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(0, 255, 0), 1);  // 优特征点数量
    // putText(_images, "noGF number:" + to_string(nProjectNumber-GF_number), cvPoint(505, 475), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(0, 255, 0), 1);  // 非优特征点数量
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
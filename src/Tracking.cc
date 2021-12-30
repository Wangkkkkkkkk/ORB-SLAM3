/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2020 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Initializer.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include <assert.h>
#include <iostream>

#include <mutex>
#include <chrono>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <include/MLPnPsolver.h>


using namespace std;

// 程序中变量名的第一个字母如果为"m"则表示为类中的成员变量，member
// 第一个、第二个字母:
// "p"表示指针数据类型
// "n"表示int类型
// "b"表示bool类型
// "s"表示set类型
// "v"表示vector数据类型
// 'l'表示list数据类型
// "KF"表示KeyFrame数据类型   
namespace ORB_SLAM3
{


Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0), time_recently_lost_visual(2.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), num_good_constr_predef(200)
{
    // Load camera parameters from settings file
    // Step 1 从配置文件中加载相机参数
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    bool b_parse_cam = ParseCamParamFile(fSettings);
    if(!b_parse_cam)
    {
        std::cout << "*Error with the camera parameters in the config file*" << std::endl;
    }

    // Load ORB parameters
    bool b_parse_orb = ParseORBParamFile(fSettings);
    if(!b_parse_orb)
    {
        std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
    }

    initID = 0; lastID = 0;

    // Load IMU parameters
    bool b_parse_imu = true;
    if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO)
    {
        b_parse_imu = ParseIMUParamFile(fSettings);
        if(!b_parse_imu)
        {
            std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
        }

        mnFramesToResetIMU = mMaxFrames;
    }

    mbInitWith3KFs = false;

    mnNumDataset = 0;

    if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
    {
        std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
        try
        {
            throw -1;
        }
        catch(exception &e)
        {

        }
    }

    // Good Feature Match
    mCurrentInfoMat.set_size(7, 7);
    mCurrentInfoMat.zeros();

    mFrameAfterInital = 0;
    camera_fps = 20;

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    int nRows= fSettings["Camera2.nRows"];
    int nCols= fSettings["Camera2.nCols"];

    mObsHandler = new Observability(fx, fy, nRows, nCols, cx, cy, 0, 0, sensor);

    mvpQueriedFeatures.resize(NUM_TOTAL_HASHTABLES);
    std::fill(mvpQueriedFeatures.begin(), mvpQueriedFeatures.begin() + NUM_ACTIVE_HASHTABLES, NUM_ADDITIONAL_FEATURES + NUM_CANDIDATE_FEATURES);
    std::fill(mvpQueriedFeatures.begin() + NUM_ACTIVE_HASHTABLES, mvpQueriedFeatures.end(), 0);

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();

    vdUpdatedLM_ms.clear();
    vdSearchLP_ms.clear();
    vdPoseOpt_ms.clear();
#endif

    vnKeyFramesLM.clear();
    vnMapPointsLM.clear();
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBA_ms[i] << ","
          << mpLocalMapper->vdKFCulling_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }


    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#KF insert[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdStereoMatch_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();

    f.open("TrackLocalMapStats.txt");
    f << fixed << setprecision(6);

    f << "# number of KF, number of MP, UpdateLM[ms], SearchLP[ms], PoseOpt[ms]" << endl;

    for(int i=0; i<vnKeyFramesLM.size(); ++i)
    {

        f << vnKeyFramesLM[i] << "," << vnMapPointsLM[i] <<  "," << vdUpdatedLM_ms[i] << "," << vdSearchLP_ms[i] << "," << vdPoseOpt_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();


    ofstream f;
    f.open("ExecTimeMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Map Tracking complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Local Map Tracking complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Local Map Tracking complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(vnKeyFramesLM);
    deviation = calcDeviation(vnKeyFramesLM, average);
    std::cout << "Local KFs: " << average << "$\\pm$" << deviation << std::endl;
    f << "Local KFs: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vnMapPointsLM);
    deviation = calcDeviation(vnMapPointsLM, average);
    std::cout << "Local MPs: " << average << "$\\pm$" << deviation << std::endl;
    f << "Local MPs: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBASync_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBASync_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCullingSync_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCullingSync_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllMaps()[0]->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMaps()[0]->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    f << "KFs in map: " << mpAtlas->GetAllMaps()[0]->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << mpAtlas->GetAllMaps()[0]->GetAllMapPoints().size() << std::endl;


    // Place recognition time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Place Recognition (mean$\\pm$std)" << std::endl << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl << std::endl;

    average = calcAverage(mpLoopClosing->vTimeBoW_ms);
    deviation = calcDeviation(mpLoopClosing->vTimeBoW_ms, average);
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vTimeSE3_ms);
    deviation = calcDeviation(mpLoopClosing->vTimeSE3_ms, average);
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vTimePRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vTimePRTotal_ms, average);
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl;

    // Loop Closing time stats
    if(mpLoopClosing->vTimeLoopTotal_ms.size() > 0)
    {
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "Loop Closing (mean$\\pm$std)" << std::endl << std::endl;
        f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl << std::endl;

        average = calcAverage(mpLoopClosing->vTimeLoopTotal_ms);
        deviation = calcDeviation(mpLoopClosing->vTimeLoopTotal_ms, average);
        std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl;
        f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(mpLoopClosing->vTimeMergeTotal_ms.size() > 0)
    {
        // Map Merging time stats
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "Map Merging (mean$\\pm$std)" << std::endl << std::endl;
        f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl << std::endl;

        average = calcAverage(mpLoopClosing->vTimeMergeTotal_ms);
        deviation = calcDeviation(mpLoopClosing->vTimeMergeTotal_ms, average);
        std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl;
        f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl;
    }


    if(mpLoopClosing->vTimeGBATotal_ms.size() > 0)
    {
        // Full GBA time stats
        std::cout << std::endl << std::endl << std::endl;
        std::cout << "Full GBA (mean$\\pm$std)" << std::endl << std::endl;
        f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl << std::endl;

        average = calcAverage(mpLoopClosing->vTimeFullGBA_ms);
        deviation = calcDeviation(mpLoopClosing->vTimeFullGBA_ms, average);
        std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
        f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vTimeMapUpdate_ms);
        deviation = calcDeviation(mpLoopClosing->vTimeMapUpdate_ms, average);
        std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
        f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
        average = calcAverage(mpLoopClosing->vTimeGBATotal_ms);
        deviation = calcDeviation(mpLoopClosing->vTimeGBATotal_ms, average);
        std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl;
        f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl;
    }

    f.close();

}

#endif

Tracking::~Tracking()
{

}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpAtlas->AddCamera(mpCamera);


        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;


        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        if(!b_miss_params)
        {
            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);

            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }


            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            if(!node.empty())
            {
                mTlr = node.mat();
                if(mTlr.rows != 3 || mTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << mTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

        mpAtlas->AddCamera(mpCamera);
        mpAtlas->AddCamera(mpCamera2);
    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }


    }

    if(mSensor==System::RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;
    int nFeatures, nLevels, fIniThFAST, fMinThFAST;
    float fScaleFactor;
	// 每一帧提取的特征点数
    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }
	// 图像建立金字塔时的变化尺度 1.2
    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }
	// 尺度金字塔的层数 8
    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }
	// 提取fast特征点的默认阈值 20
    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }
	// 如果默认阈值提取不出足够fast特征点，则使用最小阈值 8
    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    // 在单目初始化的时候，会用mpIniORBextractor来作为特征点提取器，初始化使用5倍的特征点数目
    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat Tbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        Tbc = node.mat();
        if(Tbc.rows != 4 || Tbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }

    cout << endl;

    cout << "Left camera to Imu Transform (Tbc): " << endl << Tbc << endl;

    float freq, Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        freq = node.operator int();
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(freq);
    cout << endl;
    cout << "IMU frequency: " << freq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    // 噪声从连续到离散差了一个sqrt(freq)   而随机游走从连续到离散差了一个 1/sqrt(freq)
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);


    return true;
}

//设置局部建图器
void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

//设置回环检测器
void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

//设置可视化查看器
void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}


// 输入左右目图像，可以为RGB、BGR、RGBA、GRAY
// 1、将图像转为mImGray和imGrayRight并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    // step 1 ：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    // 这里考虑得十分周全,甚至连四通道的图像都考虑到了
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(
        mImGray,                //左目图像
        imGrayRight,            //右目图像
        timestamp,              //时间戳
        mpORBextractorLeft,     //左目特征提取器
        mpORBextractorRight,    //右目特征提取器
        mpORBVocabulary,        //字典
        mK,                     //内参矩阵
        mDistCoef,              //去畸变参数
        mbf,                    //基线长度
		mThDepth,				//远点,近点的区分阈值
		mpCamera);				//相机模型
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif
	// Step 3 ：跟踪
    Track();

    //返回位姿
    return mCurrentFrame.mTcw.clone();
}

// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    // step 1：将RGB或RGBA图像转为灰度图像
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    // Step 2 ：将深度相机的disparity转为Depth , 也就是转换成为真正尺度下的深度

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);
	// Step 3：构造Frame
    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

	// Step 4：跟踪
    Track();

	//返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}

/**
 * @brief 
 * 输入左目RGB或RGBA图像，输出世界坐标系到该帧相机坐标系的变换矩阵
 * 
 * @param[in] im 单目图像
 * @param[in] timestamp 时间戳
 * @return cv::Mat 
 * 
 * Step 1 ：将彩色图像转为灰度图像
 * Step 2 ：构造Frame
 * Step 3 ：跟踪
 */
cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    mImGray = im;

    // Step 1 ：将彩色图像转为灰度图像
    //若图片是3、4通道的彩色图，还需要转化成单通道灰度图
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    // Step 2 ：构造Frame类
    if (mSensor == System::MONOCULAR)
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth, &mLastFrame);
    }
    else if(mSensor == System::IMU_MONOCULAR)
    {
        //判断该帧是不是初始化
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)  //没有成功初始化的前一个状态就是NO_IMAGES_YET
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }

    // t0存储未初始化时的第1帧图像时间戳
    if (mState==NO_IMAGES_YET)
        t0=timestamp;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    lastID = mCurrentFrame.mnId;

    // Step 3 ：跟踪
    Track();

    //返回当前帧的位姿
    return mCurrentFrame.mTcw.clone();
}


/**
 * @brief 将imu数据存放在mlQueueImuData的list链表里
 * 
 * @param[in] imuMeasurement 
 */
void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

// ?TODO
void Tracking::PreintegrateIMU()
{
    // Step 1.拿到两两帧之间待处理的预积分数据，组成一个集合 
    // cout << "start preintegration" << endl;
    // 上一帧不存在,说明两帧之间没有imu数据，不进行预积分
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        // 当前帧是否进行预积分标志
        mCurrentFrame.setIntegrated();
        return;
    }

    // cout << "start loop. Total meas:" << mlQueueImuData.size() << endl;

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    // 没有imu数据,不进行预积分
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        // 数据还没有时,会等待一段时间,直到mlQueueImuData中有imu数据.一开始不需要等待
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                // 拿到第一个imu数据作为起始数据
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                // imu起始数据会比当前帧的前一帧时间戳早,如果相差0.001则舍弃这个imu数据
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-0.001l)
                {
                    mlQueueImuData.pop_front();
                }
                // 同样最后一个的imu数据时间戳也不能理当前帧时间间隔多余0.001 
                // ? (时间戳本身的差值,单位是s秒,tum数据集里面是19位时间戳,单位为ns,这里可能只保留了10位整数和9位小数)
                else if(m->t<mCurrentFrame.mTimeStamp-0.001l)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    // 得到两帧间的imu数据放入mvImuFromLastFrame中,得到后面预积分的处理数据
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    // Step 2.对两帧之间进行中值积分处理
    // m个imu组数据会有m-1个预积分量
    const int n = mvImuFromLastFrame.size()-1;
    // 构造imu预处理器,并初始化标定数据
    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);
    // 针对预积分位置的不同做不同中值积分的处理
    /**
     *  根据上面imu帧的筛选，IMU与图像帧的时序如下：
     *  Frame---IMU0---IMU1---IMU2---IMU3---IMU4---------------IMUx---Frame---IMUx+1
     *  T_------T0-----T1-----T2-----T3-----T4-----------------Tx-----_T------Tx+1
     *  A_------A0-----A1-----A2-----A3-----A4-----------------Ax-----_T------Ax+1
     *  W_------W0-----W1-----W2-----W3-----W4-----------------Wx-----_T------Wx+1
     *  T_和_T分别表示上一图像帧和当前图像帧的时间戳，A(加速度数据)，W(陀螺仪数据)，同理
     */
    for(int i=0; i<n; i++)
    {
        float tstep;
        cv::Point3f acc, angVel;
        // 第一帧数据但不是最后两帧,imu总帧数大于2
        if((i==0) && (i<(n-1)))
        {
            // 时间差作为积分量
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            // 根据上面分析的IMU与图像帧的时序分析，获取当前imu到上一帧的时间间隔
            // 在加加速度不变的假设上算出第一帧图像到第二个imu数据的中值积分
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            // ? 这里采用离散中值积分进行预积分,获取当前imu到上一帧的时间间隔
            // 设当前时刻imu的加速度a0，下一时刻加速度a1，时间间隔tab 为t10，tini t0p
            // 正常情况下时为了求上一帧到当前时刻imu的一个平均加速度，但是imu时间不会正好落在上一帧图像的时刻，需要做补偿，要求得a0时刻到上一帧这段时间加速度的改变量
            // 有了这个改变量将其加到a0上之后就可以表示上一帧时的加速度了。其中a0 - (a1-a0)*(tini/tab) 为上一帧时刻的加速度再加上a1 之后除以2就为这段时间的加速度平均值
            // 其中tstep表示a1到上一帧的时间间隔，a0 - (a1-a0)*(tini/tab)这个式子中tini可以是正也可以是负表示时间上的先后，(a1-a0)也是一样，多种情况下这个式子依然成立
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            // 上一帧图像到下一帧imu（也就是第二个imu数据）的时间差
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        //中间帧直接求中值
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        //最后一帧,但不是第一帧
        else if((i>0) && (i==(n-1)))
        {
            // 直到倒数第二个imu时刻时，计算过程跟第一时刻类似，都需要考虑帧与imu时刻的关系
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        //只有两帧的情况
        else if((i==0) && (i==(n-1)))
        {
            // ? 就两个数据时使用第一个时刻的，之前size减一了，最后一帧imu好像没遍历，这时候应该还有一帧在后面，但这里直接没有算中值积分？
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        // Step 3.依次进行预积分计算
        // 应该是必存在的吧，一个是相对上一关键帧，一个是相对上一帧
        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    // 记录当前预积分的图像帧
    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    // ? 设置为已预积分状态
    mCurrentFrame.setIntegrated();

    Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

// ?TODO
/**
 * @brief 跟踪不成功的时候，用初始化好的imu数据做跟踪处理，通过IMU预测状态
 * 两个地方用到：
 * 1. 匀速模型计算速度,但并没有给当前帧位姿赋值；
 * 2. 跟踪丢失时不直接判定丢失，通过这个函数预测当前帧位姿看看能不能拽回来，代替纯视觉中的重定位
 * 
 * @return true 
 * @return false 
 */
bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }
    // 总结下都在什么时候地图更新，也就是mbMapUpdated为true
    // 1. 回环或融合
    // 2. 局部地图LocalBundleAdjustment
    // 3. IMU三阶段的初始化
    // 下面的代码流程一模一样，只不过计算时相对的帧不同，地图有更新后相对于上一关键帧做的，反之相对于上一帧
    // 地图更新后会更新关键帧与MP，所以相对于关键帧更准
    // 而没更新的话，距离上一帧更近，计算起来误差更小
    // 地图更新时，并且上一个图像关键帧存在
    if(mbMapUpdated && mpLastKeyFrame)
    {   
        const cv::Mat twb1 = mpLastKeyFrame->GetImuPosition();
        const cv::Mat Rwb1 = mpLastKeyFrame->GetImuRotation();
        const cv::Mat Vwb1 = mpLastKeyFrame->GetVelocity();

        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        // ? mpImuPreintegratedFromLastKF是上一帧关键帧？
        const float t12 = mpImuPreintegratedFromLastKF->dT;
        // 计算当前帧在世界坐标系的位姿,原理都是用预积分的位姿（预积分的值不会变化）与上一帧的位姿（会迭代变化）进行更新 
        // 旋转 R_wb2 = R_wb1 * R_b1b2
        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        // 位移 公式可见Forster论文公式32，只是要变化一下，邱笑晨的那篇文档有详细推导
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        // 速度 公式可见Forster论文公式32，只是要变化一下，邱笑晨的那篇文档有详细推导
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        // 设置当前帧的世界坐标系的相机位姿
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        // ? 预测当前帧imu在世界坐标系的位姿，以及记录偏置bias和bias更新
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    // 地图未更新时
    else if(!mbMapUpdated)
    {
        const cv::Mat twb1 = mLastFrame.GetImuPosition();
        const cv::Mat Rwb1 = mLastFrame.GetImuRotation();
        const cv::Mat Vwb1 = mLastFrame.mVw;
        const cv::Mat Gz = (cv::Mat_<float>(3,1) << 0,0,-IMU::GRAVITY_VALUE);
        // mpImuPreintegratedFrame是当前帧上一帧，不一定是关键帧
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        cv::Mat Rwb2 = IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        cv::Mat twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        cv::Mat Vwb2 = Vwb1 + t12*Gz + Rwb1*mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);
        mCurrentFrame.mPredRwb = Rwb2.clone();
        mCurrentFrame.mPredtwb = twb2.clone();
        mCurrentFrame.mPredVwb = Vwb2.clone();
        mCurrentFrame.mImuBias =mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}


void Tracking::ComputeGyroBias(const vector<Frame*> &vpFs, float &bwx,  float &bwy, float &bwz)
{
    const int N = vpFs.size();
    vector<float> vbx;
    vbx.reserve(N);
    vector<float> vby;
    vby.reserve(N);
    vector<float> vbz;
    vbz.reserve(N);

    cv::Mat H = cv::Mat::zeros(3,3,CV_32F);
    cv::Mat grad  = cv::Mat::zeros(3,1,CV_32F);
    for(int i=1;i<N;i++)
    {
        Frame* pF2 = vpFs[i];
        Frame* pF1 = vpFs[i-1];
        cv::Mat VisionR = pF1->GetImuRotation().t()*pF2->GetImuRotation();
        cv::Mat JRg = pF2->mpImuPreintegratedFrame->JRg;
        cv::Mat E = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaRotation().t()*VisionR;
        cv::Mat e = IMU::LogSO3(E);
        assert(fabs(pF2->mTimeStamp-pF1->mTimeStamp-pF2->mpImuPreintegratedFrame->dT)<0.01);

        cv::Mat J = -IMU::InverseRightJacobianSO3(e)*E.t()*JRg;
        grad += J.t()*e;
        H += J.t()*J;
    }

    cv::Mat bg = -H.inv(cv::DECOMP_SVD)*grad;
    bwx = bg.at<float>(0);
    bwy = bg.at<float>(1);
    bwz = bg.at<float>(2);

    for(int i=1;i<N;i++)
    {
        Frame* pF = vpFs[i];
        pF->mImuBias.bwx = bwx;
        pF->mImuBias.bwy = bwy;
        pF->mImuBias.bwz = bwz;
        pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        pF->mpImuPreintegratedFrame->Reintegrate();
    }
}

void Tracking::ComputeVelocitiesAccBias(const vector<Frame*> &vpFs, float &bax,  float &bay, float &baz)
{
    const int N = vpFs.size();
    const int nVar = 3*N +3; // 3 velocities/frame + acc bias
    const int nEqs = 6*(N-1);

    cv::Mat J(nEqs,nVar,CV_32F,cv::Scalar(0));
    cv::Mat e(nEqs,1,CV_32F,cv::Scalar(0));
    cv::Mat g = (cv::Mat_<float>(3,1)<<0,0,-IMU::GRAVITY_VALUE);

    for(int i=0;i<N-1;i++)
    {
        Frame* pF2 = vpFs[i+1];
        Frame* pF1 = vpFs[i];
        cv::Mat twb1 = pF1->GetImuPosition();
        cv::Mat twb2 = pF2->GetImuPosition();
        cv::Mat Rwb1 = pF1->GetImuRotation();
        cv::Mat dP12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaPosition();
        cv::Mat dV12 = pF2->mpImuPreintegratedFrame->GetUpdatedDeltaVelocity();
        cv::Mat JP12 = pF2->mpImuPreintegratedFrame->JPa;
        cv::Mat JV12 = pF2->mpImuPreintegratedFrame->JVa;
        float t12 = pF2->mpImuPreintegratedFrame->dT;
        // Position p2=p1+v1*t+0.5*g*t^2+R1*dP12
        J.rowRange(6*i,6*i+3).colRange(3*i,3*i+3) += cv::Mat::eye(3,3,CV_32F)*t12;
        J.rowRange(6*i,6*i+3).colRange(3*N,3*N+3) += Rwb1*JP12;
        e.rowRange(6*i,6*i+3) = twb2-twb1-0.5f*g*t12*t12-Rwb1*dP12;
        // Velocity v2=v1+g*t+R1*dV12
        J.rowRange(6*i+3,6*i+6).colRange(3*i,3*i+3) += -cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*(i+1),3*(i+1)+3) += cv::Mat::eye(3,3,CV_32F);
        J.rowRange(6*i+3,6*i+6).colRange(3*N,3*N+3) -= Rwb1*JV12;
        e.rowRange(6*i+3,6*i+6) = g*t12+Rwb1*dV12;
    }

    cv::Mat H = J.t()*J;
    cv::Mat B = J.t()*e;
    cv::Mat x(nVar,1,CV_32F);
    cv::solve(H,B,x);

    bax = x.at<float>(3*N);
    bay = x.at<float>(3*N+1);
    baz = x.at<float>(3*N+2);

    for(int i=0;i<N;i++)
    {
        Frame* pF = vpFs[i];
        x.rowRange(3*i,3*i+3).copyTo(pF->mVw);
        if(i>0)
        {
            pF->mImuBias.bax = bax;
            pF->mImuBias.bay = bay;
            pF->mImuBias.baz = baz;
            pF->mpImuPreintegratedFrame->SetNewBias(pF->mImuBias);
        }
    }
}


/**
 * @brief 跟踪过程，包括恒速模型跟踪、参考关键帧跟踪、局部地图跟踪
 * track包含两部分：估计运动、跟踪局部地图
 * 
 * Step 1：初始化
 * Step 2：跟踪
 * Step 3：记录位姿信息，用于轨迹复现
 */
void Tracking::Track()
{

    if (bStepByStep)
    {
        while(!mbStep)
            usleep(500);
        mbStep = false;
    }

    // Step 1 如局部建图里认为IMU有问题，重置当前活跃地图
    if(mpLocalMapper->mbBadImu)
    {
        cout << "TRACK: Reset map because local mapper set the bad imu flag " << endl;
        mpSystem->ResetActiveMap();
        return;
    }

    // 从Atlas中取出当前active的地图
    Map* pCurrentMap = mpAtlas->GetCurrentMap();

    // Step 2 处理时间戳异常的情况
    if(mState!=NO_IMAGES_YET)
    {
        // 进入以下两个if语句都是不正常的情况，不进行跟踪直接返回
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            // 如果当前图像时间戳比前一帧图像时间戳小，说明出错了，清除imu数据，创建新的子地图
            cerr << "ERROR: Frame with a timestamp older than previous frame detected!" << endl;
            unique_lock<mutex> lock(mMutexImuQueue);
            mlQueueImuData.clear();
            // 创建新地图
            CreateMapInAtlas();
            return;
        }
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // 如果当前图像时间戳和前一帧图像时间戳大于1s，说明时间戳明显跳变了，重置地图后直接返回
            cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            //根据是否是imu模式,进行imu的补偿
            if(mpAtlas->isInertial())
            {
                // 如果当前地图imu成功初始化
                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    // ?TODO BA2标志代表什么?,BA优化成功?
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        // 如果当前子图中imu没有经过BA2，重置active地图
                        mpSystem->ResetActiveMap();
                    }
                    else
                    {
                        // 如果当前子图中imu进行了BA2，重新创建新的子图
                        CreateMapInAtlas();
                    }
                }
                else
                {
                    // 如果当前子图中imu还没有初始化，重置active地图
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap();
                }
            }
            // 不跟踪直接返回
            return;
        }
    }

    // Step 3 IMU模式下设置IMU的Bias参数,还要保证上一帧存在非空
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && mpLastKeyFrame)
        //认为bias在两帧间不变
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;
    // Step 4 IMU模式下对IMU数据进行预积分：没有创建地图的情况下
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        // IMU数据进行预积分
        PreintegrateIMU();
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu);
#endif


    }
    mbCreatedMap = false;

    // Get Map Mutex -> Map cannot be changed
    // 地图更新时加锁。保证地图不会发生变化
    // 疑问:这样子会不会影响地图的实时更新?
    // 回答：主要耗时在构造帧中特征点的提取和匹配部分,在那个时候地图是没有被上锁的,有足够的时间更新地图
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false;

    // 判断地图id是否更新了
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        // cout << "Map update detected" << endl;
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex);
        mbMapUpdated = true;
    }


    if(mState==NOT_INITIALIZED)
    {   
        // Step 5 初始化
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO)
        {
            //双目RGBD相机的初始化共用一个函数
            StereoInitialization();
        }
        else
        {
            //单目初始化
            MonocularInitialization();
        }

        mpFrameDrawer->Update(this);

        if(mState!=OK) // If rightly initialized, mState=OK
        {
            // 如果没有成功初始化，直接返回
            mLastFrame = Frame(mCurrentFrame);
            return;
        }

        if(mpAtlas->GetAllMaps().size() == 1)
        {
            // 如果当前地图是第一个地图，记录当前帧id为第一帧
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // System is initialized. Track Frame.
        // Step 6 系统成功初始化，下面是具体跟踪过程
        bool bOK;

        mFrameAfterInital++;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
        // tracking 类构造时默认为false。在viewer中有个开关ActivateLocalizationMode，可以控制是否开启mbOnlyTracking

        if(!mbOnlyTracking)
        {

            // State OK
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            // 跟踪进入正常SLAM模式，有地图更新

            // 如果正常跟踪
            if(mState==OK)
            {

                // Local Mapping might have changed some MapPoints tracked in last frame
                // Step 6.1 检查并更新上一帧被替换的MapPoints
                // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();


                // Step 6.2 运动模型是空的并且imu未初始化或刚完成重定位，跟踪参考关键帧；否则恒速模型跟踪
                // 第一个条件,如果运动模型为空并且imu未初始化,说明是刚开始第一帧跟踪，或者已经跟丢了。
                // 第二个条件,如果当前帧紧紧地跟着在重定位的帧的后面，我们用重定位帧来恢复位姿
                // mnLastRelocFrameId 上一次重定位的那一帧
                if((mVelocity.empty() && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    //Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    //Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    // 用恒速模型跟踪。所谓的恒速就是假设上上帧到上一帧的位姿=上一帧的位姿到当前帧位姿
                    // 根据恒速模型设定当前帧的初始位姿，用最近的普通帧来跟踪当前的普通帧
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点，优化每个特征点所对应3D点的投影误差即可得到位姿
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        //根据恒速模型失败了，只能根据参考关键帧来跟踪
                        bOK = TrackReferenceKeyFrame();
                }

                // 新增了一个状态RECENTLY_LOST，主要是结合IMU看看能不能拽回来
                // Step 6.3 如果经过跟踪参考关键帧、恒速模型跟踪都失败的话，并满足一定条件就要标记为RECENTLY_LOST或LOST
                if (!bOK)
                {
                    // 条件1：如果当前帧距离上次重定位成功不到1s
                    //        mnFramesToResetIMU 表示经过多少帧后可以重置IMU，一般设置为和帧率相同，对应的时间是1s
                    // 条件2：单目+IMU 或者 双目+IMU模式
                    // 同时满足条件1，2，标记为LOST
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
                    {
                        mState = LOST;
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)   
                    {
                        // 条件1：当前地图中关键帧数目较多（大于10） 
                        // 条件2（隐藏条件）：当前帧距离上次重定位帧超过1s（说明还比较争气，值的救）或者非IMU模式
                        // 同时满足条件1，2，则将状态标记为RECENTLY_LOST，后面会结合IMU预测的位姿看看能不能拽回来
                        cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST;
                        // 记录丢失时间
                        mTimeStampLost = mCurrentFrame.mTimeStamp;
                        //mCurrentFrame.SetPose(mLastFrame.mTcw);
                    }
                    else
                    {
                        mState = LOST;
                    }
                }
            }
            else    //跟踪不正常按照下面处理
            {
                if (mState == RECENTLY_LOST)
                {
                    // 如果是RECENTLY_LOST状态
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);
                    // bOK先置为true
                    bOK = true;

                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)) 
                    {
                        // IMU模式下可以用IMU来预测位姿，看能否拽回来
                        // Step 6.4 如果当前地图中IMU已经成功初始化，就用IMU数据预测位姿
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU();
                        else
                            bOK = false;

                        // 如果IMU模式下当前帧距离跟丢帧超过5s还没有找回（time_recently_lost默认为5s）
                        // 放弃了，将RECENTLY_LOST状态改为LOST
                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else    
                    {
                        // Step 6.5 纯视觉模式则进行重定位。主要是BOW搜索，EPnP求解位姿
                        // TODO fix relocalization
                        bOK = Relocalization();
                        if(!bOK)
                        {
                            // 纯视觉模式下重定位失败，状态为LOST
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)  // 上一帧为最近丢失且重定位失败时
                {
                    // Step 6.6 如果是LOST状态
                    // 开启一个新地图
                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        // 当前地图中关键帧数目小于10，重置当前地图
                        mpSystem->ResetActiveMap();
                        cout << "Reseting current map..." << endl;
                    }else
                        // 当前地图中关键帧数目超过10，创建新地图
                        CreateMapInAtlas();
                    // 干掉上一个关键帧
                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return;
                }
            }

        }
        else  // 纯定位模式
        {
            // Localization Mode: Local Mapping is deactivated (TODO Not available in inertial mode)
            // 只进行跟踪tracking，局部地图不工作
            if(mState==LOST)
            {
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                // Step 6.1 LOST状态进行重定位    
                bOK = Relocalization();
            }
            else
            {
                // mbVO是mbOnlyTracking为true时的才有的一个变量
                // mbVO为false表示此帧匹配了很多的MapPoints，跟踪很正常 (注意有点反直觉)
                // mbVO为true表明此帧匹配了很少的MapPoints，少于10个，要跟丢
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map
                    // Step 6.2 如果跟踪状态正常，使用恒速模型或参考关键帧跟踪
                    if(!mVelocity.empty())
                    {
                        // 优先使用恒速模型跟踪
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        // 如果恒速模型不被满足,那么就只能够通过参考关键帧来跟踪
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    // mbVO为true，表明此帧匹配了很少（小于10）的地图点，要跟丢的节奏，既做跟踪又做重定位

                    // MM=Motion Model,通过运动模型进行跟踪的结果
                    bool bOKMM = false;
                    // 通过重定位方法来跟踪的结果
                    bool bOKReloc = false;
                    // 运动模型中构造的地图点
                    vector<MapPoint*> vpMPsMM;
                    // 在追踪运动模型后发现的外点
                    vector<bool> vbOutMM;
                    // 运动模型得到的位姿
                    cv::Mat TcwMM;
                    // Step 6.3 当运动模型有效的时候,根据运动模型计算位姿
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        // 将恒速模型跟踪结果暂存到这几个变量中，因为后面重定位会改变这些变量
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    // Step 6.4 使用重定位的方法来得到当前帧的位姿
                    bOKReloc = Relocalization();

                    // Step 6.5 根据前面的恒速模型、重定位结果来更新状态
                    if(bOKMM && !bOKReloc)
                    {
                        // 恒速模型成功、重定位失败，重新使用之前暂存的恒速模型结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        //? 疑似bug！这段代码是不是重复增加了观测次数？后面 TrackLocalMap 函数中会有这些操作
                        // 如果当前帧匹配的3D点很少，增加当前可视地图点的被观测次数
                        if(mbVO)
                        {
                            // 更新当前帧的地图点被观测次数
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                // 如果这个特征点形成了地图点,并且也不是外点的时候
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    // 增加能观测到该地图点的帧数
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        // 只要重定位成功整个跟踪过程正常进行（重定位与跟踪，更相信重定位）
                        mbVO = false;
                    }
                    // 有一个成功我们就认为执行成功了
                    bOK = bOKReloc || bOKMM;
                }
            }
        }
        // 将最新的关键帧作为当前帧的参考关键帧
        // mpReferenceKF先是上一时刻的参考关键帧，如果当前为新关键帧则变成当前关键帧，如果不是新的关键帧则先为上一帧的参考关键帧，而后经过更新局部关键帧重新确定
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred);
#endif


#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif

        // Step 7 在跟踪得到当前帧初始姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // 前面只是跟踪一帧得到初始位姿，这里搜索局部关键帧、局部地图点，和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        // 在帧间匹配得到初始的姿态后，现在对local map进行跟踪得到更多的匹配，并优化当前位姿
        // local map:当前帧、当前帧的MapPoints、当前关键帧与其它关键帧共视关系
        // 前面主要是两两跟踪（恒速模型跟踪上一帧、跟踪参考帧），这里搜索局部关键帧后搜集所有局部MapPoints，
        // 然后将局部MapPoints和当前帧进行投影匹配，得到更多匹配的MapPoints后进行Pose优化
        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
            {
                // 局部地图跟踪
                bOK = TrackLocalMapByGF();
                // bOK = TrackLocalMap();

            }
            if(!bOK)
                cout << "Fail to track local map!" << endl;
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMapByGF();
                // bOK = TrackLocalMap();
        }
        // 到此为止跟踪确定位姿阶段结束，下面开始做收尾工作和为下一帧做准备

        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true                     -->OK   1 跟踪局部地图成功
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true                       -->OK  2 重定位
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        //
        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK                  -->OK  1 跟踪局部地图成功
        //            \               \              \---局部地图跟踪失败---OK                  -->OK  3 正常跟踪
        //             \               \---当前帧跟踪失败---非OK
        //              \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---非OK
        //                            \           \---局部地图跟踪失败---非OK
        //                             \---重定位失败---非OK（传不到这里，因为直接return了）
        // 由上图可知当前帧的状态OK的条件是跟踪局部地图成功，重定位或正常跟踪都可
        // Step 8 根据上面的操作来判断是否追踪成功
        if(bOK)
            // 此时还OK才说明跟踪成功了
            mState = OK;
        else if (mState == OK) // 由上图可知只有当第一阶段跟踪成功，但第二阶段局部地图跟踪失败时执行
        {
            // 带有IMU时状态变为最近丢失，否则直接丢失
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            {
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    // IMU模式下IMU没有成功初始化或者没有完成IMU BA，则重置当前地图
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST;
            }
            else
                mState=LOST; // visual to lost

            // 如果当前帧距离上次重定位帧超过1s，用当前帧时间戳更新lost帧时间戳
            if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            }
        }

        // Save frame if recent relocalization, since they are used for IMU reset (as we are making copy, it shluld be once mCurrFrame is completely modified)
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) &&          // 当前帧距离上次重定位帧小于1s
        (mCurrentFrame.mnId > mnFramesToResetIMU) &&                                // 当前帧已经运行了超过1s
        ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) &&  // IMU模式
        pCurrentMap->isImuInitialized())                                            // IMU已经成功初始化
        {
            // ?TODO check this situation
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame);
            pF->mpPrevFrame = new Frame(mLastFrame);

            // Load preintegration
            // ?构造预积分处理器
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        if(pCurrentMap->isImuInitialized())     // IMU成功初始化
        {
            if(bOK)     // 跟踪成功
            {
                // 当前帧距离上次重定位帧刚好等于1s，重置（还未实现 TODO）
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    cout << "RESETING FRAME!!!" << endl;
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    mLastBias = mCurrentFrame.mImuBias;  // 没啥用，后面会重新赋值后传给普通帧
            }
        }

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack);
#endif

        // Update drawer
        // 更新显示线程中的图像、特征点、地图点等信息
        mpFrameDrawer->Update(this);
        if(!mCurrentFrame.mTcw.empty())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
        // 查看到此为止时的两个状态变化
        // bOK的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---true
        //          \               \              \---局部地图跟踪失败---false
        //           \               \---当前帧跟踪失败---false
        //            \---上一帧跟踪失败---重定位成功---局部地图跟踪成功---true
        //                          \           \---局部地图跟踪失败---false
        //                           \---重定位失败---false

        // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
        //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
        //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
        //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
        //               \                           \           \---局部地图跟踪失败---LOST
        //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
        //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）
        // Step 9 如果跟踪成功 或 最近刚刚跟丢，更新速度，清除无效地图点，按需创建关键帧
        if(bOK || mState==RECENTLY_LOST)
        {
            // Update motion model
            // Step 9.1 更新恒速运动模型 TrackWithMotionModel 中的mVelocity
            if(!mLastFrame.mTcw.empty() && !mCurrentFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));    // CameraCenter / mOw = - R的转置 * t;
                // cout<< "lastFrame RotationInverse = " << endl << mLastFrame.GetRotationInverse() <<endl;
                // cout<< "lastFrame GetCameraCenter = " << endl << mLastFrame.GetCameraCenter() <<endl;
                // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                // cout<< "LastTwc = " << endl << LastTwc <<endl;
                // cout<< "mCurrentFrame.mTcw = " << endl << mCurrentFrame.mTcw <<endl;
                mVelocity = mCurrentFrame.mTcw * LastTwc;
                mCurrentFrame.mPredictTcw = mVelocity * mCurrentFrame.mTcw;
                mCurrentFrame.mPredictTcw_last = mLastFrame.mPredictTcw.clone();
                // cout<< "mVelocity = " << endl << mVelocity <<endl;
            }
            else
                // 否则速度为空
                mVelocity = cv::Mat();
            // 使用IMU积分的位姿显示
            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            // Step 9.2 清除观测不到的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            // Step 9.3 清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
            // 上个步骤中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
            // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            // 这里不仅仅是清除mlpTemporalPoints，通过delete pMP还删除了指针指向的MapPoint
            // 不能够直接执行这个是因为其中存储的都是指针,之前的操作都是为了避免内存泄露
            mlpTemporalPoints.clear();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            // 判断是否需要插入关键帧
            bool bNeedKF = NeedNewKeyFrame();

            // Check if we need to insert a new keyframe
            // Step 9.4 根据条件来判断是否插入关键帧
            // 需要同时满足下面条件1和2
            // 条件1：bNeedKF=true，需要插入关键帧
            // 条件2：bOK=true跟踪成功 或 IMU模式下的RECENTLY_LOST模式
            if(bNeedKF && (bOK|| (mState==RECENTLY_LOST && (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO))))
                // 创建关键帧，对于双目或RGB-D会产生新的地图点
                CreateNewKeyFrame();

#ifdef REGISTER_TIMES
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF);
#endif

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame. Only has effect if lastframe is tracked
            // 作者这里说允许在BA中被Huber核函数判断为外点的传入新的关键帧中，让后续的BA来审判他们是不是真正的外点
            // 但是估计下一帧位姿的时候我们不想用这些外点，所以删掉

            //  Step 9.5 删除那些在BA中检测为外点的地图点            
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                // 这里第一个条件还要执行判断是因为, 前面的操作中可能删除了其中的地图点
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        // Step 10 如果第二阶段跟踪失败，跟踪状态为LOST
        if(mState==LOST)
        {
            // 如果地图中关键帧小于5，重置当前地图，退出当前跟踪
            if(pCurrentMap->KeyFramesInMap()<=5)
            {
                mpSystem->ResetActiveMap();
                return;
            }
            if ((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO))
                if (!pCurrentMap->isImuInitialized())
                {
                    // 如果是IMU模式并且还未进行IMU初始化，重置当前地图，退出当前跟踪
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }
            // 如果地图中关键帧超过5 并且 纯视觉模式 或 虽然是IMU模式但是已经完成IMU初始化了，创建新的地图
            CreateMapInAtlas();
        }
        //确保已经设置了参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        // 保存上一帧的数据,当前帧变上一帧
        mLastFrame = Frame(mCurrentFrame);
    }
    // 查看到此为止
    // mState的历史变化---上一帧跟踪成功---当前帧跟踪成功---局部地图跟踪成功---OK
    //            \               \              \---局部地图跟踪失败---非OK（IMU时为RECENTLY_LOST）
    //             \               \---当前帧跟踪失败---非OK(地图超过10个关键帧时 RECENTLY_LOST)
    //              \---上一帧跟踪失败(RECENTLY_LOST)---重定位成功---局部地图跟踪成功---OK
    //               \                           \           \---局部地图跟踪失败---LOST
    //                \                           \---重定位失败---LOST（传不到这里，因为直接return了）
    //                 \--上一帧跟踪失败(LOST)--LOST（传不到这里，因为直接return了）
    // last.记录位姿信息，用于轨迹复现
    // Step 11 记录位姿信息，用于最后保存所有的轨迹
    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
    	// Step 11：记录位姿信息，用于最后保存所有的轨迹
        if(!mCurrentFrame.mTcw.empty())
        {
            // 计算相对姿态Tcr = Tcw * Twr, Twr = Trw^-1
            cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost
            // 如果跟踪失败，则相对位姿使用上一次值
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);
        }

    }
}
/*
 * @brief 双目和rgbd的地图初始化，比单目简单很多
 *
 * 由于具有深度信息，直接生成MapPoints
 */
void Tracking::StereoInitialization()
{
    // 初始化要求当前帧的特征点超过500
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (cv::norm(mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA)<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO)
        {
            cv::Mat Rwb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).colRange(0,3).clone();
            cv::Mat twb0 = mCurrentFrame.mImuCalib.Tcb.rowRange(0,3).col(3).clone();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, cv::Mat::zeros(3,1,CV_32F));
        }
        else
            mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F)); // 设定初始位姿为单位旋转，0平移

        // Create KeyFrame
        // 将当前帧构造为初始关键帧
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        // 在地图中添加该初始关键帧
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
			// 为每个特征点构造MapPoint
            for(int i=0; i<mCurrentFrame.N;i++)
            {
            //只有具有正深度的点才会被构造地图点
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                	// 通过反投影得到该特征点的世界坐标系下3D坐标
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                	// 将3D点构造为MapPoint
                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());
	                // 为该MapPoint添加属性：
	                // a.观测到该MapPoint的关键帧
	                // b.该MapPoint的描述子
	                // c.该MapPoint的平均观测方向和深度范围
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    cv::Mat x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        // 在局部地图中添加该初始关键帧
        mpLocalMapper->InsertKeyFrame(pKFini);

        // 更新当前帧为上一帧
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        // 把当前（最新的）局部MapPoints作为ReferenceMapPoints
        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        //追踪成功
        mState=OK;
    }
}

/*
 * @brief 单目的地图初始化
 *
 * 并行地计算基础矩阵和单应性矩阵，选取其中一个模型，恢复出最开始两帧之间的相对姿态以及点云
 * 得到初始两帧的匹配、相对运动、初始MapPoints
 * 
 * Step 1：（未创建）得到用于初始化的第一帧，初始化需要两帧
 * Step 2：（已创建）如果当前帧特征点数大于100，则得到用于单目初始化的第二帧
 * Step 3：在mInitialFrame与mCurrentFrame中找匹配的特征点对
 * Step 4：如果初始化的两帧之间的匹配点太少，重新初始化
 * Step 5：通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
 * Step 6：删除那些无法进行三角化的匹配点
 * Step 7：将三角化得到的3D点包装成MapPoints
 */
void Tracking::MonocularInitialization()
{
    // Step 1 如果单目初始器还没有被创建，则创建。后面如果重新初始化时会清掉这个
    if(!mpInitializer)
    {
        // Set Reference Frame
        // 单目初始帧的特征点数必须大于100
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // 初始化需要两帧，分别是mInitialFrame，mCurrentFrame
            mInitialFrame = Frame(mCurrentFrame);
            // 用当前帧更新上一帧
            mLastFrame = Frame(mCurrentFrame);
            // mvbPrevMatched  记录"上一帧"所有特征点
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            // 删除前判断一下，来避免出现段错误。不过在这里是多余的判断
            // 不过在这里是多余的判断，因为前面已经判断过了
            if(mpInitializer)
                delete mpInitializer;

            // 由当前帧构造初始器 sigma:1.0 iterations:200
            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            // 初始化为-1 表示没有任何匹配。这里面存储的是匹配的点的id
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            if (mSensor == System::IMU_MONOCULAR)
            {
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;

            }
            return;
        }
    }
    else    //如果单目初始化器已经被创建
    {
		// Step 2 如果当前帧特征点数太少（不超过100），则重新构造初始器
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }

        // Find correspondences
        // Step 3 在mInitialFrame与mCurrentFrame中找匹配的特征点对
        ORBmatcher matcher(
            0.9,        //最佳的和次佳特征点评分的比值阈值，这里是比较宽松的，跟踪时一般是0.7
            true);      //检查特征点的方向

        // 对 mInitialFrame,mCurrentFrame 进行特征点匹配
        // mvbPrevMatched为参考帧的特征点坐标，初始化存储的是mInitialFrame中特征点坐标，匹配后存储的是匹配好的当前帧的特征点坐标
        // mvIniMatches 保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
        int nmatches = matcher.SearchForInitialization(
            mInitialFrame,mCurrentFrame,    //初始化时的参考帧和当前帧
            mvbPrevMatched,                 //在初始化参考帧中提取得到的特征点
            mvIniMatches,                   //保存匹配关系
            100);                           //搜索窗口大小

        // Check if there are enough correspondences
        // Step 4 验证匹配结果，如果初始化的两帧之间的匹配点太少，重新初始化
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)
		// Step 5 通过H模型或F模型进行单目初始化，得到两帧间相对运动、初始MapPoints
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,  //初始帧的特征点
		mCurrentFrame.mvKeysUn,		//当前帧的特征点
		mvIniMatches,				//当前帧和参考帧的特征点的匹配关系
		Rcw,tcw,					//初始化得到的相机的位姿
		mvIniP3D,					//进行三角化得到的空间点集合
		vbTriangulated))			//以及对应于mvIniMatches来讲,其中哪些点被三角化了
        {
            // Step 6 初始化成功后，删除那些无法进行三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            // Step 7 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            // 由Rcw和tcw构造Tcw,并赋值给mTcw，mTcw为世界坐标系到相机坐标系的变换矩阵
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            // Step 8 创建初始化地图点MapPoints
            // Initialize函数会得到mvIniP3D，
            // mvIniP3D是cv::Point3f类型的一个容器，是个存放3D点的临时变量，
            // CreateInitialMapMonocular将3D点包装成MapPoint类型存入KeyFrame和Map中
            CreateInitialMapMonocular();

        }
    }
}

/**
 * @brief 单目相机成功初始化后用三角化得到的点生成MapPoints
 * 
 */
void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames 认为单目初始化时候的参考帧和当前帧都是关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);

	// Step 1 将初始关键帧,当前关键帧的描述子转为BoW
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    // Step 2 将关键帧插入到地图
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    // Step 3 用初始化得到的3D点来生成地图点MapPoints
    //  mvIniMatches[i] 表示初始化两帧特征点匹配关系。
    //  具体解释：i表示帧1中关键点的索引值，vMatches12[i]的值为帧2的关键点索引值,没有匹配关系的话，vMatches12[i]值为 -1
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        // 没有匹配，跳过
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        // 用三角化点初始化为空间点的世界坐标
        cv::Mat worldPos(mvIniP3D[i]);
		// Step 3.1 用3D点构造地图点
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        // Step 3.2 为该MapPoint添加属性：
        // a.观测到该MapPoint的关键帧
        // b.该MapPoint的描述子
        // c.该MapPoint的平均观测方向和深度范围

        // 表示该KeyFrame的2D特征点和对应的3D地图点
        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        // a.表示该MapPoint可以被哪个KeyFrame的哪个特征点观测到
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // b.从众多观测到该MapPoint的特征点中挑选最有代表性的描述子
        pMP->ComputeDistinctiveDescriptors();
        // c.更新该MapPoint平均观测方向以及观测距离的范围
        pMP->UpdateNormalAndDepth();

        // 地图点加入响应值信息
        pMP->response = pKFcur->mvKeys[mvIniMatches[i]].response;

        //Fill Current Frame structure
        //mvIniMatches下标i表示在初始化参考帧中的特征点的序号
        //mvIniMatches[i]是初始化当前帧中的特征点的序号
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpAtlas->AddMapPoint(pMP);
    }


    // Update Connections
    // Step 3.3 更新关键帧间的连接关系
    // 在3D点和关键帧之间建立边，每个边有一个权重，边的权重是该关键帧与当前帧公共3D点的个数
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();

    // Bundle Adjustment
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    // Step 4 全局BA优化，同时优化所有位姿和三维点
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);

    pKFcur->PrintPointDistribution();
    // Step 5 取场景的中值深度，用于尺度归一化 
    // 为什么是 pKFini 而不是 pKCur ? 答：都可以的，内部做了位姿变换了
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // 4.0f
    else
        invMedianDepth = 1.0f/medianDepth;

    //两个条件,一个是平均深度要大于0,另外一个是在当前帧中被观测到的地图点的数目应该大于50
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // TODO Check, originally 100 tracks
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_NORMAL);
        mpSystem->ResetActiveMap();
        return;
    }

    // Step 6 将两帧之间的变换归一化到平均深度1的尺度下
    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    // x/z y/z 将z归一化到1 
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    // Step 7 把3D点的尺度也归一化到1
    // 为什么是pKFini? 是不是就算是使用 pKFcur 得到的结果也是相同的? 答：是的，因为是同样的三维点
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
            pMP->UpdateNormalAndDepth();
        }
    }

    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;
        pKFini->mNextKF = pKFcur;
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;

        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }

	//  Step 8 将关键帧插入局部地图，更新归一化后的位姿、局部地图点
    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;
    mnLastRelocFrameId = mInitialFrame.mnId;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    // 单目初始化之后，得到的初始地图中的所有点都是局部地图点
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    // Compute here initial velocity
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();

    cv::Mat deltaT = vKFs.back()->GetPose()*vKFs.front()->GetPoseInverse();
    mVelocity = cv::Mat();
    Eigen::Vector3d phi = LogSO3(Converter::toMatrix3d(deltaT.rowRange(0,3).colRange(0,3)));


    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;

    mLastFrame = Frame(mCurrentFrame);

    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;// 初始化成功，至此，初始化过程完成

    initID = pKFcur->mnId;
}

/**
 * @brief 在Atlas中创建新地图
 * 
 */
void Tracking::CreateMapInAtlas()
{
    // ?TODO
    mnLastInitFrameId = mCurrentFrame.mnId;
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();
    mbSetInit=false;

    mnInitialFrameId = mCurrentFrame.mnId+1;
    mState = NO_IMAGES_YET;

    // Restart the variable with information about the last KF
    mVelocity = cv::Mat();
    mnLastRelocFrameId = mnLastInitFrameId; // The last relocation KF_id is the current id, because it is the new starting point for new map
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    mbVO = false; // Init value for know if there are enough MapPoints in the last KF
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        if(mpInitializer)
            delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO ) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    }

    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);

    mLastFrame = Frame();
    mCurrentFrame = Frame();
    mvIniMatches.clear();

    mbCreatedMap = true;

}
/*
 * @brief 检查上一帧中的地图点是否需要被替换
 * 
 * Local Mapping线程可能会将关键帧中某些地图点进行替换，由于tracking中需要用到上一帧地图点，所以这里检查并更新上一帧中被替换的地图点
 * @see LocalMapping::SearchInNeighbors()
 */
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        //如果这个地图点存在
        if(pMP)
        {
            // 获取其是否被替换,以及替换后的点
            // 这也是程序不直接删除这个地图点删除的原因
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                //然后替换一下
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

/*
 * @brief 用参考关键帧的地图点来对当前普通帧进行跟踪
 * 
 * Step 1：将当前普通帧的描述子转化为BoW向量
 * Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
 * Step 3: 将上一帧的位姿态作为当前帧位姿的初始值
 * Step 4: 通过优化3D-2D的重投影误差来获得位姿
 * Step 5：剔除优化后的匹配点中的外点
 * @return 如果匹配数超10，返回true
 * 
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    // Step 1：将当前帧的描述子转化为BoW向量
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
    int nmatches = matcher.SearchByBoW(
        mpReferenceKF,          //参考关键帧
        mCurrentFrame,          //当前帧
        vpMapPointMatches);     //存储匹配关系

    // 匹配数目小于15，认为跟踪失败
    if(nmatches<15)
    {
        cout << "TRACK_REF_KF: Less than 15 matches!!\n";
        return false;
    }

    // Step 3:将上一帧的位姿态作为当前帧位姿的初始值
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 用上一次的Tcw设置初值，在PoseOptimization可以收敛快一些

    //mCurrentFrame.PrintPointDistribution();


    // cout << " TrackReferenceKeyFrame mLastFrame.mTcw:  " << mLastFrame.mTcw << endl;
	// Step 4:通过优化3D-2D的重投影误差来获得位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除优化后的匹配点中的外点
    //之所以在优化之后才剔除外点，是因为在优化的过程中就有了对这些外点的标记
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            //如果对应到的某个特征点是外点
            if(mCurrentFrame.mvbOutlier[i])
            {
                //清除它在当前帧中存在过的痕迹
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                //匹配的内点计数++
                nmatchesMap++;
        }
    }

    // TODO check these conditions
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap>=10; // 跟踪成功的数目超过10才认为跟踪成功，否则跟踪失败
}

/**
 * @brief 更新上一帧位姿，在上一帧中生成临时地图点
 * 单目情况：只计算了上一帧的世界坐标系位姿
 * 双目和rgbd情况：选取有有深度值的并且没有被选为地图点的点生成新的临时地图点，提高跟踪鲁棒性
 */
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
    // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
    KeyFrame* pRef = mLastFrame.mpReferenceKF;  
    // ref_keyframe 到 lastframe的位姿变换
    cv::Mat Tlr = mlRelativeFramePoses.back();

    // 将上一帧的世界坐标系下的位姿计算出来
    // l:last, r:reference, w:world
    // Tlw = Tlr*Trw 
    mLastFrame.SetPose(Tlr*pRef->GetPose()); 

    // 如果上一帧为关键帧，或者单目/单目惯性，SLAM模式(?)的情况，则退出
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Step 2：对于双目或rgbd相机，为上一帧生成新的临时地图点
    // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    // 如果上一帧中没有有效深度的点,那么就直接退出
    if(vDepthIdx.empty())
        return;

    // 按照深度从小到大排序
    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    // Step 2.2：从中找出不是地图点的部分  
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
            // 反投影到世界坐标系中
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);

            // 加入上一帧的地图点中
            mLastFrame.mvpMapPoints[i]=pNewMP;

            // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            // 因为从近到远排序，记录其中不需要创建地图点的个数
            nPoints++;
        }

        // Step 2.4：如果地图点质量不好，停止创建地图点
        // 停止新增临时地图点必须同时满足以下条件：
        // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
        // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
        if(vDepthIdx[j].first>mThDepth && nPoints>100)
        {
            break;
        }
    }
}
/**
 * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
 * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
 * Step 2：根据上一帧特征点对应地图点进行投影匹配
 * Step 3：优化当前帧位姿
 * Step 4：剔除地图点中外点
 * @return 如果匹配数大于10，认为跟踪成功，返回true
 */
bool Tracking::TrackWithMotionModel()
{
    // 最小距离 < 0.9*次小距离 匹配成功，检查旋转
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
    UpdateLastFrame();

    // Step 2：根据IMU或者恒速模型得到当前帧的初始位姿。
	
    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
		// IMU完成初始化 并且 距离重定位挺久不需要重置IMU，用IMU来估计位姿
        // Predict ste with IMU if it is initialized and it doesnt need reset
        PredictStateIMU();
        return true;
    }
    else
    {
		// 根据之前估计的速度，用恒速模型得到当前帧的初始位姿。
        mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);
    }

    // 清空当前帧的地图点
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    // 设置特征匹配过程中的搜索半径
    int th;

    if(mSensor==System::STEREO)
        th=7;
    else
        th=15;

    // Step 3：用上一帧地图点进行投影匹配，如果匹配点不够，则扩大搜索半径再来一次
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // If few matches, uses a wider window search
    // 如果匹配点太少，则扩大搜索半径再来一次
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    // 如果还是不能够获得足够的匹配点,那么就认为跟踪失败
    if(nmatches<20)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
            return true;
        else
            return false;
    }

    // Optimize frame pose with all matches
    // Step 4：利用3D-2D投影关系，优化当前帧位姿
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    // Step 5：剔除地图点中外点
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                // 如果优化后判断某个地图点是外点，清除它的所有关系
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;
                }
                else{
                    pMP->mbTrackInViewR = false;
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                // 累加成功匹配到的地图点数目
                nmatchesMap++;
        }
    }

    if(mbOnlyTracking)
    {
        // 纯定位模式下：如果成功追踪的地图点非常少,那么这里的mbVO标志就会置位
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
        return true;
    else
        return nmatchesMap>=10;		// 匹配超过10个点就认为跟踪成功
}

/**
 * @brief 用局部地图进行跟踪，进一步优化位姿
 * 
 * 1. 更新局部地图，包括局部关键帧和关键点
 * 2. 对局部MapPoints进行投影匹配
 * 3. 根据匹配对估计当前帧的姿态
 * 4. 根据姿态剔除误匹配
 * @return true if success
 * 
 * Step 1：更新局部关键帧mvpLocalKeyFrames和局部地图点mvpLocalMapPoints 
 * Step 2：在局部地图中查找与当前帧匹配的MapPoints, 其实也就是对局部地图点进行跟踪
 * Step 3：更新局部所有MapPoints后对位姿再次优化
 * Step 4：更新当前帧的MapPoints被观测程度，并统计跟踪局部地图的效果
 * Step 5：决定是否跟踪成功
 */
bool Tracking::TrackLocalMap()
{

    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.
    mTrackedFr++;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartLMUpdate = std::chrono::steady_clock::now();
#endif
    // Update Local KeyFrames and Local Points
    // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMap();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartSearchLP = std::chrono::steady_clock::now();

    double timeUpdatedLM_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartSearchLP - time_StartLMUpdate).count();
    vdUpdatedLM_ms.push_back(timeUpdatedLM_ms);
#endif
	// Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
    SearchLocalPoints();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartPoseOpt = std::chrono::steady_clock::now();

    double timeSearchLP_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartPoseOpt - time_StartSearchLP).count();
    vdSearchLP_ms.push_back(timeSearchLP_ms);
#endif

    // TOO check outliers before PO
    // 查看内外点数目，调试用
    int aux1 = 0, aux2=0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    // 在这个函数之前，在 Relocalization、TrackReferenceKeyFrame、TrackWithMotionModel 中都有位姿优化
    // Step 3：前面新增了更多的匹配关系，BA优化得到更准确的位姿
    int inliers;
    if (!mpAtlas->isImuInitialized())
        Optimizer::PoseOptimization(&mCurrentFrame);
    else
    {
        // 初始化，重定位，重新开启一个地图都会使mnLastRelocFrameId变化
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            // 如果距离上次重定位时间比较近，积累的IMU数据较少，优化时暂不使用IMU数据
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else    // 如果积累的IMU数据量比较多，考虑使用IMU数据优化
        {
            // mbMapUpdated变化见Tracking::PredictStateIMU()
            // 未更新地图
            if(!mbMapUpdated)
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndPoseOpt = std::chrono::steady_clock::now();

    double timePoseOpt_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPoseOpt - time_StartPoseOpt).count();
    vdPoseOpt_ms.push_back(timePoseOpt_ms);
#endif

    vnKeyFramesLM.push_back(mvpLocalKeyFrames.size());
    vnMapPointsLM.push_back(mvpLocalMapPoints.size());
    // 查看内外点数目，调试用
    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    // Step 4：更新当前帧的地图点被观测程度，并统计跟踪局部地图后匹配数目
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 由于当前帧的地图点可以被当前帧观测到，其被观测统计量加1
            if(!mCurrentFrame.mvbOutlier[i])
            {
                // 找到该点的帧数mnFound 加 1
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                //查看当前是否是在纯定位过程
                if(!mbOnlyTracking)
                {
                    // 如果该地图点被相机观测数目nObs大于0，匹配内点计数+1
                    // nObs： 被观测到的相机数目，单目+1，双目或RGB-D则+2
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    // 记录当前帧跟踪到的地图点数目，用于统计跟踪效果
                    mnMatchesInliers++;
            }
            // 如果这个地图点是外点,并且当前相机输入还是双目的时候,就删除这个点
            // 原因分析：因为双目本身可以左右互匹配，删掉无所谓
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;
 	// Step 5：根据跟踪匹配数目及重定位情况决定是否跟踪成功
    // 如果最近刚刚发生了重定位,那么至少成功匹配50个点才认为是成功跟踪
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    // RECENTLY_LOST状态下，至少成功跟踪10个才算成功
    if((mnMatchesInliers>10)&&(mState==RECENTLY_LOST))
        return true;

    // IMU模式下至少成功跟踪15个才算成功
    if (mSensor == System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO)
    {
        if(mnMatchesInliers<15)
        {
            return false;
        }
        else
            return true;
    }
    else
    {
		//以上情况都不满足，只要跟踪的地图点大于30个就认为成功了
        if(mnMatchesInliers<30)
            return false;
        else
            return true;
    }
}
/**
 * @brief 判断当前帧是否需要插入关键帧
 * 
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
bool Tracking::NeedNewKeyFrame()
{
    if(((mSensor == System::IMU_MONOCULAR) || (mSensor == System::IMU_STEREO)) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if (mSensor == System::IMU_STEREO && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }
	// Step 1：纯VO模式下不插入关键帧
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    {
        return false;
    }

    // Return false if IMU is initialazing
    if (mpLocalMapper->IsInitializing())
        return false;
		
	// 获取当前地图中的关键帧数目	
    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mCurrentFrame.mnId是当前帧的ID
    // mnLastRelocFrameId是最近一次重定位帧的ID
    // mMaxFrames等于图像输入的帧率
    //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    // Step 4：得到参考关键帧跟踪到的地图点数量
    // UpdateLocalKeyFrames 函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧 

    // 地图点的最小观测次数
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    // Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
     int nNonTrackedClose = 0;  //双目或RGB-D中没有跟踪到的近点
    int nTrackedClose= 0;       //双目或RGB-D中成功跟踪的近点（三维点）

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            // 深度值在有效范围内
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
    }
    // 双目或RGBD情况下：跟踪到的地图点中近点太少 同时 没有跟踪到的三维点太多，可以插入关键帧了
    // 单目时，为false
    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Step 7：决策是否需要插入关键帧
    // Thresholds
    // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
    float thRefRatio = 0.75f;

    // 关键帧只有一帧，那么插入关键帧的阈值设置的低一点，插入频率较低
    if(nKFs<2)
        thRefRatio = 0.4f;

    //单目情况下插入关键帧的频率很高    
    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    // Step 7.2：很长时间没有插入关键帧，可以插入
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    // Step 7.3：满足插入关键帧的最小间隔并且localMapper处于空闲状态，可以插入
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle);
    //Condition 1c: tracking is weak
	// Step 7.4：在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO &&  //只考虑在双目，RGB-D的情况
	(mnMatchesInliers<nRefMatches*0.25 || 			//当前帧和地图点匹配的数目非常少
	bNeedToInsertClose) ;							//需要插入
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR))) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        // Step 7.6：local mapping空闲时可以直接插入，不空闲的时候要根据情况插入
        if(bLocalMappingIdle)
        {
            //可以插入关键帧
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                if(mpLocalMapper->KeyframesInQueue()<3)
                    //队列中的关键帧数目不是很多,可以插入
                    return true;
                else
                    //队列中缓冲的关键帧数目太多,暂时不能插入
                    return false;
            }
            else
                //对于单目情况,就直接无法插入关键帧了
                //? 为什么这里对单目情况的处理不一样?
                //回答：可能是单目关键帧相对比较密集
                return false;
        }
    }
    else
        //不满足上面的条件,自然不能插入关键帧
        return false;
}

/**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 * 
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
void Tracking::CreateNewKeyFrame()
{
	// 如果局部建图线程正在初始化或关闭了,就无法插入关键帧
    if(mpLocalMapper->IsInitializing())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    // Step 1：将当前帧构造成关键帧
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

    if(mpAtlas->isImuInitialized())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);
	// Step 2：将当前关键帧设置为当前帧的参考关键帧
    // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;
        mpLastKeyFrame->mNextKF = pKF;
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }
   	// 这段代码和 Tracking::UpdateLastFrame 中的那一部分代码功能相同
    // Step 3：对于双目或rgbd摄像头，为当前帧生成新的地图点；单目无操作
    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        // 根据Tcw计算mRcw、mtcw和mRwc、mOw
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO)
            maxPoint = 100;
		// Step 3.1：得到当前帧有深度值的特征点（不一定是地图点）
        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                // 第一个元素是深度,第二个元素是对应的特征点的id
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            // Step 3.2：按照深度从小到大排序
            sort(vDepthIdx.begin(),vDepthIdx.end());

            // Step 3.3：从中找出不是地图点的生成临时地图点 
            // 处理的近点的个数
            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                // 如果需要就新建地图点，这里的地图点不是临时的，是全局地图中新建地图点，用于跟踪
                if(bCreateNew)
                {
                    cv::Mat x3D;

                    if(mCurrentFrame.Nleft == -1){
                        x3D = mCurrentFrame.UnprojectStereo(i);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    // 这些添加属性的操作是每次创建MapPoint后都要做的
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    // 因为从近到远排序，记录其中不需要创建地图点的个数
                    nPoints++;
                }

                // Step 3.4：停止新建地图点必须同时满足以下条件：
                // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
                // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }

            Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);

        }
    }

    // Step 4：插入关键帧
    // 关键帧插入到列表 mlNewKeyFrames中，等待local mapping线程临幸
    mpLocalMapper->InsertKeyFrame(pKF);

    // 插入好了，允许局部建图停止
    mpLocalMapper->SetNotStop(false);

    // 当前帧成为新的关键帧，更新
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
    //cout  << "end creating new KF" << endl;
}


/**
 * @brief 用局部地图点进行投影匹配，得到更多的匹配关系
 * 注意：局部地图点中已经是当前帧地图点的不需要再投影，只需要将此外的并且在视野范围内的点和当前帧进行投影匹配
 */
void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // Step 1：遍历当前帧的地图点，标记这些地图点不参与之后的投影搜索匹配
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                // 更新能观测到该点的帧数加1(被当前帧观测了)
                pMP->IncreaseVisible();
                // 标记该点被当前帧观测到
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                // 标记该点在后面搜索匹配时不被投影，因为已经有匹配了
                pMP->mbTrackInView = false;
                pMP->mbTrackInViewR = false;
            }
        }
    }

    // 准备进行投影匹配的点的数目
    int nToMatch=0;

    // Project points in frame and check its visibility
    // Step 2：判断所有局部地图点中除当前帧地图点外的点，是否在当前帧视野范围内
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        // 已经被当前帧观测到的地图点肯定在视野范围内，跳过
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // 跳过坏点
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        // 判断地图点是否在在当前帧视野内
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
        	// 观测到该点的帧数加1
            pMP->IncreaseVisible();
            // 只有在视野范围内的地图点才参与之后的投影匹配
            nToMatch++;
        }
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    // Step 3：如果需要进行投影匹配的点的数目大于0，就进行投影匹配，增加更多的匹配关系
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)   //RGBD相机输入的时候,搜索的阈值会变得稍微大一些
            th=3;
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())
                th=2;
            else
                th=3;
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO))
        {
            th=10;
        }

        // If the camera has been relocalised recently, perform a coarser search
        // 如果不久前进行过重定位，那么进行一个更加宽泛的搜索，阈值需要增大
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;

        if(mState==LOST || mState==RECENTLY_LOST) // Lost for less than 1 second
            th=15;
		// 投影匹配得到更多的匹配关系
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

/**
 * @brief 更新LocalMap
 *
 * 局部地图包括： 
 * 1、K1个关键帧、K2个临近关键帧和参考关键帧
 * 2、由这些关键帧观测到的MapPoints
 */
void Tracking::UpdateLocalMap()
{
    // This is for visualization
    // 设置参考地图点用于绘图显示局部地图点（红色）
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    // 用共视图来更新局部关键帧和局部地图点
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

/*
 * @brief 更新局部关键点。先把局部地图清空，然后将局部关键帧的有效地图点添加到局部地图中
 */
void Tracking::UpdateLocalPoints()
{
    // Step 1：清空局部地图点
    mvpLocalMapPoints.clear();
    int count_pts = 0;
	// Step 2：遍历局部关键帧 mvpLocalKeyFrames
    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        // step 2：将局部关键帧的地图点添加到mvpLocalMapPoints
        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            // 用该地图点的成员变量mnTrackReferenceForFrame 记录当前帧的id
            // 表示它已经是当前帧的局部地图点了，可以防止重复添加局部地图点
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

/**
 * @brief 跟踪局部地图函数里，更新局部关键帧
 * 方法是遍历当前帧的地图点，将观测到这些地图点的关键帧和相邻的关键帧及其父子关键帧，作为mvpLocalKeyFrames
 * Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧 
 * Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧包括以下3种类型
 *      类型1：能观测到当前帧地图点的关键帧，也称一级共视关键帧
 *      类型2：一级共视关键帧的共视关键帧，称为二级共视关键帧
 *      类型3：一级共视关键帧的子关键帧、父关键帧
 * Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
 */
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    // Step 1：遍历当前帧的地图点，记录所有能观测到当前帧地图点的关键帧
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                	// 得到观测到该地图点的关键帧和该地图点在关键帧中的索引
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                	// 由于一个地图点可以被多个关键帧观测到,因此对于每一次观测,都对观测到这个地图点的关键帧进行累计投票
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
	                    // 这里的操作非常精彩！
	                    // map[key] = value，当要插入的键存在时，会覆盖键对应的原来的值。如果键不存在，则添加一组键值对
	                    // it->first 是地图点看到的关键帧，同一个关键帧看到的地图点会累加到该关键帧计数
	                    // 所以最后keyframeCounter 第一个参数表示某个关键帧，第2个参数表示该关键帧看到了多少当前帧(mCurrentFrame)的地图点，也就是共视程度
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }


    // 存储具有最多观测次数（max）的关键帧
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    // Step 2：更新局部关键帧（mvpLocalKeyFrames），添加局部关键帧有3种类型
    // 先清空局部关键帧
    mvpLocalKeyFrames.clear();
    // 先申请3倍内存，不够后面再加
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    // Step 2.1 类型1：能观测到当前帧地图点的关键帧作为局部关键帧 （将邻居拉拢入伙）（一级共视关键帧） 
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        // 如果设定为要删除的，跳过
        if(pKF->isBad())
            continue;
        
        // 寻找具有最大观测数目的关键帧
        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        // 添加到局部关键帧的列表里
        mvpLocalKeyFrames.push_back(pKF);
        // 用该关键帧的成员变量mnTrackReferenceForFrame 记录当前帧的id
        // 表示它已经是当前帧的局部关键帧了，可以防止重复添加局部关键帧
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    // Step 2.2 遍历一级共视关键帧，寻找更多的局部关键帧 
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        // 处理的局部关键帧不超过80帧
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        // 类型2:一级共视关键帧的共视（前10个）关键帧，称为二级共视关键帧（将邻居的邻居拉拢入伙）
        // 如果共视帧不足10帧,那么就返回所有具有共视关系的关键帧
        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        // vNeighs 是按照共视程度从大到小排列
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                // mnTrackReferenceForFrame防止重复添加局部关键帧
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    //? 找到一个就直接跳出for循环？
                    break;
                }
            }
        }

        // 类型3:将一级共视关键帧的子关键帧作为局部关键帧（将邻居的孩子们拉拢入伙）
        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    //? 找到一个就直接跳出for循环？
                    break;
                }
            }
        }

        // 类型3:将一级共视关键帧的父关键帧（将邻居的父母们拉拢入伙）
        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            // mnTrackReferenceForFrame防止重复添加局部关键帧
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                //! 感觉是个bug！如果找到父关键帧会直接跳出整个循环
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO) &&mvpLocalKeyFrames.size()<80)
    {
        //cout << "CurrentKF: " << mCurrentFrame.mnId << endl;
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            //cout << "tempKF: " << tempKeyFrame << endl;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    // Step 3：更新当前帧的参考关键帧，与自己共视程度最高的关键帧作为参考关键帧
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

/**
 * @details 重定位过程
 * @return true 
 * @return false 
 * 
 * Step 1：计算当前帧特征点的词袋向量
 * Step 2：找到与当前帧相似的候选关键帧
 * Step 3：通过BoW进行匹配
 * Step 4：通过EPnP算法估计姿态
 * Step 5：通过PoseOptimization对姿态进行优化求解
 * Step 6：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
 */
bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    // Step 1: 计算当前帧特征点的Bow映射
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    // Step 2：找到与当前帧相似的候选关键帧组
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    // 如果没有候选关键帧，则退出
    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    // 每个关键帧的解算器
    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    // 每个关键帧和当前帧中特征点的匹配关系
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    // 放弃某个关键帧的标记  
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    // 有效的候选关键帧数目
    int nCandidates=0;

    // Step 3：遍历所有的候选关键帧，通过BoW进行快速匹配，用匹配结果初始化PnP Solver
    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            // 当前帧和候选关键帧用BoW进行快速匹配，匹配结果记录在vvpMapPointMatches，nmatches表示匹配的数目
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            // 如果和当前帧的匹配数小于15,那么只能放弃这个关键帧
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                // 如果匹配数目够用，用匹配结果初始化MLPnPsolver
                // ? 为什么用MLPnP? 因为考虑了鱼眼相机模型，解耦某些关系？
                // 参考论文《MLPNP-A REAL-TIME MAXIMUM LIKELIHOOD SOLUTION TO THE PERSPECTIVE-N-POINT PROBLEM》
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                // 构造函数调用了一遍，这里重新设置参数
                pSolver->SetRansacParameters(
                    0.99,                    // 模型最大概率值，默认0.9
                    10,                      // 内点的最小阈值，默认8
                    300,                     // 最大迭代次数，默认300
                    6,                       // 最小集，每次采样六个点，即最小集应该设置为6，论文里面写着I > 5
                    0.5,                     // 理论最少内点个数，这里是按照总数的比例计算，所以epsilon是比例，默认是0.4
                    5.991);                  // 卡方检验阈值 //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    // 足够的内点才能匹配使用PNP算法，MLPnP需要至少6个点
    // 是否已经找到相匹配的关键帧的标志
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    // Step 4: 通过一系列操作,直到找到能够匹配上的关键帧
    // 为什么搞这么复杂？答：是担心误闭环
    while(nCandidates>0 && !bMatch)
    {
        //遍历当前所有的候选关键帧
        for(int i=0; i<nKFs; i++)
        {
            // 忽略放弃的
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            // 内点标记
            vector<bool> vbInliers;

            // 内点数
            int nInliers;

            // 表示RANSAC已经没有更多的迭代次数可用 -- 也就是说数据不够好，RANSAC也已经尽力了。。。
            bool bNoMore;

            // Step 4.1：通过MLPnP算法估计姿态，迭代5次
            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            // PnP算法的入口函数
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            // bNoMore 为true 表示已经超过了RANSAC最大迭代次数，就放弃当前关键帧
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                // Step 4.2：如果MLPnP 计算出了位姿，对内点进行BA优化
                Tcw.copyTo(mCurrentFrame.mTcw);

                // MLPnP 里RANSAC后的内点的集合
                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                // 遍历所有内点
                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                // 只优化位姿,不优化地图点的坐标，返回的是内点的数量
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                // 如果优化之后的内点数目不多，跳过了当前候选关键帧,但是却没有放弃当前帧的重定位
                if(nGood<10)
                    continue;

                // 删除外点对应的地图点,这里直接设为空指针
                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                // Step 4.3：如果内点较少，则通过投影的方式对之前未匹配的点进行匹配，再进行优化求解
                // 前面的匹配关系是用词袋匹配过程得到的
                if(nGood<50)
                {
                    // 通过投影的方式将关键帧中未匹配的地图点投影到当前帧中, 生成新的匹配
                    int nadditional =matcher2.SearchByProjection(
                        mCurrentFrame,           // 当前帧
                        vpCandidateKFs[i],       // 关键帧
                        sFound,                  // 已经找到的地图点集合，不会用于PNP
                        10,                      // 窗口阈值，会乘以金字塔尺度
                        100);                    // 匹配的ORB描述子距离应该小于这个阈值

                    // 如果通过投影过程新增了比较多的匹配特征点对
                    if(nadditional+nGood>=50)
                    {
                        // 根据投影匹配的结果，再次采用3D-2D pnp BA优化位姿
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        // Step 4.4：如果BA后内点数还是比较少(<50)但是还不至于太少(>30)，可以挽救一下, 最后垂死挣扎 
                        // 重新执行上一步 4.3的过程，只不过使用更小的搜索窗口
                        // 这里的位姿已经使用了更多的点进行了优化,应该更准，所以使用更小的窗口搜索
                        if(nGood>30 && nGood<50)
                        {
                            // 用更小窗口、更严格的描述子阈值，重新进行投影搜索匹配
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(
                                mCurrentFrame,              // 当前帧
                                vpCandidateKFs[i],          // 候选的关键帧
                                sFound,                     // 已经找到的地图点，不会用于PNP
                                3,                          // 新的窗口阈值，会乘以金字塔尺度
                                64);                        // 匹配的ORB描述子距离应该小于这个阈值

                            // Final optimization
                            // 如果成功挽救回来，匹配数目达到要求，最后BA优化一下
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                                //更新地图点
                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                            //如果还是不能够满足就放弃了
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                // 如果对于当前的候选关键帧已经有足够的内点(50个)了,那么就认为重定位成功
                if(nGood>=50)
                {
                    bMatch = true;
                    // 只要有一个候选关键帧重定位成功，就退出循环，不考虑其他候选关键帧了
                    break;
                }
            }
        }//一直运行,知道已经没有足够的关键帧,或者是已经有成功匹配上的关键帧
    }

    // 折腾了这么久还是没有匹配上，重定位失败
    if(!bMatch)
    {
        return false;
    }
    else
    {
        // 如果匹配上了,说明当前帧重定位成功了(当前帧已经有了自己的位姿)
        // 记录成功重定位帧的id，防止短时间多次重定位
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

//整个追踪线程执行复位操作
void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);
	//基本上是挨个请求各个线程终止
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }


    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    //然后复位各种变量
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();

    mnLastInitFrameId = Frame::nNextId;
    mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    list<bool> lbLost;
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    //做标记,表示在初始化帧的时候将会是第一个帧,要对它进行一些特殊的初始化操作
    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).rowRange(0,3).col(3)=(*lit).rowRange(0,3).col(3)*s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    cv::Mat Gz = (cv::Mat_<float>(3,1) << 0, 0, -IMU::GRAVITY_VALUE);

    cv::Mat twb1;
    cv::Mat Rwb1;
    cv::Mat Vwb1;
    float t12;

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }


    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation(),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}


cv::Mat Tracking::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w*R2w.t();
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w;

    cv::Mat t12x = Converter::tocvSkewMatrix(t12);

    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;


    return K1.t().inv()*t12x*R12*K2.inv();
}


void Tracking::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    const vector<KeyFrame*> vpKFs = mpAtlas->GetAllKeyFrames();

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpLastKeyFrame->GetRotation();
    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat tcw1 = mpLastKeyFrame->GetTranslation();
    cv::Mat Tcw1(3,4,CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));
    cv::Mat Ow1 = mpLastKeyFrame->GetCameraCenter();

    const float &fx1 = mpLastKeyFrame->fx;
    const float &fy1 = mpLastKeyFrame->fy;
    const float &cx1 = mpLastKeyFrame->cx;
    const float &cy1 = mpLastKeyFrame->cy;
    const float &invfx1 = mpLastKeyFrame->invfx;
    const float &invfy1 = mpLastKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpLastKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF2 = vpKFs[i];
        if(pKF2==mpLastKeyFrame)
            continue;

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();
        cv::Mat vBaseline = Ow2-Ow1;
        const float baseline = cv::norm(vBaseline);

        if((mSensor!=System::MONOCULAR)||(mSensor!=System::IMU_MONOCULAR))
        {
            if(baseline<pKF2->mb)
            continue;
        }
        else
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpLastKeyFrame,pKF2);

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpLastKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();
        cv::Mat Tcw2(3,4,CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size();
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first;
            const int &idx2 = vMatchedIndices[ikp].second;

            const cv::KeyPoint &kp1 = mpLastKeyFrame->mvKeysUn[idx1];
            const float kp1_ur=mpLastKeyFrame->mvuRight[idx1];
            bool bStereo1 = kp1_ur>=0;

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];
            const float kp2_ur = pKF2->mvuRight[idx2];
            bool bStereo2 = kp2_ur>=0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0);
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0);

            cv::Mat ray1 = Rwc1*xn1;
            cv::Mat ray2 = Rwc2*xn2;
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpLastKeyFrame->mb/2,mpLastKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpLastKeyFrame->UnprojectStereo(idx1);
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpLastKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpLastKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpLastKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpLastKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpLastKeyFrame,mpAtlas->GetCurrentMap());

            pMP->AddObservation(mpLastKeyFrame,idx1);
            pMP->AddObservation(pKF2,idx2);

            mpLastKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpAtlas->AddMapPoint(pMP);
            nnew++;
        }
    }
    TrackReferenceKeyFrame();
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

/*
 * @brief 用基于Good Feature Match来进行局部地图跟踪
 *
 * 
*/
bool Tracking::TrackLocalMapByGF()
{
    arma::wall_clock timer;
    // mTrackedFr++;

// #ifdef REGISTER_TIMES
//     std::chrono::steady_clock::time_point time_StartLMUpdate = std::chrono::steady_clock::now();
// #endif

    // // Step 1：更新局部关键帧 mvpLocalKeyFrames 和局部地图点 mvpLocalMapPoints
    UpdateLocalMapByGF();
// #ifdef REGISTER_TIMES
//     std::chrono::steady_clock::time_point time_StartSearchLP = std::chrono::steady_clock::now();

//     double timeUpdatedLM_ms = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_StartSearchLP - time_StartLMUpdate).count();
//     vdUpdatedLM_ms.push_back(timeUpdatedLM_ms);
// #endif

    int mnMatchesInit = SearchLocalPointsByGF();

    double time_srh_ref = timer.toc();

    double time_stereo = 0;

    // timer.tic();
    int nStereo = mCurrentFrame.ComputeStereoMatches_Undistorted(true);
    // //    cout << "func TrackLocalMap: number of points being stereo matched = " << nStereo << "             "  << endl;
    // time_stereo = timer.toc();

    timer.tic();
    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;
    double time_opt = timer.toc();

    timer.tic();

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor!=System::MONOCULAR)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    double time_post = timer.toc();

    if(mCurrentFrame.mnId < mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<25)
        return false;

    if(mnMatchesInliers<15)
        return false;
    else
        return true;
}

void Tracking::UpdateLocalMapByGF()
{
    // 设置参考地图点用于绘图显示局部地图点（红色）
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // 用共视图来更新局部关键帧
    UpdateLocalKeyFrames();

    // 用共视图来更新局部地图点

// #ifdef LOCAL_SEARCH_USING_HASHING
//     arma::wall_clock timer;
//     timer.tic();
//     //    UpdateReferencePointsByHashing();
//     //        if(mCurrentFrame.mTimeStamp - timestamp_first_frame > 5.0)

//     //    if (mpLocalMapper->stopRequested() || mpLocalMapper->isStopped())
//     //    {
//     //        UpdateLocalPointsByHashing(eLocalMapSet::CovisOnly);
//     //    }
//     //    else
//     //    {
    // cout<< "--- mvp local map points number = " << mvpLocalMapPoints.size() <<endl;
    // if (mFrameAfterInital >= camera_fps * 2 || mvpLocalMapPoints.size() > 2000 )// || logCurrentFrame.lmk_num_frame > this->num_good_constr_predef)
    // {
    //     // cout<<"Collect Local Map with Multi-Index Hasing!"<<endl;
    //     UpdateLocalPointsByHashing(eLocalMapSet::Combined);
    // }
    // else
    // {
    //     // cout<<"Collect Local Map with Co-Visibility!"<<endl;
    //     UpdateLocalPointsByHashing(eLocalMapSet::CovisOnly);
    // }

    UpdateLocalPoints();

    // mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

}

/**
 * @brief 基于Good Feature Match用局部地图点进行投影匹配，得到更多的匹配关系
 * 
 */
int Tracking::SearchLocalPointsByGF()
{
    int nAlreadyMatched = 0;

    arma::wall_clock timer;

    mCurrentInfoMat = arma::eye( size(mCurrentInfoMat) ) * 0.00001;

    if (mFrameAfterInital > camera_fps * 3 && mCurrentFrame.mnId >= mnLastRelocFrameId+2) {
        //    cout << "update pose info in obs class" << endl;
        // NOTE
        // there is no need to do motion prediction again, since it's already be
        // predicted and somewhat optimzed in the 1st stage of pose tracking
        // 无需再次进行运动预测，因为它已经在姿势跟踪的第一阶段进行了预测和优化
        if(!mLastFrame.mTcw.empty()) {
            mObsHandler->updatePWLSVec(mLastFrame.mTimeStamp, mLastFrame.mTcw,
                                       mCurrentFrame.mTimeStamp, mCurrentFrame.getTwc());
        }
        //    cout << "propagate pose info in obs class" << endl;
        // NOTE
        // instead of using actual time between consequtive frames, we construct virtual frame with slightly longer horizon;
        // 我们不使用连续帧之间的实际时间，而是构建视野稍长的虚拟帧；
        // the motivation being: reducing the size of matrix to 2-segments (which is the minimim-size); meanwhile preserving the spectral property
        // 动机是：将矩阵的大小减少到 2 段（这是最小大小）； 同时保留光谱特性
        mObsHandler->predictPWLSVec( (mCurrentFrame.mTimeStamp - mLastFrame.mTimeStamp), 1 );
        mObsHandler->mKineIdx = 0;
        mObsHandler->mnFrameId = mCurrentFrame.mnId;

        // mObsHandler->pFrame = &mCurrentFrame;
        // mObsHandler->runMatrixBuilding(ORB_SLAM3::FRAME_INFO_MATRIX, 0.002, true, false);
    }

    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;

#if defined GOOD_FEATURE_MAP_MATCHING && defined FRAME_MATCHING_INFO_PRIOR
                if (pMP->updateAtFrameId == mCurrentFrame.mnId)
                    mCurrentInfoMat += pMP->ObsMat;
#endif
                nAlreadyMatched += 2;
            }
        }
    }

    int nToMatch=0;
    mObsHandler->mLeftMapPoints.clear();
    mObsHandler->mbNeedVizCheck = false;
    double time_total_match = 0.015; // 1.0; 
    int num_to_match = this->num_good_constr_predef - nAlreadyMatched; // 50;  //
    if (num_to_match <= 0) {
        // skip the rest
        for (size_t i=0; i<mvpLocalMapPoints.size(); ++i) {
            if (mvpLocalMapPoints[i] == NULL)
                continue ;
            if (mvpLocalMapPoints[i]->isBad())
                continue ;
            if(mvpLocalMapPoints[i]->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (mvpLocalMapPoints[i]->mbTrackInView == false)
                continue ;
            //
            mObsHandler->mLeftMapPoints.push_back(mvpLocalMapPoints[i]);
        }
        mObsHandler->mbNeedVizCheck = true;
        //
        return nAlreadyMatched;
    }

    double time_Viz = 0;
    timer.tic();

    // Project points in frame and check its visibility
    // 在框架中投影点并检查其可见性
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;

        time_Viz = timer.toc();
        if (time_Viz > time_total_match / 2.0) {
            //
            mObsHandler->mLeftMapPoints = vector<MapPoint*>(vit, vend);
            mvpLocalMapPoints.erase(vit, vend);
            mObsHandler->mbNeedVizCheck = true;
            break ;
        }

        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    // Here is the place to inject Obs computation; to reduce the load of computation, we might need to approximate exact point Obs with region;
    // Following are the time cost of each step in TrackLocalMap:
    int nMatched = 0;
    if(nToMatch>0)
    {
        timer.tic();

        ORBmatcher matcher(0.8);

        int th = 1;  // 1.5; // NOTE try increase the window size for vicon seq
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId < mnLastRelocFrameId+2) {
            th=5;
            nMatched = matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        }
        else if (mFrameAfterInital <= camera_fps * 5 || nToMatch < 400) { // 800)
            nMatched = matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
        }
        else {
            // try computing Jacobian for each map point
            mObsHandler->mMapPoints = &mvpLocalMapPoints;
            mObsHandler->runMatrixBuilding(ORB_SLAM3::MAP_INFO_MATRIX, (time_total_match-time_Viz)/2, true, false);
            double time_Mat_Online = timer.toc();
            nMatched = mObsHandler->runActiveMapMatching(&mCurrentFrame, ORB_SLAM3::FRAME_INFO_MATRIX, mCurrentInfoMat,
                                                         th,matcher,num_to_match,time_total_match-time_Mat_Online-time_Viz);
            mCurrentFrame.isGFpoints = true;
        }
        double time_Match = timer.toc();
    }

    return nAlreadyMatched + nMatched;
}

// 
void Tracking::UpdateLocalPointsByHashing(eLocalMapSet eLocalMap)
{
    //#ifdef TIMECOST_VERBOSE
    arma::wall_clock timer/*, timer1*/;
    double t;
    //#endif
    mbMapHashOTS = false;

    timer.tic();

    if (eLocalMap == eLocalMapSet::CovisOnly || eLocalMap == eLocalMapSet::Combined) {
        // grab the local map from co-visibility
        mvpLocalMapPoints.clear();

        for(vector<KeyFrame*>::iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
        {
            KeyFrame* pKF = *itKF;
            vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

            for(vector<MapPoint*>::iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
            {
                MapPoint* pMP = *itMP;
                if(!pMP)        // already matched or outliers
                    continue;
                if(pMP->mnIdCoVisible == mCurrentFrame.mnId)   // check if selected yet (avoid pushing back same MapPoints multiple times)
                    continue;
                if(!pMP->isBad())
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnIdCoVisible = mCurrentFrame.mnId;
                    //                    pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;

                    //                    // flow from co-visibility => maphash
                    //                    timer1.tic();
                    //                    mpHashMethod->insert(pMP);
                    //                    t += timer1.toc();
                }
            }
        }
        StoreLocalMapPointsByCoVis(mvpLocalMapPoints);
        if (mvpLocalMapPoints.size() < 2000) {
            // update flag for final local map
            for(vector<MapPoint*>::iterator itMP=mvpLocalMapPoints.begin(), itEndMP=mvpLocalMapPoints.end(); itMP!=itEndMP; itMP++)
            {
                MapPoint* pMP = *itMP;
                if(!pMP || pMP->isBad())    // not matched yet
                    continue;
                if(pMP->mnTrackReferenceForFrame != mCurrentFrame.mnId)   //
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;

                //
                //                pMP->matchedByTrackLocalMap = false;
            }
            //            std::cout << "func UpdateReferencePointsByHashing: size of co-vis map = " << mvpLocalMapPoints.size() << std::endl;
            return;
        }
    }

    //    cerr<<"Time cost:"<<timer.toc()<<", ---------------------------------"<<endl;
    if (eLocalMap == eLocalMapSet::HashOnly || eLocalMap == eLocalMapSet::Combined) {
        mbMapHashTriggered = true;
        //
        // grab the local map from map hash
        cv::Mat mDescriptors = mCurrentFrame.mDescriptors;
        //    std::vector<MapPoint*> vpMP = mCurrentFrame.mvpMapPoints;
        std::vector<unsigned int> vKPU; // keypoint indices which need to be matched to LocalMapPoints

        //#ifdef TIMECOST_VERBOSE
        // timer.tic();
        //#endif

        Frame::GetUnMatchedKPbyBucketing(&mCurrentFrame, vKPU);

        //    bool hit[pMIH->getHashTableSize()][pMIH->getBucketNum()] = {0}; // only works for false initialization
        //    std::vector<std::vector<size_t> > hitid;
        //    hitid.clear();

        mvpLeftMapPointsByHashing.clear();
        mvpMapPointsByHashing.clear();

        //        std::cout << "before iterating vKPU" << std::endl;
        t = 0.;
        for(auto i:vKPU) {
            //
            unsigned int* pDesc = mDescriptors.ptr<unsigned int>(i);
            std::vector<MapPoint*> candidates;
            std::vector<MapPoint*> additional_candidates;

            //            std::cout << "before query vKPU" << std::endl;

            timer.tic();
            mpHashMethod->query(pDesc, candidates, additional_candidates, mvpQueriedFeatures);
            mvpLeftMapPointsByHashing.insert(mvpLeftMapPointsByHashing.end(), additional_candidates.begin(), additional_candidates.end());
            // retrieve up to NUM_CANDIDATE_FEATURES x NUM_ACTIVE_HASHTABLES candidates

            t += timer.toc();

            for(vector<MapPoint*>::iterator itMP=candidates.begin(), itEndMP=candidates.end(); itMP!=itEndMP; itMP++)
            {
                MapPoint* pMP = *itMP;
                if(!pMP)
                    continue;
                if(pMP->mnIdMapHashed==mCurrentFrame.mnId)
                    continue;
                if(!pMP->isBad())
                {
                    mvpMapPointsByHashing.push_back(pMP);
                    pMP->mnIdMapHashed = mCurrentFrame.mnId;
                }
            }
        }

        //        std::cout << "done iterating vKPU" << std::endl;
        //        std::sort(mvpHashMapPoints.begin(), mvpHashMapPoints.end(), sortMapPoint_ascend);
    }

    if (eLocalMap == eLocalMapSet::Combined) {
        // set operation of both local maps
        //
        vector<MapPoint *> mvpInterMapPoints;
        for(vector<MapPoint*>::iterator itMP=mvpLocalMapPoints.begin(), itEndMP=mvpLocalMapPoints.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP || pMP->isBad())    // not matched yet
                continue;
            if(pMP->mnIdMapHashed == mCurrentFrame.mnId) {
                mvpInterMapPoints.push_back(pMP);
            }
            else {
                mvpLeftMapPointsByHashing.push_back(pMP);
            }
        }

        // copy from the intersected result
        mvpLocalMapPoints.clear();
        mvpLocalMapPoints.swap(mvpInterMapPoints);
    }
    else {

        if (eLocalMap == eLocalMapSet::HashOnly) {
            // simply copy from hashed result
            mvpLocalMapPoints.clear();
            mvpLocalMapPoints.swap(mvpMapPointsByHashing);
        }
    }

    // update flag for final local map
    for(vector<MapPoint*>::iterator itMP=mvpLocalMapPoints.begin(), itEndMP=mvpLocalMapPoints.end(); itMP!=itEndMP; itMP++)
    {
        MapPoint* pMP = *itMP;
        if(!pMP || pMP->isBad())    // not matched yet
            continue;
        if(pMP->mnTrackReferenceForFrame != mCurrentFrame.mnId)
            pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;

        //        pMP->matchedByTrackLocalMap = false;
    }
}

void Tracking::UpdateLocalPointsByGF()
{
    mvpLocalMapPoints.clear();

    unsigned long minFrameId = mCurrentFrame.mnId; // - 2; // - 1; //
    if (mbTrackLossAlert == 2)
        minFrameId = 0;

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                //                if (pMP->goodAtFrameId > 0 || mvpLocalMapPoints.size() < 2000) {
                //                    acceptMapPoint = true;
                //                }
                if (pMP->goodAtFrameId >= minFrameId)
                {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
                else
                {
                    // for those unselected map points, we still keep them and try matching them after current pose being published
                    // mvpLocalAdditionalPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }

            }
        }
    }
}

void Tracking::UpdateLocalKeyFramesByGF()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*, std::tuple<int, int>> observations = pMP->GetObservations();
                for(map<KeyFrame*, std::tuple<int, int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty()) {
        cout << "keyframeCounter is empty!" << endl;
        return;
    }

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            //            std::cout << pParent << std::endl;
            //            std::cout << pParent->mnTrackReferenceForFrame << std::endl;
            //            std::cout << mCurrentFrame.mnId << std::endl;

            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

} //namespace ORB_SLAM

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"
#include <opencv2/features2d/features2d.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "ORBextractor.h"

using namespace std;

int nlevels;
float scaleFactor;
int isGetImageSize;
int Threshold;

cudaStream_t *streams;

uchar threshold_tab[512];
__device__ uchar* threshold_tab_cuda;

thrust::device_vector<uchar1*> d_srcs;
thrust::device_vector<uchar1*> d_dsts;

__device__ ushort3* Keys;
thrust::device_vector<uchar1*> score_mats;
__device__ ushort3* KeyPoints;

inline __device__ __host__ int divUp(int A, int B) { return (A + B - 1) / B; }

// 灰度图像高斯滤波
__global__ void gray_gaussian_filtering(uchar1* img, uchar1* dst, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x + 16;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y + 16;

    if(idx < width && idy < height)
    {
        float v1 = img[idy * width + idx].x * 0.2725;
        float v2 = (img[(idy+1)*width + idx+1].x + img[(idy+1)*width + idx-1].x + img[(idy-1)*width + idx+1].x + img[(idy-1)*width + idx-1].x) * 0.0571;
        float v3 = (img[(idy+1)*width + idx].x + img[(idy-1)*width + idx].x + img[idy*width + idx+1].x + img[idy*width + idx-1].x) * 0.1248;
        dst[idy*width + idx].x = v1 + v2 + v3;
    }
}

__device__ int cornerScore(uchar v, uchar* c, int threshold) {
    const int N = 25;
    int d[N];
    d[0] = abs(v - c[0]);
    d[1] = abs(v - c[1]);
    d[2] = abs(v - c[2]);
    d[3] = abs(v - c[3]);
    d[4] = abs(v - c[4]);
    d[5] = abs(v - c[5]);
    d[6] = abs(v - c[6]);
    // d[7] = abs(v - c[7]);
    d[8] = abs(v - c[8]);
    d[9] = abs(v - c[9]);
    d[10] = abs(v - c[10]);
    d[11] = abs(v - c[11]);
    d[12] = abs(v - c[12]);
    d[13] = abs(v - c[13]);
    d[14] = abs(v - c[14]);
    // d[15] = abs(v - c[15]);
    d[16] = d[0];
    // d[17] = d[1];
    d[18] = d[2];
    // d[19] = d[3];
    d[20] = d[4];
    // d[21] = d[5];
    d[22] = d[6];
    // d[23] = d[7];
    // d[24] = d[8];

    int a0 = threshold;

    int a = min(d[1], d[2]);
    // a = min(a, d[3]);
    a = min(a, d[4]);
    // a = min(a, d[5]);
    a = min(a, d[6]);
    // a = min(a, d[7]);
    a = min(a, d[8]);
    // a = min(a, d[9]);
    a = min(a, d[0]);
    a0 = max(a0, a);

    a = min(d[5], d[6]);
    // a = min(a, d[7]);
    a = min(a, d[8]);
    // a = min(a, d[9]);
    a = min(a, d[10]);
    // a = min(a, d[11]);
    a = min(a, d[12]);
    // a = min(a, d[13]);
    a = min(a, d[14]);
    a0 = max(a0, a);

    a = min(d[9], d[10]);
    // a = min(a, d[11]);
    a = min(a, d[12]);
    // a = min(a, d[13]);
    a = min(a, d[14]);
    // a = min(a, d[15]);
    a = min(a, d[16]);
    // a = min(a, d[17]);
    a = min(a, d[18]);
    a0 = max(a0, a);

    a = min(d[13], d[14]);
    // a = min(a, d[15]);
    a = min(a, d[16]);
    // a = min(a, d[17]);
    a = min(a, d[18]);
    // a = min(a, d[19]);
    a = min(a, d[20]);
    // a = min(a, d[21]);
    a = min(a, d[22]);
    a0 = max(a0, a);


    // for (int i=0;i<16;i+=4) {
    //     int a = min(d[i+1], d[i+2]);
    //     a = min(a, d[i+3]);
    //     if (a <= a0) {
    //         continue;
    //     }
    //     a = min(a, d[i+4]);
    //     a = min(a, d[i+5]);
    //     a = min(a, d[i+6]);
    //     a = min(a, d[i+7]);
    //     a = min(a, d[i+8]);
    //     a0 = max(a0, min(a, d[i]));
    //     a0 = max(a0, min(a, d[i+9]));
    // }

    return  a0 - 1;
}

__device__ int fast_detect_16(uchar1* img, const short idx, const short idy, const int threshold, uchar* threshold_tab, int width, int height)
{
    uchar v = img[idy * width + idx].x; // 读取判定点的灰度值

    int radiu = 3;
    uchar c[25];

    c[0] = img[(idy - radiu) * width + idx].x;
    c[1] = img[(idy - radiu) * width + idx + radiu - 2].x;
    c[2] = img[(idy - radiu + 1) * width + idx + radiu - 1].x;
    c[3] = img[(idy - radiu + 2) * width + idx + radiu].x;
    c[4] = img[idy * width + idx + radiu].x;
    c[5] = img[(idy + radiu - 2) * width + idx + radiu].x;
    c[6] = img[(idy + radiu - 1) * width + idx + radiu - 1].x;
    c[7] = img[(idy + radiu) * width + idx + radiu - 2].x;
    c[8] = img[(idy + radiu) * width + idx].x;
    c[9] = img[(idy + radiu) * width + idx - radiu + 2].x;
    c[10] = img[(idy + radiu - 1) * width + idx - radiu + 1].x;
    c[11] = img[(idy + radiu - 2) * width + idx - radiu].x;
    c[12] = img[idy * width + idx - radiu].x;
    c[13] = img[(idy - radiu + 2) * width + idx - radiu].x;
    c[14] = img[(idy - radiu + 1) * width + idx - radiu + 1].x;
    c[15] = img[(idy - radiu) * width + idx - radiu + 2].x;
    c[16] = c[0];
    c[17] = c[1];
    c[18] = c[2];
    c[19] = c[3];
    c[20] = c[4];
    c[21] = c[5];
    c[22] = c[6];
    c[23] = c[7];
    c[24] = c[8];

    uchar* tab = &threshold_tab[0] - v + 255;

    int d = tab[c[0]] | tab[c[8]];

    if (d == 0) {
        return -1;
    }

    d &= tab[c[2]] | tab[c[10]];
    d &= tab[c[4]] | tab[c[12]];
    d &= tab[c[6]] | tab[c[14]];

    if (d == 0) {
        return -1;
    }

    d &= tab[c[1]] | tab[c[9]];
    d &= tab[c[3]] | tab[c[11]];
    d &= tab[c[5]] | tab[c[13]];
    d &= tab[c[7]] | tab[c[15]];

    if (d & 1) {
        int vt = v - threshold, count = 0;

        // count += c[0] < vt ? 1 : 0;
        // count += c[2] < vt ? 1 : 0;
        // count += c[4] < vt ? 1 : 0;
        // count += c[6] < vt ? 1 : 0;
        // count += c[8] < vt ? 1 : 0;
        // count += c[10] < vt ? 1 : 0;
        // count += c[12] < vt ? 1 : 0;
        // count += c[14] < vt ? 1 : 0;

        // if (count > 6) {
        //     // int response = cornerScore(v, c, threshold);
        //     // int response = 1;
        //     return response;
        // }
        int response = 255;
        for (int k = 0;k < 16;k+=2) {
            int x = c[k];
            if (x < vt) {
                response = response < x ? response : x;
                if (++count > 6) {
                    // int response = cornerScore(v, c, threshold);
                    // int response = 1;
                    return response;
                }
            }
        }
    }

    if (d & 2) {
        int vt = v + threshold, count = 0;

        // count += c[0] > vt ? 1 : 0;
        // count += c[2] > vt ? 1 : 0;
        // count += c[4] > vt ? 1 : 0;
        // count += c[6] > vt ? 1 : 0;
        // count += c[8] > vt ? 1 : 0;
        // count += c[10] > vt ? 1 : 0;
        // count += c[12] > vt ? 1 : 0;
        // count += c[14] > vt ? 1 : 0;

        // if (count > 6) {
        //     // int response = cornerScore(v, c, threshold);
        //     // int response = 1;
        //     return response;
        // }

        int response = 0;
        for (int k = 0;k < 16;k+=2) {
            int x = c[k];
            if (x > vt) {
                response = response > x ? response : x;
                if (++count > 6) {
                    // int response = cornerScore(v, c, threshold);
                    // int response = 1;
                    return response;
                }
            }
        }
    }
    
    return -1;
}

__global__ void fast(uchar1* img, const int threshold, uchar* threshold_tab, int *dev_counter,
                        ushort3 *kptsLoc2D, uchar1* score_mat,
                        int width, int height) 
{
    const ushort idx = blockIdx.x * blockDim.x + threadIdx.x + 16;
    const ushort idy = blockIdx.y * blockDim.y + threadIdx.y + 16;

    if (idx < width - 16 && idy < height - 16)
    {
        // 检查是否通过 Fast 角点的收录标准
        int response = fast_detect_16(img, idx, idy, threshold, threshold_tab, width, height);
        if (response > 0)
        {
            int pIdx = atomicAdd(dev_counter, 1);
            if (pIdx < 10000)
            {
                kptsLoc2D[pIdx] = {idx, idy, response};
                score_mat[idy * width + idx].x = response;
            }
        }
    }
}

#define NM_RADIUS 3 // 非极大抑制范围 Radius
// 非极大抑制，剔除部分 Edge 点(negtive)，同时保存特征点的三维位置信息
__global__ void nonmaxSuppression(ushort3* kptsLoc2D, int kpt_num,
                                  uchar1* scoreMat, int* dev_counter,
                                  ushort3* frame_kpts,
                                  int width, int height)
{
    const short kpIdx = threadIdx.x + blockIdx.x * blockDim.x;
    if (kpIdx < kpt_num)
    {
        ushort3 loc = kptsLoc2D[kpIdx];
        int score = scoreMat[loc.y * width + loc.x].x;

        for (int i = -NM_RADIUS; i <= NM_RADIUS; i++)
            for (int j = -NM_RADIUS; j <= NM_RADIUS; j++)
                if (score < scoreMat[(loc.y + i) * width + loc.x + j].x)
                    return;

        int idx = atomicAdd(dev_counter, 1);
        // 写入特征点图像坐标，三维空间坐标
        if (idx < 50000)
        {
            frame_kpts[idx] = loc;
        }
    }
}

// CUDA 初始化，主要得到 金字塔层数 以及 是否得到图像尺寸设为 0
void ORB_SLAM3::ORBextractor::CUDA_Initial(int _nlevel, float _scaleFactor)
{
    nlevels = _nlevel;
    scaleFactor = _scaleFactor;
    isGetImageSize = 0;

    d_srcs.resize(nlevels);
    d_dsts.resize(nlevels);
    score_mats.resize(nlevels);

    streams = (cudaStream_t*)malloc(nlevels * sizeof(cudaStream_t));
    for (int i=0;i<nlevels;i++) {
        cudaStreamCreate(&streams[i]);
    }

    Threshold = 25;
    for (int i = -255;i <= 255;i++) {
        threshold_tab[i+255] = (uchar)(i < -Threshold ? 1 : i > Threshold ? 2 : 0);
    }
    size_t memSize_tt = 512 * sizeof(uchar);
    cudaMalloc((void**)&threshold_tab_cuda, memSize_tt);
    cudaMemcpy(threshold_tab_cuda, threshold_tab, memSize_tt, cudaMemcpyHostToDevice);
}

// 创建GPU内存空间
void ORB_SLAM3::ORBextractor::getPyramid(cv::InputArray image, int level)
{
    cv::Mat src = image.getMat();
    int height = src.rows;
    int width = src.cols;

    // GPU内存初始化
    if (isGetImageSize == 0) {
        uchar1* d_src;
        uchar1* d_dst;
        uchar1* score_mat;

        size_t memSize = width * height * sizeof(uchar1);
        cudaMalloc((void**)&d_src, memSize);
        cudaMalloc((void**)&d_dst, memSize);
        cudaMalloc((void**)&score_mat, memSize);

        d_srcs[level] = d_src;
        d_dsts[level] = d_dst;
        score_mats[level] = score_mat;

        // 只创建一次内存空间
        if (level == nlevels-1) {
            isGetImageSize = 1;

            size_t memSizes = 10000 * sizeof(ushort3);
            cudaMalloc((void**)&Keys, memSizes);

            cudaMalloc((void**)&KeyPoints, memSizes);
        }
    }
    
    // 图像信息赋值
    size_t memSize = width * height * sizeof(uchar1);
    cudaMemcpy(d_srcs[level], src.data, memSize, cudaMemcpyHostToDevice);
}

void ORB_SLAM3::ORBextractor::GBandCD_CUDA(cv::InputArray image, int level)
{
    cv::Mat src = image.getMat();

    size_t memSize = src.cols * src.rows * sizeof(uchar1);

    dim3 block(16, 16);
    dim3 grid(divUp(src.cols, block.x),
              divUp(src.rows, block.y));
    gray_gaussian_filtering<<<grid, block>>>(d_srcs[level], d_dsts[level], src.cols, src.rows);

    cudaMemcpy(src.data, d_dsts[level], memSize, cudaMemcpyDeviceToHost);

}

void ORB_SLAM3::ORBextractor::ExtractorPoint(cv::InputArray image, int level, vector<cv::KeyPoint> &vKeys)
{
    cv::Mat src = image.getMat();
    int Rows = src.rows;
    int Cols = src.cols;

    int *dev_counter;
    cudaMalloc((void**)&dev_counter, sizeof(int));
    cudaMemset(dev_counter, 0, sizeof(int));

    size_t memSize = Cols * Rows * sizeof(uchar1);
    cudaMemset(score_mats[level], 0, memSize);

    dim3 block(16, 16);
    dim3 grid(divUp(Cols - 2*16, block.x),
              divUp(Rows - 2*16, block.y));
    fast<<<grid, block>>>(d_srcs[level], Threshold, threshold_tab_cuda, dev_counter, Keys, score_mats[level], Cols, Rows);

    int* n;
    n = (int*)malloc(sizeof(int));
    cudaMemcpy(n, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // cout<< "fast KeyPoint num: " << *n <<endl;

    int *kp_counter;
    cudaMalloc((void**)&kp_counter, sizeof(int));
    cudaMemset(kp_counter, 0, sizeof(int));

    dim3 block_nm(64);
    dim3 grid_nm(divUp(*n, block_nm.x));
    nonmaxSuppression<<<grid_nm, block_nm>>>(Keys, *n, score_mats[level], kp_counter, KeyPoints, Cols, Rows);

    int* num;
    num = (int*)malloc(sizeof(int));
    cudaMemcpy(num, kp_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // cout<< "nonmax KeyPoint num: " << *num <<endl;

    ushort3* kptsLoc = (ushort3*)malloc(*num * sizeof(ushort3));
    cudaMemcpy(kptsLoc, KeyPoints, *num * sizeof(ushort3), cudaMemcpyDeviceToHost);

    for (int pidx = 0; pidx < *num; pidx++)
    {
        cv::KeyPoint kp;
        kp.pt.x = kptsLoc[pidx].x - 16;
        kp.pt.y = kptsLoc[pidx].y - 16;
        kp.response = kptsLoc[pidx].z;
        kp.octave = level;
        kp.size = 0;
        vKeys.push_back(kp);

        // cv::Scalar color(0, 255, 0);
        // cv::circle(src, {kptsLoc[pidx].x, kptsLoc[pidx].y}, 3, color);
    }

    // cv::imshow("color coarse fast ", src);
    // cv::waitKey();

    cudaFree(dev_counter);
    cudaFree(kp_counter);

    free(kptsLoc);
    free(n);
    free(num);
}


void ORB_SLAM3::ORBextractor::ExtractorPointStream(std::vector<cv::Mat> &mvImagePyramid, vector<vector<cv::KeyPoint>> &vKeys)
{
    const int W = 35;

    // 初始化 DtoH 数组
    vector<ushort3*> keys;
    for (int i = 0;i < nlevels;i++) {
        ushort3* _k = (ushort3*)malloc(10000 * sizeof(ushort3));
        keys.push_back(_k);
    }
    vector<int> keysNum;
    keysNum.resize(nlevels);

    int *dev_counter;
    cudaMalloc((void**)&dev_counter, nlevels * sizeof(int));
    cudaMemset(dev_counter, 0, nlevels * sizeof(int));

    int* n= (int*)malloc(nlevels * sizeof(int));

    int *kp_counter;
    cudaMalloc((void**)&kp_counter, nlevels * sizeof(int));
    cudaMemset(kp_counter, 0, nlevels * sizeof(int));

    int* num = (int*)malloc(nlevels * sizeof(int));

    for (int level = 0; level < nlevels; level++)
    {
        const int Cols = mvImagePyramid[level].cols;
        const int Rows = mvImagePyramid[level].rows;
        size_t memSize = Cols * Rows * sizeof(uchar1);
        cudaMemset(score_mats[level], 0, memSize);
    }

    // CUDA stream
    for (int level = 0; level < nlevels; level++)
    {
        cv::Mat img = mvImagePyramid[level].clone();
        const int Cols = img.cols;
        const int Rows = img.rows;

        size_t memSize = Cols * Rows * sizeof(uchar1);
        cudaMemcpyAsync(d_srcs[level], img.data, memSize, cudaMemcpyHostToDevice, streams[level]);

        // fast角点提取
        dim3 block(16, 16);
        dim3 grid(divUp(Cols - 2*16, block.x),
                divUp(Rows - 2*16, block.y));
        fast<<<grid, block, 0, streams[level]>>>(d_srcs[level], Threshold, threshold_tab_cuda, &dev_counter[level], Keys, score_mats[level], Cols, Rows);

        cudaMemcpyAsync(&n[level], &dev_counter[level], sizeof(int), cudaMemcpyDeviceToHost, streams[level]);

        // 非极大值抑制
        dim3 block_nm(64);
        dim3 grid_nm(divUp(n[level], block_nm.x));
        nonmaxSuppression<<<grid_nm, block_nm, 0, streams[level]>>>(Keys, n[level], score_mats[level], &kp_counter[level], KeyPoints, Cols, Rows);

        cudaMemcpyAsync(&num[level], &kp_counter[level], sizeof(int), cudaMemcpyDeviceToHost, streams[level]);
        cudaMemcpyAsync(keys[level], KeyPoints, num[level] * sizeof(ushort3), cudaMemcpyDeviceToHost, streams[level]);
    }

    cudaDeviceSynchronize();

    // for (int level = 0;level < nlevels;level++) {
    //     cout<< "key point number: " << num[level] <<endl;
    // }

    // for (int pidx = 0; pidx < *num; pidx++)
    // {
    //     cv::KeyPoint kp;
    //     kp.pt.x = kptsLoc[pidx].x - 16;
    //     kp.pt.y = kptsLoc[pidx].y - 16;
    //     kp.response = kptsLoc[pidx].z;
    //     kp.octave = level;
    //     kp.size = 0;
    //     // vKeys.push_back(kp);
    // }

}

// 释放内存空间
void ORB_SLAM3::ORBextractor::deleteMem()
{
    if (!d_srcs.empty()) {
        for (int i=0;i<nlevels;i++) {
            cudaFree(d_srcs[i]);
            cudaFree(d_dsts[i]);
        }
    }
}
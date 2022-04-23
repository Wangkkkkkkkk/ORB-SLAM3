#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "ORBextractor.h"

using namespace std;

__device__ int nlevels;
__device__ float scaleFactor;
__device__ int isGetImageSize;

thrust::device_vector<uchar1*> d_srcs;
thrust::device_vector<uchar1*> d_dsts;

inline __device__ __host__ int divUp(int A, int B) { return (A + B - 1) / B; }

// 灰度图像高斯滤波
__global__ void gray_gaussian_filtering(uchar1* src, uchar1* dst, int width,int height)

{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < width && idy < height)
    {
        float v1 = src[idy * width + idx].x * 0.2725;
        float v2 = (src[(idy+1)*width + idx+1].x + src[(idy+1)*width + idx-1].x + src[(idy-1)*width + idx+1].x + src[(idy-1)*width + idx-1].x) * 0.0571;
        float v3 = (src[(idy+1)*width + idx].x + src[(idy-1)*width + idx].x + src[idy*width + idx+1].x + src[idy*width + idx-1].x) * 0.1248;
        dst[idy*width + idx].x = v1 + v2 + v3;
    }

    // if (idx < src.cols && idy < src.rows && src.at<int>(idy, idx) != 0)
    // {
    //     float v1 = src.at<int>(idy, idx) * 0.2725;
    //     float v2 = (src.at<int>(idy + 1, idx + 1) + src.at<int>(idy + 1, idx - 1) + src.at<int>(idy - 1, idx + 1) + src.at<int>(idy - 1, idx - 1)) * 0.0571;
    //     float v3 = (src.at<int>(idy + 1, idx) + src.at<int>(idy - 1, idx) + src.at<int>(idy, idx + 1) + src.at<int>(idy, idx - 1)) * 0.1248;
    //     dst.at<int>(idy, idx) = v1 + v2 + v3;
    // }
}

// CUDA 初始化，主要得到 金字塔层数 以及 是否得到图像尺寸设为 0
void ORB_SLAM3::ORBextractor::CUDA_Initial(int _nlevel, float _scaleFactor) {
    nlevels = _nlevel;
    scaleFactor = _scaleFactor;
    isGetImageSize = 0;

    d_srcs.resize(nlevels);
    d_dsts.resize(nlevels);
}

// 创建GPU内存空间
void ORB_SLAM3::ORBextractor::getPyramid(int level, cv::Size sz) {
    if (isGetImageSize == 0) {
        uchar1* d_src;
        uchar1* d_dst;

        int rows = sz.height;
        int cols = sz.width;
        // cout<< "rows: " << rows << " cols: " << cols <<endl;

        size_t memSize = cols * rows * sizeof(uchar1);
        cudaMalloc((void**)&d_src, memSize);
        cudaMalloc((void**)&d_dst, memSize);

        d_srcs[level] = d_src;
        d_dsts[level] = d_dst;

        // 只创建一次内存空间
        if (level == nlevels-1) {
            isGetImageSize = 1;
        }
    }
/*
    level:0 rows: 480 cols: 752
    level:1 rows: 400 cols: 627
    level:2 rows: 333 cols: 522
    level:3 rows: 278 cols: 435
    level:4 rows: 231 cols: 363
    level:5 rows: 193 cols: 302
    level:6 rows: 161 cols: 252
    level:7 rows: 134 cols: 210
*/
}

void ORB_SLAM3::ORBextractor::GBandCD_CUDA(cv::InputArray src, int level) {
    cv::Mat _src = src.getMat();
    size_t memSize = _src.cols * _src.rows * sizeof(uchar1);

    cudaMemcpy(d_srcs[level], _src.data, memSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(divUp(_src.cols, block.x),
              divUp(_src.rows, block.y));
    gray_gaussian_filtering<<<grid, block>>>(d_srcs[level], d_dsts[level], _src.cols, _src.rows);

    cudaMemcpy(_src.data, d_dsts[level], memSize, cudaMemcpyDeviceToHost);

    // cudaFree(d_src);
    // cudaFree(d_dst);
}

// 释放内存空间
void ORB_SLAM3::ORBextractor::deleteMem() {
    if (!d_srcs.empty()) {
        for (int i=0;i<nlevels;i++) {
            cudaFree(d_srcs[i]);
            cudaFree(d_dsts[i]);
        }
    }
}
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

int indexs;

__device__ int nlevels;
__device__ float scaleFactor;
__device__ int isGetImageSize;

thrust::device_vector<uchar1*> d_srcs;
thrust::device_vector<uchar1*> d_dsts;

// __device__ ushort2* Keys;

inline __device__ __host__ int divUp(int A, int B) { return (A + B - 1) / B; }

// 灰度图像高斯滤波
__global__ void gray_gaussian_filtering(uchar1* img, uchar1* dst, int width, int height)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x + 16;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y + 16;

    // uchar v = img[idy * width + idx].x; // 读取判定点的灰度值
    // printf("idx: %d idy: %d width: %d\n", idx, idy, width);
    // printf("v: %d \n", v);

    if(idx < width && idy < height)
    {
        float v1 = img[idy * width + idx].x * 0.2725;
        float v2 = (img[(idy+1)*width + idx+1].x + img[(idy+1)*width + idx-1].x + img[(idy-1)*width + idx+1].x + img[(idy-1)*width + idx-1].x) * 0.0571;
        float v3 = (img[(idy+1)*width + idx].x + img[(idy-1)*width + idx].x + img[idy*width + idx+1].x + img[idy*width + idx-1].x) * 0.1248;
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

// 灰度图像高斯滤波 shared内存不能用于图像，会超出共享内存最大值
// __global__ void gray_gaussian_filtering_shared(uchar1* src, uchar1* dst, int width,int height)

// {
//     const int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     __shared__ int tile[height][width];
//     tile[idy][idx] = src[idy * width + idx].x;
//     __syncthreads();

//     if(idx < width && idy < height)
//     {
//         float v1 = tile[idy][idx] * 0.2725;
//         float v2 = (tile[idy+1][idx+1] + tile[idy+1][idx-1] + tile[idy-1][idx+1] + tile[idy-1][idx-1]) * 0.0571;
//         float v3 = (tile[idy+1][idx] + tile[idy-1][idx] + tile[idy][idx+1] + tile[idy][idx-1]) * 0.1248;
//         dst[idy*width + idx].x = v1 + v2 + v3;
//     }
// }

// 使用类Fast 9-16 角点计算方法,检查该点是否通过 FAST 的筛选标准
__device__ bool fast_check(uchar1* img, const short idx, const short idy, const float radius, const int threshold, int width, int height)
{
    uchar v = img[idy * width + idx].x; // 读取判定点的灰度值
    // printf("idx: %d idy: %d width: %d\n", idx, idy, width);
    // printf("v: %d \n", v);
    uchar c[4];
    uchar drop_cnt = 0;
    for (uchar i = 0; i < 4; i++)
    {
        uchar values[4] = {0};
        float theta = M_PI_2 / 4 * i;
        uchar ox = radius * sin(theta);
        uchar oy = radius * cos(theta);

        c[0] = img[(idy - oy) * width + idx + ox].x;
        c[2] = img[(idy + oy) * width + idx - ox].x;

        values[0] = abs(c[0] - v) < threshold ? 1 : 0;
        values[1] = abs(c[2] - v) < threshold ? 1 : 0;
        if (values[0] & values[1])
            return false;

        c[1] = img[(idy + oy) * width + idx + ox].x;
        c[3] = img[(idy - oy) * width + idx - ox].x;
        values[2] = abs(c[1] - v) < threshold ? 1 : 0;
        values[3] = abs(c[3] - v) < threshold ? 1 : 0;
        if (values[2] & values[3])
            return false;

        drop_cnt += values[0] + values[1] + values[2] + values[3];
        if (drop_cnt >= 9)
            return false;
    }
    return true;
}

__device__ int fast_detect_16(uchar1* img, const short idx, const short idy, const int threshold, int width, int height)
{
    uchar v = img[idy * width + idx].x; // 读取判定点的灰度值

    int radiu = 3;
    uchar c[16];
    
    c[0] = img[(idy - radiu) * width + idx].x;
    c[4] = img[idy * width + idx + radiu].x;
    c[8] = img[(idy + radiu) * width + idx].x;
    c[12] = img[idy * width + idx - radiu].x;

    uchar num = 0;
    for (int i=0;i<4;i++) {
        uchar ind = i * 4;
        uchar values = abs(c[ind] - v);
        if (values > threshold) {
            num++;
        }
    }
    if (num < 3) {
        return -1;
    }

    c[1] = img[(idy - radiu) * width + idx + radiu - 2].x;
    c[2] = img[(idy - radiu + 1) * width + idx + radiu - 1].x;
    c[3] = img[(idy - radiu + 2) * width + idx + radiu].x;
    c[5] = img[(idy + radiu - 2) * width + idx + radiu].x;
    c[6] = img[(idy + radiu - 1) * width + idx + radiu - 1].x;
    c[7] = img[(idy + radiu) * width + idx + radiu - 2].x;
    c[9] = img[(idy + radiu) * width + idx - radiu + 2].x;
    c[10] = img[(idy + radiu - 1) * width + idx - radiu + 1].x;
    c[11] = img[(idy + radiu - 2) * width + idx - radiu].x;
    c[13] = img[(idy - radiu + 2) * width + idx - radiu].x;
    c[14] = img[(idy - radiu + 1) * width + idx - radiu + 1].x;
    c[15] = img[(idy - radiu) * width + idx - radiu + 2].x;

    num = 0;
    int response = 0;
    for (int i=0;i<16;i++) {
        uchar values = abs(c[i] - v);
        if (values > threshold) {
            num++;
        }
        response += values;
    }
    if (num >= 12) {
        return response;
    }
    else {
        return -1;
    }
}

// Harris Response计算
#define HARRIES_RADIUS 3
#define GAUSSIAN_SIGMA2 0.64f
#define CONSTANT_K 0.05f
__device__ float harris_response(uchar1* img, const int idx, const int idy, const float scale, int width, int height)
{
    // X, Y方向的梯度平方
    double A = 0, B = 0, C = 0;
    for (int v = -HARRIES_RADIUS; v <= HARRIES_RADIUS; v++)
        for (int u = -HARRIES_RADIUS; u <= HARRIES_RADIUS; u++)
        {
            int nx = idx + u * scale, ny = idy + v * scale;
            int scalei = int(scale);
            int Ix = (img[ny * width + nx + scalei].x - img[ny * width + nx - scalei].x) * 2 +
                        (img[(ny - scalei) * width + nx + scalei].x - img[(ny - scalei) * width + nx - scalei].x) +
                        (img[(ny + scalei) * width + nx + scalei].x - img[(ny + scalei) * width + nx - scalei].x);

            int Iy = (img[(ny + scalei) * width + nx].x - img[(ny - scalei) * width + nx].x) * 2 +
                        (img[(ny + scalei) * width + nx - scalei].x - img[(ny - scalei) * width + nx - scalei].x) +
                        (img[(ny + scalei) * width + nx + scalei].x - img[(ny - scalei) * width + nx + scalei].x);

            double gaussian_weight = exp(-(u * u + v * v) / (2 * 0.64));
            A += gaussian_weight * Ix * Ix;
            B += gaussian_weight * Iy * Iy;
            C += gaussian_weight * Ix * Iy;
        }
    double det_m = A * B - C * C;
    double trace_m = A + B;
    float score = det_m - CONSTANT_K * trace_m * trace_m;
    return score * 1e-9;
}

__global__ void fast(uchar1* img, const int threshold, int *dev_counter,
                        ushort3 *kptsLoc2D, uchar1* score_mat,
                        int width, int height) 
{
    const ushort idx = blockIdx.x * blockDim.x + threadIdx.x + 16;
    const ushort idy = blockIdx.y * blockDim.y + threadIdx.y + 16;

    if (idx < width - 16 && idy < height - 16)
    {
        // 检查是否通过 Fast 角点的收录标准
        int response = fast_detect_16(img, idx, idy, threshold, width, height);
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

__global__ void fast_grid(uchar1* img, int iniY, int maxY, int iniX, int maxX,
                            const int threshold, int *dev_counter,
                            ushort3 *kpts, uchar1* score_mat,
                            int width, int height) 
{
    const ushort idx = blockIdx.x * blockDim.x + threadIdx.x + iniX;
    const ushort idy = blockIdx.y * blockDim.y + threadIdx.y + iniY;

    if (idx < maxX && idy < maxY)
    {
        // 检查是否通过 Fast 角点的收录标准
        int response = fast_detect_16(img, idx, idy, threshold, width, height);
        if (response > 0)
        {
            int pIdx = atomicAdd(dev_counter, 1);
            if (pIdx < 50000)
            {
                kpts[pIdx] = {idx, idy, response};
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
    indexs = 0;

    nlevels = _nlevel;
    scaleFactor = _scaleFactor;
    isGetImageSize = 0;

    d_srcs.resize(nlevels);
    d_dsts.resize(nlevels);

}

int ORB_SLAM3::ORBextractor::getIndex()
{
    return indexs;
}

// 创建GPU内存空间
void ORB_SLAM3::ORBextractor::getPyramid(cv::InputArray image, int level)
{

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
    cv::Mat src = image.getMat();
    int height = src.rows;
    int width = src.cols;

    // GPU内存初始化
    if (isGetImageSize == 0) {
        uchar1* d_src;
        uchar1* d_dst;

        int rows = height;
        int cols = width;
        // cout<< "rows: " << rows << " cols: " << cols <<endl;

        size_t memSize = cols * rows * sizeof(uchar1);
        cudaMalloc((void**)&d_src, memSize);
        cudaMalloc((void**)&d_dst, memSize);

        d_srcs[level] = d_src;
        d_dsts[level] = d_dst;

        // 只创建一次内存空间
        if (level == nlevels-1) {
            isGetImageSize = 1;

            // size_t memSizes = 5000 * sizeof(ushort2);
            // cudaMalloc((void**)&Keys, memSizes);
        }
    }

    // 图像信息赋值
    // size_t memSize = width * height * sizeof(uchar1);
    // cudaMemcpy(d_srcs[level], src.data, memSize, cudaMemcpyHostToDevice);

}

void ORB_SLAM3::ORBextractor::GBandCD_CUDA(cv::InputArray image, int level)
{
    indexs++;

    cv::Mat src = image.getMat();
    // cout<< "src[16][16]: " << int(src.at<uchar>(16, 16)) <<endl;

    size_t memSize = src.cols * src.rows * sizeof(uchar1);

    cudaMemcpy(d_srcs[level], src.data, memSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(divUp(src.cols, block.x),
              divUp(src.rows, block.y));
    // dim3 block(1, 1);
    // dim3 grid(1, 1);
    gray_gaussian_filtering<<<grid, block>>>(d_srcs[level], d_dsts[level], src.cols, src.rows);

    cudaMemcpy(src.data, d_dsts[level], memSize, cudaMemcpyDeviceToHost);

    // cudaFree(d_src);
    // cudaFree(d_dst);
}

void ORB_SLAM3::ORBextractor::ExtractorPoint(cv::InputArray image, int level, vector<cv::KeyPoint> &vKeys)
{
    cv::Mat src = image.getMat();
    int Rows = src.rows;
    int Cols = src.cols;

    // size_t memSize = src.cols * src.rows * sizeof(uchar1);
    // uchar1* d_src;
    // cudaMalloc((void**)&d_src, memSize);
    // cudaMemcpy(d_src, src.data, memSize, cudaMemcpyHostToDevice);

    int iniThFAST = 25;
    int *dev_counter;
    cudaMalloc((void**)&dev_counter, sizeof(int));
    cudaMemset(dev_counter, 0, sizeof(int));

    ushort3* Keys;
    size_t memSizes = 50000 * sizeof(ushort3);
    cudaMalloc((void**)&Keys, memSizes);

    size_t memSize = Cols * Rows * sizeof(uchar1);
    cudaMemcpy(d_srcs[level], src.data, memSize, cudaMemcpyHostToDevice);

    uchar1* score_mat;
    cudaMalloc((void**)&score_mat, memSize);
    cudaMemset(score_mat, 0, memSize);

    dim3 block(16, 16);
    dim3 grid(divUp(Cols - 2*16, block.x),
              divUp(Rows - 2*16, block.y));
    fast<<<grid, block>>>(d_srcs[level], iniThFAST, dev_counter, Keys, score_mat, Cols, Rows);

    int* n;
    n = (int*)malloc(sizeof(int));
    cudaMemcpy(n, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cout<< "fast KeyPoint num: " << *n <<endl;
    *n = min(*n, 50000);

    int *kp_counter;
    cudaMalloc((void**)&kp_counter, sizeof(int));
    cudaMemset(kp_counter, 0, sizeof(int));

    ushort3* KeyPoints;
    cudaMalloc((void**)&KeyPoints, memSizes);

    dim3 block_nm(64);
    dim3 grid_nm(divUp(*n, block_nm.x));
    nonmaxSuppression<<<grid_nm, block_nm>>>(Keys, *n, score_mat, kp_counter, KeyPoints, Cols, Rows);

    int* num;
    num = (int*)malloc(sizeof(int));
    cudaMemcpy(num, kp_counter, sizeof(int), cudaMemcpyDeviceToHost);
    cout<< "nonmax KeyPoint num: " << *num <<endl;

    *num = min(*num, 50000);
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
        // cv::circle(src, pt, 3, color);
    }

    // cv::imshow("color coarse fast ", src);
    // cv::waitKey();

    // cudaThreadSynchronize();

    cudaFree(dev_counter);
    cudaFree(kp_counter);
    cudaFree(Keys);
    cudaFree(score_mat);
    cudaFree(KeyPoints);

    free(kptsLoc);
    free(n);
    free(num);
}

void ORB_SLAM3::ORBextractor::ExtractorPointGrid(cv::InputArray image, int level, vector<cv::KeyPoint> &vKeys,
                                                int minBorderX, int minBorderY, int maxBorderX, int maxBorderY,
                                                int nCols, int nRows, int wCell, int hCell)
{
    cv::Mat src = image.getMat();

    int iniThFAST = 50;
    int *dev_counter;
    cudaMalloc((void**)&dev_counter, sizeof(int));
    cudaMemset(dev_counter, 0, sizeof(int));

    ushort3* Keys;
    size_t memSizes = 10000 * sizeof(ushort3);
    cudaMalloc((void**)&Keys, memSizes);

    size_t memSize = src.cols * src.rows * sizeof(uchar1);
    cudaMemcpy(d_srcs[level], src.data, memSize, cudaMemcpyHostToDevice);

    uchar1* score_mat;
    cudaMalloc((void**)&score_mat, memSize);
    cudaMemset(score_mat, 0, memSize);

    // cudaStream_t stream[nRows * nCols];

    // for (int i=0;i<nRows;i++) {
    //     for (int j=0;j<nCols;j++) {
    //         cudaStreamCreate(&stream[i * nRows + j]);
    //     }
    // }

    //开始遍历图像网格，还是以行开始遍历的
    for(int i=0; i<nRows; i++)
    {
        //计算当前网格初始行坐标
        const int iniY =minBorderY+i*hCell;
        //计算当前网格最大的行坐标，这里的+6=+3+3，即考虑到了多出来3是为了cell边界像素进行FAST特征点提取用
        //前面的EDGE_THRESHOLD指的应该是提取后的特征点所在的边界，所以minBorderY是考虑了计算半径时候的图像边界
        //目测一个图像网格的大小是25*25啊
        int maxY = iniY+hCell+6;

        //如果初始的行坐标就已经超过了有效的图像边界了，这里的“有效图像”是指原始的、可以提取FAST特征点的图像区域
        if(iniY>=maxBorderY-3)
            //那么就跳过这一行
            continue;
        //如果图像的大小导致不能够正好划分出来整齐的图像网格，那么就要委屈最后一行了
        if(maxY>maxBorderY)
            maxY = maxBorderY;

        //开始列的遍历
        for(int j=0; j<nCols; j++)
        {
            //计算初始的列坐标
            const int iniX =minBorderX+j*wCell;
            //计算这列网格的最大列坐标，+6的含义和前面相同
            int maxX = iniX+wCell+6;

            //判断坐标是否在图像中
            if(iniX>=maxBorderX-3)
                continue;

            //如果最大坐标越界那么委屈一下
            if(maxX>maxBorderX)
                maxX = maxBorderX;

            dim3 block(16, 16);
            dim3 grid(divUp(maxX - iniX, block.x),
                    divUp(maxY - iniY, block.y));
            
            fast_grid<<<grid, block>>>(d_srcs[level], iniY, maxY, iniX, maxX,
                                        iniThFAST, dev_counter, Keys, score_mat,
                                        src.cols, src.rows);

        }
    }

    int* n;
    n = (int*)malloc(sizeof(int));
    cudaMemcpy(n, dev_counter, sizeof(int), cudaMemcpyDeviceToHost);
    // cout<< "KeyPoint num: " << *n <<endl;



    ushort2* kptsLoc = (ushort2*)malloc(*n * sizeof(ushort2));
    cudaMemcpy(kptsLoc, Keys, *n * sizeof(ushort2), cudaMemcpyDeviceToHost);

    // for (int pidx = 0; pidx < *n; pidx++)
    // {
    //     static cv::Point2i pt;
    //     pt.x = kptsLoc[pidx].x;
    //     pt.y = kptsLoc[pidx].y;
    //     // cout<< "x: " << pt.x << " y: " << pt.y <<endl;
    //     cv::Scalar color(0, 255, 0);
    //     cv::circle(src, pt, 3, color);
    // }

    // cv::imshow("color coarse fast ", src);
    // cv::waitKey();

    // for (int i=0;i<nRows;i++) {
    //     for (int j=0;j<nCols;j++) {
    //         cudaStreamDestroy(stream[i * nRows + j]);
    //     }
    // }

    cudaFree(dev_counter);
    cudaFree(Keys);
    cudaFree(score_mat);

    free(n);
    free(kptsLoc);
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
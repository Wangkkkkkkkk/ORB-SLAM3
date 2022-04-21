#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/core/cuda_stream_accessor.hpp"

#include "ORBextractor.h"

using namespace std;

inline __device__ __host__ int divUp(int A, int B) { return (A + B - 1) / B; }

// 灰度图像高斯滤波
__global__ void gray_gaussian_filtering(uchar1* src, uchar1* dst, int width,int height)

{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if(idx < width && idy < height)
    {
        // printf("src: %d \n", src[idy * width + idx].x);
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

void ORB_SLAM3::ORBextractor::GaussianBlur_CUDA(cv::InputArray src) {
    // cout<< "in CUDA_Acc.cu" <<endl;

    cv::Mat _src = src.getMat();
    // cout<< _src <<endl;

    size_t memSize = _src.cols*_src.rows*sizeof(uchar1);
    uchar1* d_src = NULL;
    uchar1* d_dst = NULL;

    cudaMalloc((void**)&d_src, memSize);
    cudaMalloc((void**)&d_dst, memSize);
    cudaMemcpy(d_src, _src.data, memSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(divUp(_src.cols, block.x),
              divUp(_src.rows, block.y));
    gray_gaussian_filtering<<<grid, block>>>(d_src, d_dst, _src.cols, _src.rows);

    cudaThreadSynchronize();

    cudaMemcpy(_src.data,d_dst,memSize,cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}
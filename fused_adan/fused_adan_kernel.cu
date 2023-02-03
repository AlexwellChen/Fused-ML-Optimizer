/* Copyright 2021 The LightSeq Team
   Copyright NVIDIA/apex
   This apex_adam_cuda_kernel is adapted from NVIDIA/apex
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cmath>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/TensorUtils.h"
#include "ATen/cuda/CUDAContext.h"
#include "ATen/cuda/detail/IndexUtils.cuh"
#include "ATen/cuda/Exceptions.h"
#include "../include/fused_adan_kernel.h"
#include "../include/multi_tensor_apply.cuh"

// void adan(at::Tensor& p, at::Tensor& p_copy, at::Tensor& g, at::Tensor& exp_avg, 
//           at::Tensor& exp_avg_sq, at::Tensor& exp_avg_diff,
//           at::Tensor& pre_g, float beta1, float beta2, float beta3, 
//           float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
//           float lr, float decay, float eps, bool no_prox, float grad_scale);

template <typename T, typename GRAD_T>
__global__ void adan_cuda_kernel(
    T* __restrict__ p,
    GRAD_T* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    const GRAD_T* __restrict__ g, T* __restrict__ exp_avg, T* __restrict__ exp_avg_sq, T* __restrict__ exp_avg_diff,
    const GRAD_T* __restrict__ pre_g, const float b1, const float b2, const float b3, 
    const float bias_correction1, const float bias_correction2, const float bias_correction3_sqrt,
    const float lr, const float decay, const float eps, const bool no_prox, const float grad_scale
    ){
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id >= total_size) return;

    T scaled_grad = g[global_id] / grad_scale;

    diff = scaled_grad - pre_g[global_id];
    update = scaled_grad + b2 * diff;

    // exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
    exp_avg[global_id] = b1 * exp_avg[global_id] + (1 - b1) * scaled_grad;

    // exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
    exp_avg_diff[global_id] = b2 * exp_avg_diff[global_id] + (1 - b2) * diff;

    // exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t
    exp_avg_sq[global_id] = b3 * exp_avg_sq[global_id] + (1 - b3) * update * update;

    // denom = ((exp_avg_sq).sqrt() / bias_correction3_sqrt).add_(eps)
    // update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)
    float denom;
    denom = sqrtf(exp_avg_sq[global_id]) / bias_correction3_sqrt + eps;
    update = (exp_avg[global_id] / bias_correction1 + b2 * exp_avg_diff[global_id] / bias_correction2) / denom;
    
    if (no_prox){
        // param.mul_(1 - lr * weight_decay)
        // param.add_(update, alpha=-lr)
        p[global_id] = p[global_id] * (1 - lr * decay) + update * (-lr);
    }else{
        // param.add_(update, alpha=-lr)
        // param.div_(1 + lr * weight_decay)
        p[global_id] = p[global_id] + update * (-lr) / (1 + lr * decay);
    } 
    if (p_copy != NULL) p_copy[global_id] = (GRAD_T)p[global_id];
}

template <>
__global__ void adan_cuda_kernel<float, float>(
    float* __restrict__ p,
    float* __restrict__ p_copy,  // For mixed precision training, pass NULL if
                                  // not needed
    const float* __restrict__ g, float* __restrict__ exp_avg, float* __restrict__ exp_avg_sq, float* __restrict__ exp_avg_diff,
    const float* __restrict__ pre_g, const float b1, const float b2, const float b3, 
    const float bias_correction1, const float bias_correction2, const float bias_correction3_sqrt,
    const float lr, const float decay, const float eps, const bool no_prox, const float grad_scale){

        int global_id = blockIdx.x * blockDim.x + threadIdx.x;

        if (global_id * 4 >= total_size) return;

        const float4* g4_ptr = reinterpret_cast<const float4*>(g);
        const float4* pre_g4_ptr = reinterpret_cast<const float4*>(pre_g);
        float4* exp_avg4_ptr = reinterpret_cast<float4*>(exp_avg);
        float4* exp_avg_sq4_ptr = reinterpret_cast<float4*>(exp_avg_sq);
        float4* exp_avg_diff4_ptr = reinterpret_cast<float4*>(exp_avg_diff);

        const float4 g4 = g4_ptr[global_id];
        const float4 pre_g4 = pre_g4_ptr[global_id];
        float4 exp_avg4 = exp_avg4_ptr[global_id];
        float4 exp_avg_sq4 = exp_avg_sq4_ptr[global_id];
        float4 exp_avg_diff4 = exp_avg_diff4_ptr[global_id];

        float4 new_exp_avg4;
        float4 new_exp_avg_sq4;
        float4 new_exp_avg_diff4;

        float scaled_grad1 = g4.x / grad_scale;
        float scaled_grad2 = g4.y / grad_scale;
        float scaled_grad3 = g4.z / grad_scale;
        float scaled_grad4 = g4.w / grad_scale;

        float diff1 = scaled_grad1 - pre_g4.x;
        float diff2 = scaled_grad2 - pre_g4.y;
        float diff3 = scaled_grad3 - pre_g4.z;
        float diff4 = scaled_grad4 - pre_g4.w;

        float update1 = scaled_grad1 + b2 * diff1;
        float update2 = scaled_grad2 + b2 * diff2;
        float update3 = scaled_grad3 + b2 * diff3;
        float update4 = scaled_grad4 + b2 * diff4;

        new_exp_avg4.x = b1 * exp_avg4.x + (1 - b1) * scaled_grad1;
        new_exp_avg4.y = b1 * exp_avg4.y + (1 - b1) * scaled_grad2;
        new_exp_avg4.z = b1 * exp_avg4.z + (1 - b1) * scaled_grad3;
        new_exp_avg4.w = b1 * exp_avg4.w + (1 - b1) * scaled_grad4;

        new_exp_avg_sq4.x = b3 * exp_avg_sq4.x + (1 - b3) * update1 * update1;
        new_exp_avg_sq4.y = b3 * exp_avg_sq4.y + (1 - b3) * update2 * update2;
        new_exp_avg_sq4.z = b3 * exp_avg_sq4.z + (1 - b3) * update3 * update3;
        new_exp_avg_sq4.w = b3 * exp_avg_sq4.w + (1 - b3) * update4 * update4;

        new_exp_avg_diff4.x = b2 * exp_avg_diff4.x + (1 - b2) * diff1;
        new_exp_avg_diff4.y = b2 * exp_avg_diff4.y + (1 - b2) * diff2;
        new_exp_avg_diff4.z = b2 * exp_avg_diff4.z + (1 - b2) * diff3;
        new_exp_avg_diff4.w = b2 * exp_avg_diff4.w + (1 - b2) * diff4;

        float4 denom4;
        denom4.x = sqrt(new_exp_avg_sq4.x - new_exp_avg_diff4.x * new_exp_avg_diff4.x / b2) + eps;
        denom4.y = sqrt(new_exp_avg_sq4.y - new_exp_avg_diff4.y * new_exp_avg_diff4.y / b2) + eps;
        denom4.z = sqrt(new_exp_avg_sq4.z - new_exp_avg_diff4.z * new_exp_avg_diff4.z / b2) + eps;
        denom4.w = sqrt(new_exp_avg_sq4.w - new_exp_avg_diff4.w * new_exp_avg_diff4.w / b2) + eps;

        // update = (exp_avg[global_id] / bias_correction1 + b2 * exp_avg_diff[global_id] / bias_correction2) / denom;
        update1 = (new_exp_avg4.x / bias_correction1 + b2 * new_exp_avg_diff4.x / bias_correction2) / denom4.x;
        update2 = (new_exp_avg4.y / bias_correction1 + b2 * new_exp_avg_diff4.y / bias_correction2) / denom4.y;
        update3 = (new_exp_avg4.z / bias_correction1 + b2 * new_exp_avg_diff4.z / bias_correction2) / denom4.z;
        update4 = (new_exp_avg4.w / bias_correction1 + b2 * new_exp_avg_diff4.w / bias_correction2) / denom4.w;

        if (no_prox){
            // p[global_id] = p[global_id] * (1 - lr * decay) + update * (-lr);
            new_p4.x = p4.x * (1 - lr * decay) + update1 * (-lr);
            new_p4.y = p4.y * (1 - lr * decay) + update2 * (-lr);
            new_p4.z = p4.z * (1 - lr * decay) + update3 * (-lr);
            new_p4.w = p4.w * (1 - lr * decay) + update4 * (-lr);
        }else{
            // p[global_id] = p[global_id] + update * (-lr) / (1 + lr * decay);
            new_p4.x = p4.x + update1 * (-lr) / (1 + lr * decay);
            new_p4.y = p4.y + update2 * (-lr) / (1 + lr * decay);
            new_p4.z = p4.z + update3 * (-lr) / (1 + lr * decay);
            new_p4.w = p4.w + update4 * (-lr) / (1 + lr * decay);
        }   

        p4_ptr[global_id] = new_p4;
        exp_avg4_ptr[global_id] = new_exp_avg4;
        exp_avg_sq4_ptr[global_id] = new_exp_avg_sq4;
        exp_avg_diff4_ptr[global_id] = new_exp_avg_diff4;
}

void fused_adan_cuda(at::Tensor& p, at::Tensor& p_copy, at::Tensor& g, at::Tensor& exp_avg, 
          at::Tensor& exp_avg_sq, at::Tensor& exp_avg_diff,
          at::Tensor& pre_g, float beta1, float beta2, float beta3, 
          float bias_correction1, float bias_correction2, float bias_correction3_sqrt, 
          float lr, float decay, float eps, bool no_prox, float grad_scale){
    // Get tensor size
    int total_size = p.numel();
    AT_ASSERTM(at::cuda::detail::canUse32BitIndexMath(p),
              "parameter tensor is too large to be indexed with int32");
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    if (g.scalar_type() == at::ScalarType::Half) {
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim);
        const dim3 blocks(grid_dim);
        // all other values should be fp32 for half gradients
        AT_ASSERTM(p.scalar_type() == at::ScalarType::Float,
                  "expected parameter to be of float type");
        // dispatch is done on the gradient type
        using namespace at;  // prevents "toString is undefined" errors
        DISPATCH_FLOAT_AND_HALF(
            g.scalar_type(), 0, "adan_cuda_kernel",
            using accscalar_t = at::acc_type<scalar_t_0, true>;
            adan_cuda_kernel<accscalar_t, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.DATA_PTR<accscalar_t>(),
                p_copy.numel() ? p_copy.DATA_PTR<scalar_t_0>() : NULL,
                g.DATA_PTR<scalar_t_0>(), exp_avg.DATA_PTR<accscalar_t>(), exp_avg_sq.DATA_PTR<accscalar_t>(),exp_avg_diff.DATA_PTR<accscalar_t>(), 
                pre_g.DATA_PTR<scalar_t_0>(), 
                beta1, beta2, beta3, bias_correction1, bias_correction2, bias_correction3_sqrt, 
                lr, decay, eps, no_prox, grad_scale
                );
            );
    } else {
        using namespace at;
        const int block_dim = 1024;
        int grid_dim = ((total_size + block_dim - 1) / block_dim) >> 2;
        if (grid_dim == 0) grid_dim = 1;
        const dim3 blocks(grid_dim);
        DISPATCH_DOUBLE_AND_FLOAT(
            g.scalar_type(), 0, "adan_cuda_kernel",
            adan_cuda_kernel<scalar_t_0, scalar_t_0>
            <<<blocks, block_dim, 0, stream>>>(
                p.DATA_PTR<scalar_t_0>(),
                NULL,
                g.DATA_PTR<scalar_t_0>(), exp_avg.DATA_PTR<scalar_t_0>(), exp_avg_sq.DATA_PTR<scalar_t_0>(),exp_avg_diff.DATA_PTR<scalar_t_0>(), 
                pre_g.DATA_PTR<scalar_t_0>(), 
                beta1, beta2, beta3, bias_correction1, bias_correction2, bias_correction3_sqrt, 
                lr, decay, eps, no_prox, grad_scale
            );
        );
    }
    AT_CUDA_CHECK(cudaGetLastError());
}


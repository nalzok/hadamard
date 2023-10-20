/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef FWT_KERNEL_CUH
#define FWT_KERNEL_CUH
#ifndef fwt_kernel_cuh
#define fwt_kernel_cuh

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include "helper_cuda.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <torch/types.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert[%s:%d]: %s\n", file, line, cudaGetErrorString(code));
    if (abort) exit(code);
  }
}

// roundup(x, y) is equal to x rounded up to the nearest multiple of y.
__device__ static inline int roundup(int x, int y) {
  return y * ((x + y - 1) / y);
}

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define MAX_SMEM_LOG2SIZE 13


template <typename scalar_t>
__global__ static void fwtBatch1Kernel(scalar_t *d_Output, scalar_t *d_Input, int batchSize, int vecSize) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();

  // 2 ** 13 bytes == 8KB -- maximum s_data[] size for A6000
  extern __shared__ __align__(8) unsigned char sdata_raw[];     // align to 8 bytes for double
  scalar_t *s_data = reinterpret_cast<scalar_t*>(sdata_raw);

  for (int batch = blockIdx.x; batch < roundup(batchSize, gridDim.x); batch += gridDim.x) {
    if (batch >= batchSize) {
      for (int stride = vecSize >> 2; stride > 0; stride >>= 2) {
        cg::sync(cta);
      }
      if (__popc(vecSize - 1) % 2 == 1) {
        cg::sync(cta);
      }
      cg::sync(cta);
      break;
    }

    scalar_t *d_Src = d_Input + batch * vecSize;
    scalar_t *d_Dst = d_Output + batch * vecSize;

    for (int pos = threadIdx.x; pos < vecSize; pos += blockDim.x) {
      s_data[pos] = d_Src[pos];
    }

    // Main radix-4 stages
    for (int stride = vecSize >> 2; stride > 0; stride >>= 2) {
      for (int pos = threadIdx.x; pos < roundup(vecSize/4, blockDim.x); pos += blockDim.x) {
        if (pos >= vecSize/4) {
          cg::sync(cta);
          break;
        }

        int lo = pos & (stride - 1);
        int i0 = ((pos - lo) << 2) + lo;
        int i1 = i0 + stride;
        int i2 = i1 + stride;
        int i3 = i2 + stride;

        cg::sync(cta);
        scalar_t D0 = s_data[i0];
        scalar_t D1 = s_data[i1];
        scalar_t D2 = s_data[i2];
        scalar_t D3 = s_data[i3];

        scalar_t T;
        T = D0;
        D0 = D0 + D2;
        D2 = T - D2;
        T = D1;
        D1 = D1 + D3;
        D3 = T - D3;
        T = D0;
        s_data[i0] = D0 + D1;
        s_data[i1] = T - D1;
        T = D2;
        s_data[i2] = D2 + D3;
        s_data[i3] = T - D3;
      }
    }

    // Do single radix-2 stage for odd power of two
    if (__popc(vecSize - 1) % 2 == 1) {
      cg::sync(cta);

      for (int pos = threadIdx.x; pos < vecSize/2; pos += blockDim.x) {
        int i0 = pos << 1;
        int i1 = i0 + 1;

        scalar_t D0 = s_data[i0];
        scalar_t D1 = s_data[i1];
        s_data[i0] = D0 + D1;
        s_data[i1] = D0 - D1;
      }
    }

    cg::sync(cta);

    for (int pos = threadIdx.x; pos < vecSize; pos += blockDim.x) {
      d_Dst[pos] = s_data[pos];
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-8 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ static void fwtBatch3Kernel(scalar_t *d_Output, scalar_t *d_Input, int batchSize, int vecSize, int stride) {
  const int tidY = blockIdx.y * blockDim.y + threadIdx.y;
  for (int batch = tidY; batch < batchSize; batch += blockDim.y * gridDim.y) {
    scalar_t *d_Src = d_Input + batch * vecSize;
    scalar_t *d_Dst = d_Output + batch * vecSize;

    const int tidX = blockIdx.x * blockDim.x + threadIdx.x;
    for (int pos = tidX; pos < vecSize/8; pos += blockDim.x * gridDim.x) {
      int lo = pos & (stride - 1);
      int i0 = ((pos - lo) << 3) + lo;
      int i1 = i0 + stride;
      int i2 = i1 + stride;
      int i3 = i2 + stride;
      int i4 = i3 + stride;
      int i5 = i4 + stride;
      int i6 = i5 + stride;
      int i7 = i6 + stride;

      scalar_t D0 = d_Src[i0];
      scalar_t D1 = d_Src[i1];
      scalar_t D2 = d_Src[i2];
      scalar_t D3 = d_Src[i3];
      scalar_t D4 = d_Src[i4];
      scalar_t D5 = d_Src[i5];
      scalar_t D6 = d_Src[i6];
      scalar_t D7 = d_Src[i7];

      scalar_t T;
      T = D0;
      D0 = D0 + D4;
      D4 = T - D4;
      T = D1;
      D1 = D1 + D5;
      D5 = T - D5;
      T = D2;
      D2 = D2 + D6;
      D6 = T - D6;
      T = D3;
      D3 = D3 + D7;
      D7 = T - D7;

      T = D0;
      scalar_t E0 = D0 + D2;
      scalar_t E2 = T - D2;
      T = D1;
      scalar_t E1 = D1 + D3;
      scalar_t E3 = T - D3;
      T = D4;
      scalar_t E4 = D4 + D6;
      scalar_t E6 = T - D6;
      T = D5;
      scalar_t E5 = D5 + D7;
      scalar_t E7 = T - D7;

      T = E0;
      d_Dst[i0] = E0 + E1;
      d_Dst[i1] = T - E1;
      T = E2;
      d_Dst[i2] = E2 + E3;
      d_Dst[i3] = T - E3;
      T = E4;
      d_Dst[i4] = E4 + E5;
      d_Dst[i5] = T - E5;
      T = E6;
      d_Dst[i6] = E6 + E7;
      d_Dst[i7] = T - E7;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
__host__ extern void fwtBatchGPU(torch::Tensor& d_Data, int batchSize, int log2N) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, d_Data.scalar_type(), "fwtBatchGPU", [&] {
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    scalar_t *data_ptr = d_Data.data_ptr<scalar_t>();

    const int origBatchSize = batchSize;
    const int origVecSize = 1 << log2N;

    dim3 dimGrid(128, 2, 1);
    dim3 dimBlock(256, 2, 1);

    int vecSize = origVecSize;
    for (; vecSize * sizeof(scalar_t) > (1 << MAX_SMEM_LOG2SIZE); batchSize <<= 3, vecSize >>= 3) {
      fwtBatch3Kernel<<<dimGrid, dimBlock, 0, stream>>>(data_ptr, data_ptr, origBatchSize, origVecSize, vecSize/8);
      gpuErrchk(cudaPeekAtLastError());
      gpuErrchk(cudaDeviceSynchronize());
    }

    fwtBatch1Kernel<<<128, std::min(std::max(1, vecSize/4), 256), vecSize * sizeof(scalar_t), stream>>>(data_ptr, data_ptr, batchSize, vecSize);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  });
}


////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
template <typename scalar_t>
__global__ static void modulateKernel(scalar_t *d_A, scalar_t *d_B, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numThreads = blockDim.x * gridDim.x;

  for (int pos = tid; pos < N; pos += numThreads) {
    d_A[pos] *= d_B[pos] / N;
  }
}


// Interface to modulateKernel()
__host__ extern void modulateGPU(torch::Tensor &d_A, torch::Tensor &d_B, int N) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, d_A.scalar_type(), "fwtBatchGPU", [&] {
    modulateKernel<<<128, 256>>>(d_A.data_ptr<scalar_t>(), d_B.data_ptr<scalar_t>(), N);
  });
}

#endif
#endif

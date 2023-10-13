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

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// Elementary(for vectors less than elementary size) in-shared memory
// combined radix-2 + radix-4 Fast Walsh Transform
///////////////////////////////////////////////////////////////////////////////
#define ELEMENTARY_LOG2SIZE 11

__global__ void fwtBatch1Kernel(__half *d_Output, __half *d_Input, int log2N) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  const int N = 1 << log2N;
  const int base = blockIdx.x << log2N;

  //(2 ** 11) * 4 bytes == 8KB -- maximum s_data[] size for G80
  extern __shared__ __half s_data[];
  __half *d_Src = d_Input + base;
  __half *d_Dst = d_Output + base;

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    s_data[pos] = d_Src[pos];
  }

  // Main radix-4 stages
  const int pos = threadIdx.x;

  for (int stride = N >> 2; stride > 0; stride >>= 2) {
    int lo = pos & (stride - 1);
    int i0 = ((pos - lo) << 2) + lo;
    int i1 = i0 + stride;
    int i2 = i1 + stride;
    int i3 = i2 + stride;

    cg::sync(cta);
    __half D0 = s_data[i0];
    __half D1 = s_data[i1];
    __half D2 = s_data[i2];
    __half D3 = s_data[i3];

    __half T;
    T = D0;
    D0 = __hadd(D0, D2);
    D2 = __hsub(T, D2);
    T = D1;
    D1 = __hadd(D1, D3);
    D3 = __hsub(T, D3);
    T = D0;
    s_data[i0] = __hadd(D0, D1);
    s_data[i1] = __hsub(T, D1);
    T = D2;
    s_data[i2] = __hadd(D2, D3);
    s_data[i3] = __hsub(T, D3);
  }

  // Do single radix-2 stage for odd power of two
  if (log2N & 1) {
    cg::sync(cta);

    for (int pos = threadIdx.x; pos < N / 2; pos += blockDim.x) {
      int i0 = pos << 1;
      int i1 = i0 + 1;

      __half D0 = s_data[i0];
      __half D1 = s_data[i1];
      s_data[i0] = __hadd(D0, D1);
      s_data[i1] = __hsub(D0, D1);
    }
  }

  cg::sync(cta);

  for (int pos = threadIdx.x; pos < N; pos += blockDim.x) {
    d_Dst[pos] = s_data[pos];
  }
}

////////////////////////////////////////////////////////////////////////////////
// Single in-global memory radix-4 Fast Walsh Transform pass
// (for strides exceeding elementary vector size)
////////////////////////////////////////////////////////////////////////////////
__global__ void fwtBatch2Kernel(__half *d_Output, __half *d_Input, int stride) {
  const int pos = blockIdx.x * blockDim.x + threadIdx.x;
  const int N = blockDim.x * gridDim.x * 4;

  __half *d_Src = d_Input + blockIdx.y * N;
  __half *d_Dst = d_Output + blockIdx.y * N;

  int lo = pos & (stride - 1);
  int i0 = ((pos - lo) << 2) + lo;
  int i1 = i0 + stride;
  int i2 = i1 + stride;
  int i3 = i2 + stride;

  __half D0 = d_Src[i0];
  __half D1 = d_Src[i1];
  __half D2 = d_Src[i2];
  __half D3 = d_Src[i3];

  __half T;
  T = D0;
  D0 = __hadd(D0, D2);
  D2 = __hsub(T, D2);
  T = D1;
  D1 = __hadd(D1, D3);
  D3 = __hsub(T, D3);
  T = D0;
  d_Dst[i0] = __hadd(D0, D1);
  d_Dst[i1] = __hsub(T, D1);
  T = D2;
  d_Dst[i2] = __hadd(D2, D3);
  d_Dst[i3] = __hsub(T, D3);
}

////////////////////////////////////////////////////////////////////////////////
// Put everything together: batched Fast Walsh Transform CPU front-end
////////////////////////////////////////////////////////////////////////////////
void fwtBatchGPU(__half *d_Data, int M, int log2N) {
  const int THREAD_N = 256;

  int N = 1 << log2N;
  dim3 grid((1 << log2N) / (4 * THREAD_N), M, 1);

  for (; log2N > ELEMENTARY_LOG2SIZE; log2N -= 2, N >>= 2, M <<= 2) {
    fwtBatch2Kernel<<<grid, THREAD_N>>>(d_Data, d_Data, N / 4);
    getLastCudaError("fwtBatch2Kernel() execution failed\n");
  }

  fwtBatch1Kernel<<<M, N / 4, N * sizeof(__half)>>>(d_Data, d_Data, log2N);
  getLastCudaError("fwtBatch1Kernel() execution failed\n");
}

////////////////////////////////////////////////////////////////////////////////
// Modulate two arrays
////////////////////////////////////////////////////////////////////////////////
__global__ void modulateKernel(__half *d_A, __half *d_B, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int numThreads = blockDim.x * gridDim.x;

  for (int pos = tid; pos < N; pos += numThreads) {
    d_A[pos] = __hmul(d_A[pos], __hdiv(d_B[pos], __int2half_rn(N)));
  }
}

// Interface to modulateKernel()
void modulateGPU(__half *d_A, __half *d_B, int N) {
  modulateKernel<<<128, 256>>>(d_A, d_B, N);
}

#endif
#endif

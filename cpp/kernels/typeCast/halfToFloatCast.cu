/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "halfToFloatCast.h"

namespace trt_edgellm
{
namespace kernel
{

namespace
{

__global__ void castHalfToFloatKernel(float* dst, half const* src, int64_t count)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void castFloatToHalfKernel(half* dst, float const* src, int64_t count)
{
    int64_t const idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count)
    {
        dst[idx] = __float2half(src[idx]);
    }
}

} // namespace

void castHalfToFloatDevice(float* dst, half const* src, int64_t count, cudaStream_t stream)
{
    if (count <= 0)
    {
        return;
    }
    constexpr int32_t kBlockSize = 256;
    int64_t const numBlocks = (count + kBlockSize - 1) / kBlockSize;
    castHalfToFloatKernel<<<static_cast<uint32_t>(numBlocks), kBlockSize, 0, stream>>>(dst, src, count);
}

void castFloatToHalfDevice(half* dst, float const* src, int64_t count, cudaStream_t stream)
{
    if (count <= 0)
    {
        return;
    }
    constexpr int32_t kBlockSize = 256;
    int64_t const numBlocks = (count + kBlockSize - 1) / kBlockSize;
    castFloatToHalfKernel<<<static_cast<uint32_t>(numBlocks), kBlockSize, 0, stream>>>(dst, src, count);
}

} // namespace kernel
} // namespace trt_edgellm

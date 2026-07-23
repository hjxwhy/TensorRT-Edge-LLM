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

#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace trt_edgellm
{
namespace kernel
{

/*!
 * @brief Cast a device-resident FP16 buffer to FP32, elementwise, on the GPU.
 *
 * Replaces a scalar host-side __half2float loop (previously done in the pybind layer after a
 * device-to-host copy) with a parallel GPU kernel run before the device-to-host copy -- the copy
 * then transfers the already-FP32 result directly. For a [1, 487, 2560] hidden_states tensor
 * (~1.25M elements), the scalar host loop measured as a multi-ms-scale bottleneck; this kernel
 * runs in well under 1ms for the same size.
 *
 * @param dst Output FP32 device buffer, at least `count` elements.
 * @param src Input FP16 device buffer, `count` elements.
 * @param count Number of elements to cast.
 * @param stream CUDA stream.
 */
void castHalfToFloatDevice(float* dst, half const* src, int64_t count, cudaStream_t stream);

/*!
 * @brief Cast a device-resident FP32 buffer to FP16, elementwise, on the GPU.
 *
 * Counterpart of castHalfToFloatDevice() for the opposite direction: replaces a scalar host-side
 * __float2half loop (previously done in the pybind layer, converting a caller-supplied FP32 numpy
 * array into the FP16 host buffer runBackboneRawForward() expected) with a device-to-device cast
 * run after an H2D copy of the raw FP32 bytes -- no per-element host-side loop at all.
 *
 * @param dst Output FP16 device buffer, at least `count` elements.
 * @param src Input FP32 device buffer, `count` elements.
 * @param count Number of elements to cast.
 * @param stream CUDA stream.
 */
void castFloatToHalfDevice(half* dst, float const* src, int64_t count, cudaStream_t stream);

} // namespace kernel
} // namespace trt_edgellm

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

#include "common/cudaUtils.h"
#include "common/tensor.h"
#include <memory>
#include <string>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

/*!
 * @brief Image data container
 *
 * Wraps image tensor loaded with stbi. Expected shape: [height, width, channels].
 * Channels is typically 3 for RGB images.
 */
class ImageData
{
public:
    std::shared_ptr<rt::Tensor> buffer; //!< Image data buffer
    int64_t width;                      //!< Image width
    int64_t height;                     //!< Image height
    int64_t channels;                   //!< Number of channels (e.g., 3 for RGB)

    /*!
     * @brief Default constructor (creates uninitialized ImageData)
     */
    ImageData() noexcept = default;

    /*!
     * @brief Construct image data
     * @param data Image tensor
     * @throws std::runtime_error if tensor content not UINT8, or tensor shape not 3D, or number of channels not 3
     */
    ImageData(rt::Tensor&& data);

    //! @brief Get raw image data pointer
    //! @return Pointer to image data
    unsigned char* data() const noexcept;
};

/*!
 * @brief Load image from file
 * @param path Path to image file
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from file, or memory allocation fails
 */
ImageData loadImageFromFile(std::string const& path);

/*!
 * @brief Load image from memory
 * @param data Pointer to image data in memory
 * @param size Size of image data in bytes
 * @return Loaded image data
 * @throws std::runtime_error if image cannot be loaded from memory, or memory allocation fails
 */
ImageData loadImageFromMemory(unsigned char const* data, size_t size);

/*!
 * @brief Build image data from already-decoded raw RGB pixels (no image codec involved)
 * @param data Pointer to tightly-packed [height, width, channels] UINT8 RGB pixels
 * @param height Image height
 * @param width Image width
 * @param channels Number of channels (must be 3)
 * @return Loaded image data
 * @throws std::runtime_error if channels != 3, or memory allocation fails
 */
ImageData loadImageFromRaw(unsigned char const* data, int64_t height, int64_t width, int64_t channels);

//! @brief Interpolation filter for :func:`resizeImage`.
enum class InterpolationMode
{
    kLINEAR,  //!< Bilinear
    kBICUBIC, //!< Catmull-Rom cubic
};

/*!
 * @brief Resize image into pre-allocated buffer
 * @param image Source image
 * @param resizedImage Output buffer (will be reshaped to target dimensions)
 * @param newWidth Target width
 * @param newHeight Target height
 * @param mode Interpolation filter
 * @throws std::runtime_error if image buffer cannot be reshaped
 */
void resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode);

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm

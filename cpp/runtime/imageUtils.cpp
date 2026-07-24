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

#include "runtime/imageUtils.h"
#include "common/checkMacros.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_resize2.h>

namespace trt_edgellm
{
namespace rt
{
namespace imageUtils
{

ImageData::ImageData(rt::Tensor&& data)
{
    check::check(data.getDataType() == nvinfer1::DataType::kUINT8, "Image data must be UINT8");
    check::check(data.getShape().getNumDims() == 3, "Image data must have 3 dimensions");
    check::check(data.getShape()[2] == 3, "Image data must have 3 channels");

    // Image data is expected to have shape [height, width, channels]
    height = data.getShape()[0];
    width = data.getShape()[1];
    channels = data.getShape()[2];
    buffer = std::make_shared<rt::Tensor>(std::move(data));
}

unsigned char* ImageData::data() const noexcept
{
    return buffer ? buffer->dataPointer<unsigned char>() : nullptr;
}

ImageData loadImageFromFile(std::string const& path)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load(path.c_str(), &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image: " + path + " - " + std::string(stbi_failure_reason()));

    rt::Tensor imgTensor{};
    // Need to handle the logic where space allocation for image tensor failed. We need to free the image data and
    // throw an exception.
    try
    {
        imgTensor = rt::Tensor({height, width, desiredChannels}, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8,
            "imageUtils::loadImageFromFile::imgTensor");
    }
    catch (std::exception const& e)
    {
        stbi_image_free(imageData);
        throw std::runtime_error("Failed to allocate space for image tensor: " + std::string(e.what()));
    }
    memcpy(imgTensor.dataPointer<unsigned char>(), imageData, width * height * desiredChannels);
    stbi_image_free(imageData);
    return ImageData(std::move(imgTensor));
}

namespace
{
//! Copy `height*width*channels` bytes of raw RGB pixels into a persistent, reused pinned-host
//! buffer and return a NON-OWNING Tensor view into it. Backing pool of pinned buffers instead of a
//! fresh cudaMallocHost/cudaFreeHost pair per call (measured as a real cost: image loading happens
//! once per camera image, every request). Sized well above any realistic simultaneous-image count
//! per request; slots cycle round-robin, each reshaped (grown via move-assignment only if needed)
//! rather than reallocated. Safe because the pool itself has static storage duration (outlives any
//! ImageData returned from a single request-processing pass) and slots aren't revisited within the
//! small number of calls one request makes. Shared by loadImageFromMemory (post-decode) and
//! loadImageFromRaw (already-decoded caller pixels).
rt::Tensor copyIntoPooledImageSlot(unsigned char const* src, int64_t height, int64_t width, int64_t channels)
{
    constexpr size_t kPoolSize = 16;
    static std::vector<rt::Tensor> sPool(kPoolSize);
    static size_t sNextSlot = 0;
    rt::Tensor& slot = sPool[sNextSlot];
    sNextSlot = (sNextSlot + 1) % kPoolSize;

    Coords const shape({height, width, channels});
    if (!slot.reshape(shape))
    {
        slot = rt::Tensor(shape, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "imageUtils::pooledImageSlot");
    }
    memcpy(slot.dataPointer<unsigned char>(), src, width * height * channels);
    return rt::Tensor(
        slot.dataPointer<unsigned char>(), shape, rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8, "imageUtils::pooledImageSlot::view");
}
} // namespace

ImageData loadImageFromMemory(unsigned char const* data, size_t size)
{
    int width{0}, height{0}, channels{0};
    // Only support RGB images
    int desiredChannels = 3;
    unsigned char* imageData = stbi_load_from_memory(data, size, &width, &height, &channels, desiredChannels);
    ELLM_CHECK(imageData != nullptr, "Failed to load image from memory: " + std::string(stbi_failure_reason()));

    ImageData result(copyIntoPooledImageSlot(imageData, height, width, desiredChannels));
    stbi_image_free(imageData);
    return result;
}

ImageData loadImageFromRaw(unsigned char const* data, int64_t height, int64_t width, int64_t channels)
{
    // Already-decoded RGB pixels (e.g. from a serving path that has the frame in memory already):
    // skip stbi entirely and hand the bytes straight to the pooled-slot copy. Avoids the pure-waste
    // JPEG encode->decode round-trip a caller would otherwise pay just to reach loadImageFromMemory.
    ELLM_CHECK(channels == 3, "loadImageFromRaw only supports 3-channel (RGB) images");
    return ImageData(copyIntoPooledImageSlot(data, height, width, channels));
}

void resizeImage(
    ImageData const& image, ImageData& resizedImage, int64_t newWidth, int64_t newHeight, InterpolationMode mode)
{
    // Reshape pre-allocated buffer to target dimensions
    bool success = resizedImage.buffer->reshape({newHeight, newWidth, image.channels});
    ELLM_CHECK(success, "Failed to reshape resized image buffer");
    resizedImage.height = newHeight;
    resizedImage.width = newWidth;
    resizedImage.channels = image.channels;

    // Resize the image into the pre-allocated buffer
    constexpr int32_t kINPUT_STRIDE_BYTES{0};
    constexpr int32_t kOUTPUT_STRIDE_BYTES{0};
    if (mode == InterpolationMode::kBICUBIC)
    {
        stbir_resize(image.data(), image.width, image.height, kINPUT_STRIDE_BYTES, resizedImage.data(), newWidth,
            newHeight, kOUTPUT_STRIDE_BYTES, STBIR_RGB, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
    }
    else
    {
        stbir_resize_uint8_linear(image.data(), image.width, image.height, kINPUT_STRIDE_BYTES, resizedImage.data(),
            newWidth, newHeight, kOUTPUT_STRIDE_BYTES, STBIR_RGB);
    }
}

} // namespace imageUtils
} // namespace rt
} // namespace trt_edgellm

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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "builder/llmBuilder.h"
#include "builder/visualBuilder.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "common/trtUtils.h"
#include "profiling/metrics.h"
#include "kernels/typeCast/halfToFloatCast.h"
#include "runtime/audioLoader.h"
#include "runtime/audioUtils.h"
#include "runtime/imageUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/melSpectrogram.h"

#include <chrono>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

//! RAII wrapper for CUDA stream management.
class CudaStreamWrapper
{
public:
    CudaStreamWrapper()
    {
        CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    ~CudaStreamWrapper()
    {
        if (mStream != nullptr)
        {
            cudaStreamDestroy(mStream);
        }
    }

    CudaStreamWrapper(CudaStreamWrapper const&) = delete;
    CudaStreamWrapper& operator=(CudaStreamWrapper const&) = delete;

    CudaStreamWrapper(CudaStreamWrapper&& other) noexcept
        : mStream(other.mStream)
    {
        other.mStream = nullptr;
    }

    CudaStreamWrapper& operator=(CudaStreamWrapper&& other) noexcept
    {
        if (this != &other)
        {
            if (mStream != nullptr)
            {
                cudaStreamDestroy(mStream);
            }
            mStream = other.mStream;
            other.mStream = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const
    {
        return mStream;
    }

private:
    cudaStream_t mStream{nullptr};
};

//! Convert a device- or host-resident FP16/FP32 Tensor into a float32 numpy array, synchronizing
//! `stream` first when the source is device-resident. Used by the raw-forward/vision-only entry
//! points, which hand back whatever dtype the internal pipeline buffer happens to use (FP16 for
//! embeddings/hidden_states, FP32 for logits/mrope) as one consistent Python-facing dtype.
//! Converts `tensor` (device- or host-resident, FP16 or FP32) to a freshly allocated FP32 numpy
//! array. Two perf-critical design points, both confirmed via profiling (comparing this pipeline's
//! end-to-end latency against `llm_bench`'s direct engine-only benchmark on the identical engine
//! and input size, which showed ~3x less time -- ~23ms vs ~72ms -- pointing squarely at this
//! function as the source of the gap):
//!   1. The device-to-host copy stages through a persistent, reused, PINNED host buffer instead of
//!      a fresh `std::vector` -- unpinned host memory forces cudaMemcpyAsync to fall back to a
//!      synchronous staged copy internally (same issue as numpyFloat32ToHostHalfTensorInto()).
//!   2. For FP16 device tensors (e.g. a [1, 487, 2560] hidden_states -- ~1.25M elements), the
//!      half->float cast runs as a GPU kernel (castHalfToFloatDevice) BEFORE the device-to-host
//!      copy, so the transferred bytes are already FP32. The previous implementation copied the
//!      raw FP16 bytes to host first, then converted with a scalar, single-threaded
//!      `__half2float` loop over every element on the CPU -- measured as the dominant cost for
//!      this call.
py::array_t<float> tensorToNumpyFloat32(Tensor const& tensor, cudaStream_t stream)
{
    Coords const shape = tensor.getShape();
    std::vector<ssize_t> npShape;
    npShape.reserve(static_cast<size_t>(shape.getNumDims()));
    for (int32_t d = 0; d < shape.getNumDims(); ++d)
    {
        npShape.push_back(static_cast<ssize_t>(shape[d]));
    }
    int64_t const volume = shape.volume();
    py::array_t<float> out(npShape);

    // Persistent pinned-host staging buffer for the final device-to-host copy, reused/grown
    // across calls regardless of which caller (runVisionOnly / runBackboneRawForward /
    // runFullPreprocessOnly) or tensor (embeddings / deepstack / hidden_states / logits) is being
    // converted -- each call fully copies the staged data into its own freshly allocated `out`
    // before returning, so reuse across distinct tensors/callers is safe.
    static Tensor hostStagingBuffer;
    Coords const floatShape(std::vector<int64_t>{volume});
    if (!hostStagingBuffer.reshape(floatShape))
    {
        hostStagingBuffer
            = Tensor(floatShape, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "tensorToNumpyFloat32::hostStaging");
    }
    float* hostBuf = hostStagingBuffer.dataPointer<float>();

    if (tensor.getDataType() == nvinfer1::DataType::kFLOAT)
    {
        if (tensor.getDeviceType() == DeviceType::kGPU)
        {
            CUDA_CHECK(cudaMemcpyAsync(
                hostBuf, tensor.rawPointer(), static_cast<size_t>(volume) * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        else
        {
            std::memcpy(hostBuf, tensor.rawPointer(), static_cast<size_t>(volume) * sizeof(float));
        }
    }
    else if (tensor.getDataType() == nvinfer1::DataType::kHALF)
    {
        if (tensor.getDeviceType() == DeviceType::kGPU)
        {
            static Tensor deviceCastBuffer;
            if (!deviceCastBuffer.reshape(floatShape))
            {
                deviceCastBuffer = Tensor(
                    floatShape, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "tensorToNumpyFloat32::deviceCast");
            }
            kernel::castHalfToFloatDevice(deviceCastBuffer.dataPointer<float>(),
                reinterpret_cast<half const*>(tensor.rawPointer()), volume, stream);
            CUDA_CHECK(cudaMemcpyAsync(hostBuf, deviceCastBuffer.rawPointer(),
                static_cast<size_t>(volume) * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        else
        {
            // CPU-resident half tensor: no device to run the cast kernel on. Not currently
            // exercised by any runVisionOnly/runBackboneRawForward/runFullPreprocessOnly caller.
            half const* halfSrc = reinterpret_cast<half const*>(tensor.rawPointer());
            for (int64_t i = 0; i < volume; ++i)
            {
                hostBuf[i] = __half2float(halfSrc[i]);
            }
        }
    }
    else
    {
        throw std::runtime_error("tensorToNumpyFloat32: unsupported tensor dtype (only FP16/FP32 supported)");
    }
    std::memcpy(out.mutable_data(), hostBuf, static_cast<size_t>(volume) * sizeof(float));
    return out;
}

//! Copy a caller-supplied FP32 numpy array (mrope_cos_sin) into `buffer` -- a persistent,
//! caller-owned, PINNED host FP32 Tensor reused across calls. Deliberately NOT zero-copy: the
//! subsequent cudaMemcpyAsync(..., HostToDevice, ...) in runBackboneRawForward() needs pinned
//! source memory to actually run asynchronously -- copying from ordinary (non-pinned) numpy-owned
//! host memory forces the CUDA driver to fall back to a synchronous staged copy, which blocks the
//! calling CPU thread for the full transfer duration (confirmed via nsys: cudaMemcpyAsync total
//! time dropped from ~200ms to low milliseconds across 8 forward passes after switching this call
//! site to a pinned buffer).
void numpyFloat32ToHostFloatTensorInto(
    Tensor& buffer, py::array_t<float, py::array::c_style | py::array::forcecast> const& arr)
{
    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(arr.ndim()));
    for (ssize_t d = 0; d < arr.ndim(); ++d)
    {
        dims.push_back(arr.shape(d));
    }
    Coords const shape(dims);
    if (!buffer.reshape(shape))
    {
        buffer = Tensor(shape, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "numpyFloat32ToHostFloatTensor");
    }
    std::memcpy(buffer.dataPointer<float>(), arr.data(), static_cast<size_t>(shape.volume()) * sizeof(float));
}

//! Python-facing result of PyLLMRuntime::runVisionOnly().
struct PyVisionOnlyResult
{
    bool success{false};
    py::array_t<float> outputEmbedding;
    std::vector<py::array_t<float>> deepstackFeatures;
};

//! Python-facing result of PyLLMRuntime::runBackboneRawForward().
struct PyRawForwardResult
{
    bool success{false};
    py::array_t<float> logits;
    py::array_t<float> hiddenStates;
};

//! Python-facing result of PyLLMRuntime::runFullPreprocessOnly().
struct PyFullPreprocessResult
{
    bool success{false};
    py::array_t<float> inputsEmbeds;
    std::vector<py::array_t<float>> deepstackEmbeds;
    py::array_t<float> mropeCosSin;
    std::vector<int32_t> idsInput;
};

//! Unified Python wrapper for LLMInferenceRuntime.
//! Supports both vanilla decoding (no draft model) and Eagle speculative decoding
//! through constructor overloading — mirrors the C++ unified runtime.
class PyLLMRuntime
{
public:
    //! Vanilla constructor (no speculative decoding).
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap)
    {
        mPluginHandle = loadEdgellmPluginLib();
        mRuntime = std::make_unique<LLMInferenceRuntime>(engineDir, multimodalEngineDir, loraWeightsMap, mStream.get());
    }

    //! Eagle speculative decoding constructor.
    PyLLMRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, int32_t draftTopK, int32_t draftStep,
        int32_t verifyTreeSize)
    {
        mPluginHandle = loadEdgellmPluginLib();
        SpecDecodeDraftingConfig draftingConfig{draftTopK, draftStep, verifyTreeSize};
        mRuntime = std::make_unique<LLMInferenceRuntime>(
            engineDir, multimodalEngineDir, loraWeightsMap, draftingConfig, mStream.get());
    }

    LLMGenerationResponse handleRequest(LLMGenerationRequest const& request)
    {
        LLMGenerationResponse response;
        bool const success = mRuntime->handleRequest(request, response, mStream.get());
        ELLM_CHECK(success, "Failed to handle generation request");
        return response;
    }

    //! Run only the vision encoder for `request`'s image(s), skipping text tokenization/M-RoPE.
    //! Returns the output embedding and per-layer deepstack features as float32 numpy arrays, for
    //! host-side embedding assembly ahead of runBackboneRawForward().
    PyVisionOnlyResult runVisionOnly(LLMGenerationRequest const& request)
    {
        auto const result = mRuntime->runVisionOnly(request, mStream.get());
        PyVisionOnlyResult pyResult;
        pyResult.success = result.success;
        if (!result.success)
        {
            return pyResult;
        }
        pyResult.outputEmbedding = tensorToNumpyFloat32(*result.outputEmbedding, mStream.get());
        pyResult.deepstackFeatures.reserve(result.deepstackFeatures.size());
        for (auto const& feature : result.deepstackFeatures)
        {
            pyResult.deepstackFeatures.push_back(tensorToNumpyFloat32(feature.get(), mStream.get()));
        }
        return pyResult;
    }

    //! Run one independent prefill forward pass of the base decoder engine over a caller-assembled
    //! `inputs_embeds` [1, seq, hidden] (+ optional per-layer `deepstack_embeds`, `mrope_cos_sin]),
    //! bypassing embedding lookup/sampling entirely. Returns raw logits/hidden_states as float32
    //! numpy arrays. All three inputs are plain float32 numpy; fp32->fp16 casting for the engine's
    //! actual fp16 embedding buffers happens inside this wrapper, not in the caller.
    PyRawForwardResult runBackboneRawForward(
        py::array_t<float, py::array::c_style | py::array::forcecast> const& inputsEmbeds,
        std::vector<py::array_t<float, py::array::c_style | py::array::forcecast>> const& deepstackEmbeds,
        py::array_t<float, py::array::c_style | py::array::forcecast> const& mropeCosSin)
    {
        // Reused, persistent pinned-host FP32 staging buffers (grow-if-needed) instead of
        // allocating a fresh Tensor per call. Deliberately NOT cast to FP16 here: passed through
        // as FP32, runBackboneRawForward() does the H2D copy + FP32->FP16 cast itself on the GPU
        // (see LLMInferenceRuntime::runBackboneRawForward) -- avoids a scalar host-side
        // __float2half loop that was measured as a multi-ms bottleneck for a ~1.25M-element
        // [1, ~500, 2560] embedding tensor.
        numpyFloat32ToHostFloatTensorInto(mHostInputsEmbedsBuffer, inputsEmbeds);

        if (mHostDeepstackBuffers.size() != deepstackEmbeds.size())
        {
            mHostDeepstackBuffers.resize(deepstackEmbeds.size());
        }
        OptionalInputTensors deepstackRefs;
        deepstackRefs.reserve(deepstackEmbeds.size());
        for (size_t i = 0; i < deepstackEmbeds.size(); ++i)
        {
            numpyFloat32ToHostFloatTensorInto(mHostDeepstackBuffers[i], deepstackEmbeds[i]);
            deepstackRefs.emplace_back(std::cref(mHostDeepstackBuffers[i]));
        }

        numpyFloat32ToHostFloatTensorInto(mHostMropeCosSinBuffer, mropeCosSin);

        // Release the GIL for the engine forward (pure C++/CUDA, touches no Python objects) so the
        // server's control-loop / asyncio threads keep running during this ~30-80ms call. Reacquire
        // before tensorToNumpyFloat32(), which allocates numpy arrays and needs the GIL.
        auto const result = [&]
        {
            py::gil_scoped_release release;
            return mRuntime->runBackboneRawForward(
                mHostInputsEmbedsBuffer, deepstackRefs, mHostMropeCosSinBuffer, mStream.get());
        }();

        PyRawForwardResult pyResult;
        pyResult.success = result.success;
        if (!result.success)
        {
            return pyResult;
        }
        pyResult.logits = tensorToNumpyFloat32(*result.outputLogits, mStream.get());
        pyResult.hiddenStates = tensorToNumpyFloat32(*result.outputHiddenStates, mStream.get());
        return pyResult;
    }

    //! Run full request preprocessing (chat template, tokenize, vision/audio encode, M-RoPE,
    //! embedding + deepstack assembly) exactly as handle_request() would, then return before the
    //! base decoder engine actually executes. For diffing the engine's own preprocessing against
    //! an eager reimplementation: feed the returned inputs_embeds/deepstack_embeds/mrope_cos_sin
    //! straight into run_backbone_raw_forward() to isolate the backbone decoder from
    //! embedding-assembly differences.
    PyFullPreprocessResult runFullPreprocessOnly(LLMGenerationRequest const& request)
    {
        auto const result = mRuntime->runFullPreprocessOnly(request, mStream.get());
        PyFullPreprocessResult pyResult;
        pyResult.success = result.success;
        if (!result.success)
        {
            return pyResult;
        }
        pyResult.inputsEmbeds = tensorToNumpyFloat32(*result.inputsEmbeds, mStream.get());
        pyResult.deepstackEmbeds.reserve(result.deepstackEmbeds.size());
        for (auto const& feature : result.deepstackEmbeds)
        {
            pyResult.deepstackEmbeds.push_back(tensorToNumpyFloat32(feature.get(), mStream.get()));
        }
        pyResult.mropeCosSin = tensorToNumpyFloat32(*result.mropeCosSin, mStream.get());
        pyResult.idsInput = result.idsInput;
        return pyResult;
    }

    bool captureDecodingCudaGraph()
    {
        return mRuntime->captureDecodingCUDAGraph(mStream.get());
    }

    //! Like runFullPreprocessOnly(), but for the device-resident fast path: returns only
    //! `ids_input` (tiny, needed on host to locate a special token's position) and leaves
    //! inputs_embeds/deepstack_embeds/mrope_cos_sin on the device, un-converted -- pair with
    //! inject_state_embedding() + run_backbone_raw_forward_from_preprocessed() to avoid the
    //! device->host->device round-trip run_full_preprocess_only() +
    //! run_backbone_raw_forward() would otherwise require for a caller that only needs to patch
    //! one token's embedding (e.g. injecting a third-modality/robot-state embedding that has no
    //! slot in the engine's own preprocessing).
    std::pair<bool, std::vector<int32_t>> runFullPreprocessOnlyDeviceResident(LLMGenerationRequest const& request)
    {
        // Chat-template/tokenize/vision-encode/M-RoPE/embedding-assembly (~13-20ms) is pure
        // C++/CUDA; release the GIL so the server's control/asyncio threads keep running. The
        // returned idsInput is a plain std::vector (marshalled to Python after reacquire).
        auto const result = [&]
        {
            py::gil_scoped_release release;
            return mRuntime->runFullPreprocessOnly(request, mStream.get());
        }();
        return {result.success, result.idsInput};
    }

    //! Overwrite one token position of the just-preprocessed inputs_embeds in place, on the
    //! device. Must follow run_full_preprocess_only_device_resident() and precede
    //! run_backbone_raw_forward_from_preprocessed(). `state_embedding` is a small
    //! `[hidden_size]` FP32 host array (e.g. the robot-state projector's output) -- NOT the full
    //! `[1, seq, hidden]` embedding tensor.
    bool injectStateEmbedding(int32_t position, py::array_t<float, py::array::c_style | py::array::forcecast> const& stateEmbedding)
    {
        numpyFloat32ToHostFloatTensorInto(mHostStateEmbeddingBuffer, stateEmbedding);
        return mRuntime->injectStateEmbedding(position, mHostStateEmbeddingBuffer, mStream.get());
    }

    //! Like run_backbone_raw_forward(), but consumes inputs_embeds/deepstack_embeds/mrope_cos_sin
    //! as already populated by a preceding run_full_preprocess_only_device_resident() (+ optional
    //! inject_state_embedding()) -- no host round-trip for these tensors. See
    //! LLMInferenceRuntime::runBackboneRawForwardFromPreprocessed() for the vision-always
    //! restriction this relies on.
    PyRawForwardResult runBackboneRawForwardFromPreprocessed()
    {
        // GIL released for the engine forward (pure C++/CUDA); reacquired for the numpy marshalling
        // below (tensorToNumpyFloat32 allocates Python arrays). Lets the server's control/asyncio
        // threads run during this ~30ms call instead of stalling on the held GIL.
        auto const result = [&]
        {
            py::gil_scoped_release release;
            return mRuntime->runBackboneRawForwardFromPreprocessed(mStream.get());
        }();
        PyRawForwardResult pyResult;
        pyResult.success = result.success;
        if (!result.success)
        {
            return pyResult;
        }
        pyResult.logits = tensorToNumpyFloat32(*result.outputLogits, mStream.get());
        pyResult.hiddenStates = tensorToNumpyFloat32(*result.outputHiddenStates, mStream.get());
        return pyResult;
    }

    bool saveSystemPromptKVCache(std::string const& prompt, std::string const& loraWeightsName)
    {
        return mRuntime->genAndSaveSystemPromptKVCache(prompt, loraWeightsName, mStream.get());
    }

    bool hasDraftModel() const
    {
        return mRuntime->hasDraftModel();
    }

    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mRuntime->getPrefillMetrics();
    }

    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mRuntime->getGenerationMetrics();
    }

    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const
    {
        return mRuntime->getSpecDecodeGenerationMetrics();
    }

    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mRuntime->getMultimodalMetrics();
    }

private:
    CudaStreamWrapper mStream;
    std::unique_ptr<LLMInferenceRuntime> mRuntime;
    std::unique_ptr<void, DlDeleter> mPluginHandle;

    // Persistent pinned-host staging buffers for runBackboneRawForward(), reused across calls
    // (grown via move-assignment only when a larger shape is requested) -- see
    // numpyFloat32ToHostFloatTensorInto().
    Tensor mHostInputsEmbedsBuffer;
    std::vector<Tensor> mHostDeepstackBuffers;
    Tensor mHostMropeCosSinBuffer;
    Tensor mHostStateEmbeddingBuffer; //!< For injectStateEmbedding() -- a single [hiddenSize] row.
};

imageUtils::ImageData loadImageFromPath(std::string const& path)
{
    return imageUtils::loadImageFromFile(path);
}

imageUtils::ImageData loadImageFromBytes(py::bytes const& data)
{
    std::string dataStr = data;
    return imageUtils::loadImageFromMemory(reinterpret_cast<unsigned char const*>(dataStr.data()), dataStr.size());
}

//! \brief Build an ImageData from already-decoded raw pixels: an (H, W, 3) uint8 RGB numpy array.
//! Skips the image codec entirely, for callers (e.g. the serving path) that already hold the frame
//! decoded in memory and would otherwise pay a pure-waste encode->decode round-trip.
imageUtils::ImageData loadImageFromArray(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> const& arr)
{
    auto buf = arr.request();
    ELLM_CHECK(buf.ndim == 3 && buf.shape[2] == 3, "load_image_from_array expects an (H, W, 3) uint8 RGB array");
    return imageUtils::loadImageFromRaw(
        static_cast<unsigned char const*>(buf.ptr), buf.shape[0], buf.shape[1], buf.shape[2]);
}

//! \brief Build an AudioData from raw encoded audio bytes (wav / mp3 / flac).
//! Decodes via vendored miniaudio (16 kHz mono FP32) and hands raw PCM off to
//! the runner; mel extraction happens inside the audio runner per its
//! ``audio/config.json``.
audioUtils::AudioData loadAudioBufferFromBytes(py::bytes data)
{
    constexpr int32_t kTargetSampleRate = 16000;
    std::string dataStr = data;
    audioUtils::AudioData audio;
    if (!audioUtils::loadAudioDataFromBytes(
            reinterpret_cast<uint8_t const*>(dataStr.data()), dataStr.size(), kTargetSampleRate, audio))
    {
        throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
    }
    return audio;
}

//! \brief Decode audio bytes + extract mel to numpy.float32 (host-resident).
//! Diagnostic helper exposing the C++ ``MelExtractor`` output directly — the
//! production path (``loadAudioBufferFromBytes`` -> runner) only returns PCM
//! at the Python boundary and runs mel extraction inside the audio runner.
//! Returns the FE's natural 2-D layout.
py::array_t<float> extractMelToNumpy(py::bytes data, std::string const& feType)
{
    audio::MelExtractor extractor = audio::makeExtractorByName(feType);
    int32_t const targetSampleRate = extractor.config().sampleRate;

    char const* rawPtr = nullptr;
    Py_ssize_t rawSize = 0;
    if (PyBytes_AsStringAndSize(data.ptr(), const_cast<char**>(&rawPtr), &rawSize) != 0)
    {
        throw py::error_already_set();
    }

    audio::AudioPCM pcm;
    if (!audio::loadAudioBytes(
            reinterpret_cast<uint8_t const*>(rawPtr), static_cast<size_t>(rawSize), targetSampleRate, pcm))
    {
        throw std::runtime_error("Audio decode failed (unsupported container or corrupt bytes)");
    }

    Tensor hostMel;
    if (!extractor.extract(pcm, hostMel))
    {
        throw std::runtime_error("Mel extraction failed");
    }

    Coords const& shape = hostMel.getShape();
    std::vector<ssize_t> npShape;
    npShape.reserve(static_cast<size_t>(shape.getNumDims()));
    for (int32_t d = 0; d < shape.getNumDims(); ++d)
    {
        npShape.push_back(static_cast<ssize_t>(shape[d]));
    }
    py::array_t<float> out(npShape);
    std::memcpy(out.mutable_data(), hostMel.dataPointer<float>(), static_cast<size_t>(shape.volume()) * sizeof(float));
    return out;
}

} // anonymous namespace

PYBIND11_MODULE(_edgellm_runtime, m)
{
    m.doc() = "TensorRT Edge LLM Python Bindings";

    // ========================================================================
    // Profiling
    // ========================================================================
    m.def("set_profiling_enabled", &setProfilingEnabled, py::arg("enabled"),
        "Enable or disable profiling data collection");
    m.def("get_profiling_enabled", &getProfilingEnabled, "Check if profiling is currently enabled");

    // ========================================================================
    // Metrics
    // ========================================================================
    py::class_<metrics::LLMPrefillMetrics>(m, "LLMPrefillMetrics")
        .def_readonly("reused_tokens", &metrics::LLMPrefillMetrics::reusedTokens)
        .def_readonly("computed_tokens", &metrics::LLMPrefillMetrics::computedTokens)
        .def("get_total_runs", &metrics::LLMPrefillMetrics::getTotalRuns);

    py::class_<metrics::LLMGenerationMetrics>(m, "LLMGenerationMetrics")
        .def_readonly("generated_tokens", &metrics::LLMGenerationMetrics::generatedTokens)
        .def("get_total_runs", &metrics::LLMGenerationMetrics::getTotalRuns);

    py::class_<metrics::SpecDecodeGenerationMetrics>(m, "SpecDecodeGenerationMetrics")
        .def_readonly("total_iterations", &metrics::SpecDecodeGenerationMetrics::totalIterations)
        .def_readonly("total_generated_tokens", &metrics::SpecDecodeGenerationMetrics::totalGeneratedTokens)
        .def("get_total_runs", &metrics::SpecDecodeGenerationMetrics::getTotalRuns);

    py::class_<metrics::MultimodalMetrics>(m, "MultimodalMetrics")
        .def_readonly("total_images", &metrics::MultimodalMetrics::totalImages)
        .def_readonly("total_image_tokens", &metrics::MultimodalMetrics::totalImageTokens)
        .def("get_total_runs", &metrics::MultimodalMetrics::getTotalRuns);

    // ========================================================================
    // Image utilities
    // ========================================================================
    py::class_<imageUtils::ImageData>(m, "ImageData")
        .def(py::init<>())
        .def_readonly("width", &imageUtils::ImageData::width)
        .def_readonly("height", &imageUtils::ImageData::height)
        .def_readonly("channels", &imageUtils::ImageData::channels);

    m.def("load_image_from_path", &loadImageFromPath, py::arg("path"), "Load image from file path");
    m.def("load_image_from_bytes", &loadImageFromBytes, py::arg("data"), "Load image from bytes");
    m.def("load_image_from_array", &loadImageFromArray, py::arg("array"),
        "Build ImageData from a raw (H,W,3) uint8 RGB numpy array (no decode)");

    // ========================================================================
    // Audio utilities
    // ========================================================================
    py::class_<audioUtils::AudioData>(m, "AudioData")
        .def(py::init<>())
        .def_readwrite("sample_rate", &audioUtils::AudioData::sampleRate);

    m.def("load_audio_buffer_from_bytes", &loadAudioBufferFromBytes, py::arg("data"),
        "Build an AudioData from raw encoded audio bytes (wav/mp3/flac). "
        "Decodes to mono FP32 PCM @ 16 kHz via miniaudio; the audio runner "
        "extracts mel internally per its audio/config.json.");

    m.def("extract_mel_to_numpy", &extractMelToNumpy, py::arg("data"), py::arg("fe_type"),
        "Decode audio bytes (wav/mp3/flac) and return the host float32 "
        "mel-spectrogram directly to numpy (no f16 cast, no GPU upload). "
        "Test / diagnostic helper for comparing the C++ pipeline against HF "
        "feature extractors at full float32 precision.");

    // ========================================================================
    // Message structures
    // ========================================================================
    py::class_<Message::MessageContent>(m, "MessageContent")
        .def(py::init<>())
        .def(py::init([](std::string const& type, std::string const& content) {
            Message::MessageContent mc;
            mc.type = type;
            mc.content = content;
            return mc;
        }),
            py::arg("type"), py::arg("content"))
        .def_readwrite("type", &Message::MessageContent::type)
        .def_readwrite("content", &Message::MessageContent::content);

    py::class_<Message>(m, "Message")
        .def(py::init<>())
        .def(py::init([](std::string const& role, std::vector<Message::MessageContent> const& contents) {
            Message msg;
            msg.role = role;
            msg.contents = contents;
            return msg;
        }),
            py::arg("role"), py::arg("contents"))
        .def_readwrite("role", &Message::role)
        .def_readwrite("contents", &Message::contents);

    m.def(
        "create_text_message",
        [](std::string const& role, std::string const& text) {
            Message msg;
            msg.role = role;
            Message::MessageContent content;
            content.type = "text";
            content.content = text;
            msg.contents.push_back(content);
            return msg;
        },
        py::arg("role"), py::arg("text"), "Create a simple text message");

    // ========================================================================
    // Request / Response
    // ========================================================================
    py::class_<LLMGenerationRequest::FormattedRequest>(m, "FormattedRequest")
        .def(py::init<>())
        .def_readwrite("formatted_system_prompt", &LLMGenerationRequest::FormattedRequest::formattedSystemPrompt)
        .def_readwrite("formatted_complete_request", &LLMGenerationRequest::FormattedRequest::formattedCompleteRequest);

    py::class_<LLMGenerationRequest::Request>(m, "Request")
        .def(py::init<>())
        .def(py::init([](std::vector<Message> const& messages) {
            LLMGenerationRequest::Request req;
            req.messages = messages;
            return req;
        }),
            py::arg("messages"))
        .def_readwrite("messages", &LLMGenerationRequest::Request::messages)
        .def_readwrite("image_buffers", &LLMGenerationRequest::Request::imageBuffers)
        .def_readwrite("audio_buffers", &LLMGenerationRequest::Request::audioBuffers)
        .def_readwrite("stop_strings", &LLMGenerationRequest::Request::stopStrings);

    // ========================================================================
    // Streaming
    // ========================================================================
    py::enum_<FinishReason>(m, "FinishReason")
        .value("NOT_FINISHED", FinishReason::kNotFinished)
        .value("END_ID", FinishReason::kEndId)
        .value("LENGTH", FinishReason::kLength)
        .value("CANCELLED", FinishReason::kCancelled)
        .value("ERROR", FinishReason::kError)
        .value("STOP_WORDS", FinishReason::kStopWords);

    py::class_<StreamChunk>(m, "StreamChunk")
        .def(py::init<>())
        .def_readonly("token_ids", &StreamChunk::tokenIds)
        .def_readonly("text", &StreamChunk::text)
        .def_readonly("finished", &StreamChunk::finished)
        .def_readonly("reason", &StreamChunk::reason);

    py::class_<StreamChannel, std::shared_ptr<StreamChannel>>(m, "StreamChannel")
        .def_static("create", &StreamChannel::create)
        .def("try_pop", &StreamChannel::tryPop)
        .def(
            "wait_pop",
            [](StreamChannel& self, int64_t timeoutMs) { return self.waitPop(std::chrono::milliseconds{timeoutMs}); },
            py::arg("timeout_ms"), py::call_guard<py::gil_scoped_release>())
        .def("is_finished", &StreamChannel::isFinished)
        .def("get_reason", &StreamChannel::getReason)
        .def("is_cancelled", &StreamChannel::isCancelled)
        .def("cancel", &StreamChannel::cancel)
        .def("set_stream_interval", &StreamChannel::setStreamInterval, py::arg("n"))
        .def("get_stream_interval", &StreamChannel::getStreamInterval)
        .def("set_skip_special_tokens", &StreamChannel::setSkipSpecialTokens, py::arg("skip"))
        .def("get_skip_special_tokens", &StreamChannel::getSkipSpecialTokens);

    // ========================================================================
    // Request / Response (continued)
    // ========================================================================
    py::class_<LLMGenerationRequest>(m, "LLMGenerationRequest")
        .def(py::init<>())
        .def_readwrite("requests", &LLMGenerationRequest::requests)
        .def_readwrite("formatted_requests", &LLMGenerationRequest::formattedRequests)
        .def_readwrite("temperature", &LLMGenerationRequest::temperature)
        .def_readwrite("top_p", &LLMGenerationRequest::topP)
        .def_readwrite("top_k", &LLMGenerationRequest::topK)
        .def_readwrite("max_generate_length", &LLMGenerationRequest::maxGenerateLength)
        .def_readwrite("lora_weights_name", &LLMGenerationRequest::loraWeightsName)
        .def_readwrite("save_system_prompt_kv_cache", &LLMGenerationRequest::saveSystemPromptKVCache)
        .def_readwrite("apply_chat_template", &LLMGenerationRequest::applyChatTemplate)
        .def_readwrite("add_generation_prompt", &LLMGenerationRequest::addGenerationPrompt)
        .def_readwrite("enable_thinking", &LLMGenerationRequest::enableThinking)
        .def_readwrite("disable_spec_decode", &LLMGenerationRequest::disableSpecDecode)
        .def_readwrite("stream_channels", &LLMGenerationRequest::streamChannels);

    py::class_<LLMGenerationResponse>(m, "LLMGenerationResponse")
        .def(py::init<>())
        .def_readwrite("output_ids", &LLMGenerationResponse::outputIds)
        .def_readwrite("output_texts", &LLMGenerationResponse::outputTexts)
        .def_readonly("finish_reasons", &LLMGenerationResponse::finishReasons);

    py::class_<PyVisionOnlyResult>(m, "VisionOnlyResult")
        .def(py::init<>())
        .def_readonly("success", &PyVisionOnlyResult::success)
        .def_readonly("output_embedding", &PyVisionOnlyResult::outputEmbedding)
        .def_readonly("deepstack_features", &PyVisionOnlyResult::deepstackFeatures);

    py::class_<PyRawForwardResult>(m, "RawForwardResult")
        .def(py::init<>())
        .def_readonly("success", &PyRawForwardResult::success)
        .def_readonly("logits", &PyRawForwardResult::logits)
        .def_readonly("hidden_states", &PyRawForwardResult::hiddenStates);

    py::class_<PyFullPreprocessResult>(m, "FullPreprocessResult")
        .def(py::init<>())
        .def_readonly("success", &PyFullPreprocessResult::success)
        .def_readonly("inputs_embeds", &PyFullPreprocessResult::inputsEmbeds)
        .def_readonly("deepstack_embeds", &PyFullPreprocessResult::deepstackEmbeds)
        .def_readonly("mrope_cos_sin", &PyFullPreprocessResult::mropeCosSin)
        .def_readonly("ids_input", &PyFullPreprocessResult::idsInput);

    // ========================================================================
    // Runtime: unified (vanilla + Eagle speculative decoding)
    // ========================================================================
    py::class_<PyLLMRuntime>(m, "LLMRuntime",
        "Unified LLM inference runtime. Supports both vanilla decoding and Eagle speculative decoding.")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir") = "",
            py::arg("lora_weights_map") = std::unordered_map<std::string, std::string>{},
            "Construct for vanilla (non-speculative) decoding")
        .def(py::init<std::string const&, std::string const&, std::unordered_map<std::string, std::string> const&,
                 int32_t, int32_t, int32_t>(),
            py::arg("engine_dir"), py::arg("multimodal_engine_dir"), py::arg("lora_weights_map"),
            py::arg("draft_top_k"), py::arg("draft_step"), py::arg("verify_tree_size"),
            "Construct for Eagle speculative decoding")
        .def("handle_request", &PyLLMRuntime::handleRequest, py::arg("request"),
            py::call_guard<py::gil_scoped_release>(), "Process a generation request and return the response")
        .def("run_vision_only", &PyLLMRuntime::runVisionOnly, py::arg("request"),
            "Run only the vision encoder for a request's image(s) (skips text tokenization/M-RoPE); "
            "returns output_embedding + per-layer deepstack_features as float32 numpy arrays")
        .def("run_backbone_raw_forward", &PyLLMRuntime::runBackboneRawForward, py::arg("inputs_embeds"),
            py::arg("deepstack_embeds"), py::arg("mrope_cos_sin"),
            "Run one independent prefill forward pass of the base decoder engine given a caller-assembled "
            "inputs_embeds [1, seq, hidden] (+ optional deepstack_embeds, mrope_cos_sin), bypassing embedding "
            "lookup/sampling. Returns raw logits/hidden_states as float32 numpy arrays")
        .def("run_full_preprocess_only", &PyLLMRuntime::runFullPreprocessOnly, py::arg("request"),
            "Run full request preprocessing (chat template, tokenize, vision/audio encode, M-RoPE, embedding + "
            "deepstack assembly) exactly as handle_request() would, returning before the base decoder engine "
            "executes -- for diffing against an eager reimplementation")
        .def("run_full_preprocess_only_device_resident", &PyLLMRuntime::runFullPreprocessOnlyDeviceResident,
            py::arg("request"),
            "Like run_full_preprocess_only(), but leaves inputs_embeds/deepstack_embeds/mrope_cos_sin "
            "device-resident (returns only ids_input) -- pair with inject_state_embedding() + "
            "run_backbone_raw_forward_from_preprocessed() to avoid a host round-trip when only one token's "
            "embedding needs patching")
        .def("inject_state_embedding", &PyLLMRuntime::injectStateEmbedding, py::arg("position"),
            py::arg("state_embedding"),
            "Overwrite one token position of the just-preprocessed inputs_embeds in place, on the device. "
            "state_embedding is [hidden_size] FP32, not the full [1, seq, hidden] tensor")
        .def("run_backbone_raw_forward_from_preprocessed", &PyLLMRuntime::runBackboneRawForwardFromPreprocessed,
            "Like run_backbone_raw_forward(), but consumes inputs_embeds/deepstack_embeds/mrope_cos_sin as "
            "already populated by run_full_preprocess_only_device_resident() (+ optional "
            "inject_state_embedding()) -- no host round-trip for these tensors")
        .def("capture_decoding_cuda_graph", &PyLLMRuntime::captureDecodingCudaGraph,
            "Capture CUDA graphs for optimized decoding")
        .def("save_system_prompt_kv_cache", &PyLLMRuntime::saveSystemPromptKVCache, py::arg("prompt"),
            py::arg("lora_weights_name") = "", "Pre-generate and cache system prompt KV cache")
        .def("has_draft_model", &PyLLMRuntime::hasDraftModel, "Check if speculative decoding draft model is loaded")
        .def("get_prefill_metrics", &PyLLMRuntime::getPrefillMetrics, py::return_value_policy::reference_internal)
        .def("get_generation_metrics", &PyLLMRuntime::getGenerationMetrics, py::return_value_policy::reference_internal)
        .def("get_spec_decode_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::return_value_policy::reference_internal)
        .def("get_eagle_generation_metrics", &PyLLMRuntime::getSpecDecodeGenerationMetrics,
            py::return_value_policy::reference_internal) // deprecated alias
        .def("get_multimodal_metrics", &PyLLMRuntime::getMultimodalMetrics);

    // ========================================================================
    // Builder: LLM
    // ========================================================================
    py::class_<builder::LLMBuilderConfig>(
        m, "LLMBuilderConfig", "Configuration for building TensorRT LLM engines from ONNX.")
        .def(py::init<>())
        .def_readwrite("max_input_len", &builder::LLMBuilderConfig::maxInputLen)
        .def_readwrite("spec_draft", &builder::LLMBuilderConfig::specDraft)
        .def_readwrite("spec_base", &builder::LLMBuilderConfig::specBase)
        .def_readwrite("eagle_draft", &builder::LLMBuilderConfig::specDraft) // deprecated alias
        .def_readwrite("eagle_base", &builder::LLMBuilderConfig::specBase)   // deprecated alias
        .def_readwrite("max_batch_size", &builder::LLMBuilderConfig::maxBatchSize)
        .def_readwrite("max_lora_rank", &builder::LLMBuilderConfig::maxLoraRank)
        .def_readwrite("max_kv_cache_capacity", &builder::LLMBuilderConfig::maxKVCacheCapacity)
        .def_readwrite("max_verify_tree_size", &builder::LLMBuilderConfig::maxVerifyTreeSize)
        .def_readwrite("max_draft_tree_size", &builder::LLMBuilderConfig::maxDraftTreeSize)
        .def_readwrite("use_trt_native_ops", &builder::LLMBuilderConfig::useTrtNativeOps)
        .def("__repr__", &builder::LLMBuilderConfig::toString);

    py::class_<builder::LLMBuilder>(m, "LLMBuilder", "Build a TensorRT engine from an ONNX directory.")
        .def(py::init<std::filesystem::path const&, std::filesystem::path const&, builder::LLMBuilderConfig const&>(),
            py::arg("onnx_dir"), py::arg("engine_dir"), py::arg("config"))
        .def("build", &builder::LLMBuilder::build, "Build the TensorRT engine. Returns True on success.");

    // ========================================================================
    // Builder: Visual
    // ========================================================================
    py::class_<builder::VisualBuilderConfig>(
        m, "VisualBuilderConfig", "Configuration for building TensorRT visual encoder engines from ONNX.")
        .def(py::init<>())
        .def_readwrite("min_image_tokens", &builder::VisualBuilderConfig::minImageTokens)
        .def_readwrite("max_image_tokens", &builder::VisualBuilderConfig::maxImageTokens)
        .def_readwrite("max_image_tokens_per_image", &builder::VisualBuilderConfig::maxImageTokensPerImage)
        .def("__repr__", &builder::VisualBuilderConfig::toString);

    py::class_<builder::VisualBuilder>(
        m, "VisualBuilder", "Build a TensorRT engine for a visual encoder from an ONNX directory.")
        .def(
            py::init<std::filesystem::path const&, std::filesystem::path const&, builder::VisualBuilderConfig const&>(),
            py::arg("onnx_dir"), py::arg("engine_dir"), py::arg("config"))
        .def("build", &builder::VisualBuilder::build, "Build the TensorRT visual engine. Returns True on success.");

    // ========================================================================
    // Convenience: create_generation_request
    // ========================================================================
    m.def(
        "create_generation_request",
        [](std::vector<std::vector<Message>> const& batchMessages, float temperature, float topP, int64_t topK,
            int64_t maxGenerateLength, bool applyChatTemplate, bool addGenerationPrompt, bool enableThinking,
            std::string const& loraWeightsName, bool saveSystemPromptKvCache, bool disableSpecDecode) {
            LLMGenerationRequest request;
            request.temperature = temperature;
            request.topP = topP;
            request.topK = topK;
            request.maxGenerateLength = maxGenerateLength;
            request.applyChatTemplate = applyChatTemplate;
            request.addGenerationPrompt = addGenerationPrompt;
            request.enableThinking = enableThinking;
            request.loraWeightsName = loraWeightsName;
            request.saveSystemPromptKVCache = saveSystemPromptKvCache;
            request.disableSpecDecode = disableSpecDecode;

            for (auto const& messages : batchMessages)
            {
                LLMGenerationRequest::Request req;
                req.messages = messages;
                request.requests.push_back(std::move(req));
            }
            return request;
        },
        py::arg("batch_messages"), py::arg("temperature") = 1.0f, py::arg("top_p") = 0.8f, py::arg("top_k") = 50,
        py::arg("max_generate_length") = 256, py::arg("apply_chat_template") = true,
        py::arg("add_generation_prompt") = true, py::arg("enable_thinking") = false, py::arg("lora_weights_name") = "",
        py::arg("save_system_prompt_kv_cache") = false, py::arg("disable_spec_decode") = false,
        "Create a generation request from a batch of message lists.");
}

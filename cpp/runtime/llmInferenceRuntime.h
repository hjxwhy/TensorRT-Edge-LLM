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

#include "action/alpamayo1ActionRunner.h"
#include "common/hashUtils.h"
#include "common/tensor.h"
#include "multimodal/multimodalRunner.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/decoding/decoderRegistry.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/systemPromptKVCache.h"
#include "runtime/streaming.h"
#include "tokenizer/tokenizer.h"
#include <memory>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{
/*!
 * @brief Unified LLM inference runtime with optional speculative decoding
 *
 * Manages inference pipeline for vanilla and speculative decoding modes (EAGLE, MTP, etc.).
 * When constructed without a drafting config, operates as a pure vanilla decoding runtime
 * with zero draft-model memory overhead.
 * Coordinates base model, optional draft model, and multimodal processing (vision + audio).
 */
class LLMInferenceRuntime
{
public:
    /*!
     * @brief Construct runtime with speculative decoding
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param draftingConfig Speculative decoding drafting configuration
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        SpecDecodeDraftingConfig const& draftingConfig, cudaStream_t stream);

    /*!
     * @brief Construct runtime for vanilla-only decoding (no draft model)
     * @param engineDir Directory containing engine files
     * @param multimodalEngineDir Directory containing multimodal engine files
     * @param loraWeightsMap Map of LoRA weight names to file paths
     * @param stream CUDA stream for operations
     * @throws std::runtime_error if directories do not contain expected data, or runner initialization fails
     */
    LLMInferenceRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream);

    //! @brief Destructor
    ~LLMInferenceRuntime() noexcept = default;

    //! @brief Capture CUDA graphs for decoding stages to optimize performance.
    //!
    //! When draft model is present, captures graphs for draft proposal, draft accept token,
    //! base verification, and base vanilla decoding. Without draft model, captures only
    //! vanilla decoding graphs.
    //!
    //! @param stream CUDA stream
    //! @return True if all stage captures succeed, false otherwise
    //! @throws std::runtime_error if a tensor reshape operation fails
    //! @note If capture fails for any stage, the inference can proceed without CUDA graph capture,
    //! but at cost of performance degradation.
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    /*!
     * @brief Handle generation request
     * @param request Generation request with prompts and parameters
     * @param response Output response with generated tokens and text
     * @param stream CUDA stream
     * @return True on success, false on failure
     * @throws std::runtime_error if an LLM or CUDA operation fails
     */
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream,
        bool outputThinkerEmbeddings = false);

    /*!
     * @brief Generate and save system prompt KV cache (public API matching standard runtime signature)
     * @param prompt The system prompt to generate the KVCache
     * @param loraWeightsName The name of the LoRA weights
     * @param stream The CUDA stream used for the generation
     * @return True if the KVCache is generated and saved successfully, false otherwise
     * @throws std::runtime_error if a CUDA operation fails
     */
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    /*! \brief Set the random seed used when initializing the action diffusion noise trajectory
     *  \param seed Random seed value; has no effect if no action runner is loaded
     */
    void setActionNoiseSeed(int32_t seed) noexcept;

    //! Get LLM prefill stage metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const noexcept
    {
        return mPrefillMetrics;
    }

    //! Get speculative decoding generation stage metrics (only meaningful when draft model is present)
    metrics::SpecDecodeGenerationMetrics const& getSpecDecodeGenerationMetrics() const noexcept
    {
        return mSpecDecodeGenerationMetrics;
    }

    char const* getSpeculativeDecodingStrategyName() const noexcept
    {
        return mDecoderRegistry ? mDecoderRegistry->speculativeDecoderName() : "vanilla";
    }

    //! Get vanilla generation stage metrics (only meaningful when no draft model / vanilla path)
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const noexcept
    {
        return mGenerationMetrics;
    }

    //! Get multimodal metrics (returns empty metrics if no multimodal runner)
    metrics::MultimodalMetrics getMultimodalMetrics() const noexcept
    {
        return mVisionRunner ? mVisionRunner->getMultimodalMetrics()
            : mAudioRunner   ? mAudioRunner->getMultimodalMetrics()
                             : metrics::MultimodalMetrics{};
    }

    //! Get the embedding table (for Talker streaming pipeline)
    rt::Tensor const& getEmbeddingTable() const
    {
        return mEmbedding.table;
    }

    //! @brief Get a base model hidden-states buffer for the requested layer index.
    //!
    //! Buffers are owned by the runtime and reused across requests. Layer 0 corresponds to
    //! the post-multimodal input embeddings (backed up before the decode loop reshapes them);
    //! other layer indices correspond to engine-output hidden states (e.g. acceptHiddenLayer
    //! for the Qwen3-Omni Talker, or future MTP layers).
    //!
    //! Lifetime contract:
    //!   - Buffers are sized to {maxRuntimeBatchSize, maxSupportedInputLength, hiddenSize}.
    //!   - Contents are cleared (overwritten) at the start of each handleRequest() call and
    //!     remain valid until the next handleRequest() begins. The buffer is reshaped to
    //!     {activeBatchSize, prefillLength, hiddenSize} for the most recent request — use
    //!     getBaseModelPrefillLength() to query the valid prefill length.
    //!   - The caller is responsible for consuming the data within that window.
    //!
    //! @param layerIdx Layer index. 0 = input embeddings (post-multimodal); other indices are
    //!                 model-specific (e.g. acceptHiddenLayer for Qwen3-Omni Talker).
    //! @return Pointer to the buffer, or nullptr if no buffer is registered for that layer.
    rt::Tensor const* getBaseModelHiddenStates(int32_t layerIdx) const noexcept
    {
        auto it = mHiddenStatesRegistry.find(layerIdx);
        return it != mHiddenStatesRegistry.end() ? it->second : nullptr;
    }

    //! @brief Number of valid prefill tokens in the hidden-states buffers from the most
    //! recent handleRequest() call. Returns 0 if no hidden-states output was requested.
    int32_t getBaseModelPrefillLength() const noexcept
    {
        return mLastPrefillLength;
    }

    //! @brief Per-batch input token IDs from the most recent handleRequest() call.
    //! Cleared at the start of each handleRequest(); valid until the next one begins.
    std::vector<std::vector<int32_t>> const& getBaseModelInputTokenIds() const noexcept
    {
        return mLastInputTokenIds;
    }

    //! @brief Check if draft model is loaded and spec-decode is available
    bool hasDraftModel() const noexcept
    {
        return mDecoderRegistry && mDecoderRegistry->hasSpeculativeDecoder();
    }

    //! Result of runVisionOnly(): the vision encoder's output embedding and per-layer
    //! deepstack residual features for the request's image(s).
    struct VisionOnlyResult
    {
        bool success{false};
        Tensor const* outputEmbedding{nullptr}; //!< [numImageTokens, hiddenSize], GPU-resident.
        OptionalInputTensors deepstackFeatures;  //!< One [numImageTokens, hiddenSize] tensor per deepstack layer.
    };

    /*!
     * @brief Run only the vision multimodal encoder for the request's image(s), skipping text
     * tokenization and M-RoPE generation.
     *
     * Reuses MultimodalRunner::preprocess's own native image loading/patchification/RoPE logic
     * via its existing `imageOnly=true` flag, for host-driven pipelines that assemble
     * `inputs_embeds` themselves (e.g. a robotics VLA harness injecting an extra modality) and
     * only need the vision tower's raw output.
     *
     * @param request Generation request; only `imageBuffers` on each `Request` is consulted.
     * @param stream CUDA stream for the vision engine's preprocessing/inference.
     * @return Result with `success=false` if there is no vision runner loaded, or if
     *         preprocessing/inference failed. The returned tensors are non-owning views into
     *         runtime-internal buffers, valid until the next call that uses the vision runner.
     */
    VisionOnlyResult runVisionOnly(LLMGenerationRequest const& request, cudaStream_t stream);

    //! Result of runBackboneRawForward(): raw engine outputs for one caller-assembled prefill.
    struct RawForwardResult
    {
        bool success{false};
        Tensor const* outputLogits{nullptr};
        Tensor const* outputHiddenStates{nullptr};
    };

    /*!
     * @brief Run one independent, one-off prefill forward pass of the base decoder engine given
     * caller-assembled embeddings, bypassing token-embedding lookup and sampling.
     *
     * For a host-driven harness that assembles `inputsEmbeds` itself (text-embedding lookup +
     * vision-embedding splice + a custom extra modality) and wants raw `hidden_states`/`logits`
     * back without generation/streaming. Each call resets the base engine's KV cache to empty
     * first, so calls are mutually independent -- this is a raw-forward escape hatch for one
     * sequence at a time (batch size must be exactly 1), not an autoregressive decode path.
     *
     * @param inputsEmbeds Host tensor, [1, seqLen, hiddenSize], FP32 (cast to the engine's
     *        activation dtype internally, on the GPU, before the forward pass -- callers should
     *        NOT pre-cast; passing FP32 directly avoids a scalar host-side cast loop).
     * @param deepstackEmbeds Host tensors, one per deepstack layer configured on the base
     *        engine, each [1, seqLen, hiddenSize], FP32 (same internal-cast note as above).
     *        Pass empty when the request has no image.
     * @param mropeCosSin Host tensor holding the caller-computed position encoding, in the
     *        layout `mPipelineIO->mropeCosSin` expects. Ignored for non-MRope base engines.
     * @param stream CUDA stream for the H2D copies and engine execution.
     * @return Result with `success=false` on any validation or engine failure. The returned
     *         tensors are non-owning views into runtime-internal buffers, valid until the next
     *         call into this runtime.
     */
    RawForwardResult runBackboneRawForward(Tensor const& inputsEmbeds, OptionalInputTensors const& deepstackEmbeds,
        Tensor const& mropeCosSin, cudaStream_t stream);

    //! Result of runFullPreprocessOnly(): the fully-assembled prefill inputs the base decoder
    //! engine would consume, captured just before mBaseExecutor->prepare/execute.
    struct FullPreprocessResult
    {
        bool success{false};
        Tensor const* inputsEmbeds{nullptr}; //!< [1, seqLen, hiddenSize], GPU-resident.
        OptionalInputTensors deepstackEmbeds; //!< One [1, seqLen, hiddenSize] tensor per deepstack layer.
        Tensor const* mropeCosSin{nullptr};   //!< Position-encoding tensor in the engine's own layout.
        std::vector<int32_t> idsInput;        //!< Effective (post system-prompt-reuse) token ids, batch size 1.
    };

    /*!
     * @brief Run full request preprocessing (chat template, tokenize, vision/audio encode,
     * M-RoPE, embedding + deepstack assembly) exactly as handleRequest() would, then return
     * before the base decoder engine actually executes.
     *
     * For diffing the engine's own preprocessing against an eager reimplementation: the caller
     * gets the runtime's real, fully-assembled `inputsEmbeds`/`deepstackEmbeds`/`mropeCosSin`,
     * which can be fed unmodified into runBackboneRawForward() to isolate the backbone decoder
     * + M-RoPE from embedding-assembly differences. Batch size must be exactly 1.
     *
     * @param request Generation request (image/text as normal).
     * @param stream CUDA stream for preprocessing.
     * @return Result with `success=false` on any preprocessing/setup failure. The returned
     *         tensors are non-owning views into runtime-internal buffers, valid until the next
     *         call into this runtime.
     */
    FullPreprocessResult runFullPreprocessOnly(LLMGenerationRequest const& request, cudaStream_t stream);

    /*!
     * @brief Overwrite one token position of the just-preprocessed `inputsEmbeds` in place, on
     * the device -- for injecting a third-modality (e.g. robot-state) embedding that has no slot
     * in runFullPreprocessOnly()'s own preprocessing.
     *
     * Must be called after runFullPreprocessOnly() and before
     * runBackboneRawForwardFromPreprocessed() (both operate on the same persistent
     * `mPipelineIO->inputsEmbeds` buffer). Copies only `stateEmbeddingHost`'s `hiddenSize`
     * elements (H2D + GPU cast) -- NOT a round-trip of the full `[1, seqLen, hiddenSize]` tensor,
     * which stays device-resident throughout (see runBackboneRawForwardFromPreprocessed()).
     *
     * @param position Token index to overwrite, in [0, seqLen).
     * @param stateEmbeddingHost Host tensor, [hiddenSize], FP32.
     * @param stream CUDA stream.
     * @return false if `position` is out of range or the shapes don't match.
     */
    bool injectStateEmbedding(int32_t position, Tensor const& stateEmbeddingHost, cudaStream_t stream);

    /*!
     * @brief Like runBackboneRawForward(), but consumes `mPipelineIO->inputsEmbeds` /
     * `deepstackEmbeds` / `mropeCosSin` as already populated by a preceding
     * runFullPreprocessOnly() (+ optional injectStateEmbedding()) call, instead of taking them as
     * parameters to copy in -- skips the host round-trip for these tensors entirely (the caller
     * would otherwise need to pull them out as numpy, patch one token host-side, and copy the
     * whole tensor back in, which is exactly the redundant device->host->device hop this avoids).
     *
     * Restricted to this runtime's actual current preprocessing state: unlike
     * runBackboneRawForward() (which takes an explicit possibly-empty `deepstackEmbeds`), this
     * assumes deepstack features are real whenever `mPipelineIO->deepstackEmbeds` is non-empty --
     * correct for a vision-always caller (e.g. a robot with cameras on every request), not a
     * general text-or-vision request mix.
     *
     * @param stream CUDA stream for engine execution.
     * @return Result with `success=false` on any validation or engine failure. The returned
     *         tensors are non-owning views into runtime-internal buffers, valid until the next
     *         call into this runtime.
     */
    RawForwardResult runBackboneRawForwardFromPreprocessed(cudaStream_t stream);

private:
    //! @brief Common initialization logic shared between both constructors
    void initializeCommon(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap,
        std::optional<SpecDecodeDraftingConfig> const& draftingConfig, cudaStream_t stream);

    //! @brief Capture a CUDA graph on the base executor for the default (no-adapter)
    //! state, then one additional graph per registered LoRA adapter. Returns the
    //! logical AND of all captures — any single failure flips the aggregate to
    //! false but capture continues for remaining adapters (graceful degrade).
    bool captureBaseGraphWithLoraFanout(InferenceDims const& dims, cudaStream_t stream);

    //! @brief Build the strategy runtime reference bundle after common resources are allocated.
    void buildDecodingRuntimeContext();

    rt::Tensor mSharedExecContextMemory{}; //!< Shared device memory for all execution contexts
    int32_t mMaxRuntimeBatchSize{1};       //!< Maximum runtime batch size
    bool mHasFFPALayer{false};             //!< True if any attention layer uses headDim=512 (FFPA)

    DeploymentConfig mDeployment{};                    //!< Parsed base+draft configs + consolidated strategy settings
    std::unique_ptr<EngineExecutor> mBaseExecutor;     //!< Base model TRT wrapper
    std::unique_ptr<SharedResources> mSharedResources; //!< KV caches / RoPE / LoRA / context memory
    std::unique_ptr<PipelineIO> mPipelineIO;           //!< Per-pipeline I/O tensors
    // FP32 device scratch for runBackboneRawForward()'s H2D-then-GPU-cast path (caller-supplied
    // inputsEmbeds/deepstackEmbeds arrive as FP32; cast to the engine's FP16 activation dtype on
    // the GPU instead of a scalar host-side loop). Reused/grown across calls.
    rt::Tensor mRawForwardInputsEmbedsFp32Scratch{};
    std::vector<rt::Tensor> mRawForwardDeepstackFp32Scratch;
    // FP32 device scratch for injectStateEmbedding() -- a single [hiddenSize] row, not a full
    // [1, seqLen, hiddenSize] tensor.
    rt::Tensor mStateEmbeddingFp32Scratch{};
    TensorMap mBaseTensorMap;                          //!< Base engine binding map
    std::unique_ptr<DecodingRuntimeContext> mDecodingRuntimeContext;
    std::unique_ptr<DecoderRegistry> mDecoderRegistry;
    std::unique_ptr<StepPreparer> mStepPreparer;             //!< Per-step sequence preprocessor
    std::unique_ptr<EmbeddingPreprocessor> mEmbeddingPre;    //!< Embedding-lookup preprocessor
    std::unique_ptr<Gemma4EmbeddingPreprocessor> mGemma4Ple; //!< Gemma4 PLE token-identity preprocessor
    //! Base-engine deepstack binding (nullptr when the base engine was built
    //! without deepstack features). Swaps between `io.deepstackEmbeds[i]`
    //! (prefill) and the shared `zeroDeepstackBroadcast` (all other phases).
    std::unique_ptr<DeepstackBinding> mDeepstack;

    std::unique_ptr<MultimodalRunner> mVisionRunner{nullptr};      //!< Vision multimodal runner (optional)
    std::unique_ptr<MultimodalRunner> mAudioRunner{nullptr};       //!< Audio multimodal runner (optional)
    std::unique_ptr<Alpamayo1ActionRunner> mActionRunner{nullptr}; //!< Action/diffusion head runner (optional)
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer;              //!< Tokenizer
    hash_utils::HashMap<std::tuple<std::string, std::string>, SystemPromptKVCache>
        mSystemPromptKVCacheBase;          //!< System prompt KVCache for base model
    std::string mEmptyLoraWeightsName{""}; //!< Empty LoRA weights name for default case

    // Pre-define key runtime GPU tensors and initialize them during construction.
    // [1] Runtime-local I/O tensors. Embedding table is shared between base and draft models.
    // Core per-pipeline tensors (inputsEmbeds, outputLogits, deepstackEmbeds, baseHiddenStates,
    // draftHiddenStatesIn/Out, contextLengths, mropeCosSin) live on `mPipelineIO`.
    EmbeddingData mEmbedding; //!< Embedding table [vocabSize, hiddenSize] and optional FP8 scales
    rt::Tensor mIdsInput;     //!< Input token IDs (used for embedding lookup)

    // [2] Sampling workspace and output tensors that used across all the sampling operations.
    rt::Tensor mSamplingWorkspace;
    rt::Tensor mSamplingIndices;
    rt::Tensor mSamplingScores;
    rt::Tensor mBaseVocabMappingTable; // Vocab mapping table for base model reduced vocab (empty if not used)

    // [3] Batch eviction support tensors.
    rt::Tensor mDeviceBatchMapping;

    // [4] Host pinned memory tensors for optimized CPU-GPU memory transfers
    rt::Tensor mHostPackedTokenIds;      //!< Host pinned memory for packed token IDs
    rt::Tensor mHostSelectedTokenIds;    //!< Host pinned memory for selected token IDs from sampling
    rt::Tensor mHostReuseKVCacheLengths; //!< Host pinned memory for reuse KV cache lengths

    // [5] Multimodal support tensors for audio/image token indexing
    rt::Tensor mMultimodalIndices; //!< Multimodal indices tensor [batchSize, seqLen] for audio/image embeddings

    // [8] Base model hidden states portal (Qwen3-Omni audio generation, future MTP).
    //     The actual buffers (engine-output and prefill-embeddings backup) live on
    //     PipelineIO so they can be wired into the engine TensorMap. The registry
    //     below holds non-owning pointers into those buffers, populated per request,
    //     plus the per-request prefill length and the raw input token ids — all are
    //     transient request-scoped state, not pipeline tensors. See
    //     getBaseModelHiddenStates() for the lifetime contract.
    std::unordered_map<int32_t, rt::Tensor const*> mHiddenStatesRegistry; //!< Per-request layer→buffer map
    int32_t mLastPrefillLength{0};                                        //!< Valid prefill length in buffers
    std::vector<std::vector<int32_t>> mLastInputTokenIds;                 //!< Per-batch input token IDs

    //! @brief Restore recurrent/conv states from a cached system prompt.
    void restoreRecurrentStates(int32_t batchIdx, SystemPromptKVCache const& cachedStates, cudaStream_t stream);

    //! @brief Zero all recurrent/conv states for a given batch index.
    void zeroRecurrentStates(int32_t batchIdx, cudaStream_t stream);

    //! @brief Zero embedding/PLE tensors at padding positions before engine execution.
    //! FFPA (headDim=512) has no cu_seqlens support and processes all positions uniformly,
    //! so non-zero padding embeddings cause fp16 overflow → NaN propagation.
    void zeroPaddingForFFPA(
        int32_t const* contextLengths, int32_t batchSize, int32_t inputIdsLength, cudaStream_t stream);

    // Key functions to drive the runtime, defined in a consumer-producer pattern.
    // Consume tokenized IDS as input and produce hidden states for the whole sequence and first generated token.
    //! @throws std::runtime_error if a CUDA error occurs
    bool runBaseModelPrefill(DecodingInferenceContext& context);

    //! Validate request shape/runtime compatibility.
    bool validateRequestConfig(LLMGenerationRequest const& request);

    //! Prepare per-request runtime state for models built with multimodal support.
    //! Runs multimodal preprocessing when audio or vision inputs are present.
    //! For text-only requests on MRope-based multimodal models, restores text-only RoPE state
    //! and clears stale multimodal request state.
    bool multiModalRuntimePreprocess(
        LLMGenerationRequest const& request, DecodingInferenceContext& context, cudaStream_t stream);

    // Consume system prompt, produce the hash table of system prompt KVCache if kv cache reuse is enabled.
    //! @throws std::runtime_error if a CUDA operation fails
    bool genAndSaveSystemPromptKVCache(DecodingInferenceContext& context, int32_t genAndSaveBatchIdx);

    // Consume batched input ids and the hash table of system prompt KVCache, produce the padded input ids and input
    // lengths. Instantiate the KVCache from the hash table if the system prompt has been cached.
    //! @throws std::runtime_error if system prompt is malformed
    bool setUpForPrefillExecution(DecodingInferenceContext& context, DecodingStrategy& strategy);

    // Batch eviction support
    //! @brief Perform batch eviction
    //! @param context Inference context
    //! @return True on success, false on failure
    //! @throws std::runtime_error if a CUDA error occurs
    bool performBatchEvict(DecodingInferenceContext& context, DecodingStrategy& strategy);

    // Stage-specific metrics
    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::SpecDecodeGenerationMetrics mSpecDecodeGenerationMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics; //!< Vanilla generation metrics (used when no spec-decode)
};

} // namespace rt
} // namespace trt_edgellm

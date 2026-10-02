# Literature review for the V17 rate-proxy screen

Registered 2026-10-02 before either V17 GPU submission. This review adapts components to the existing learned preprocessor; it is not a reproduction of the complete cited systems. The acceptance metric remains real-codec task BD-rate, not a proxy or perceptual metric.

## Sources and reading scope

Twelve local PDFs were triaged from `D:/STUDY/LAB/News` and `D:/STUDY/LAB/bao_1/papers`. The companion `V17_literature_sources.json` records original paths, physical page counts, SHA256 hashes and reading scope. Text extraction and three rendered formula pages stay in local scratch `D:/STUDY/LAB/bao_1/output/research_v17_20261002/`; PDF originals were not edited or copied into experiment payloads. The scanned Adaptive High-Frequency paper has poor text extraction: its physical page2 was visually read, not an asserted full-text review. Gaussian filtering was triaged only. Page numbers below refer to physical local PDF pages.

The Lu local journal paper (December2024) differs from the [NeurIPS2022 preprint](https://arxiv.org/abs/2206.05650). The [SECVCM record](https://arxiv.org/abs/2510.15347) is v4, revised June2026 and accepted by TIP; older v1 summaries should not supply its current author list. The Zhao [preprint record](https://arxiv.org/abs/2512.15331) notes a DCC2024 poster despite its December2025 submission date. The exact source version used for V17 equations is the local 20-page Sandwiched PDF, SHA256 a7f57a9f319ebe77b30eb1baa7b7e2e91f7acb287b55bfdb72e41ee2a874e4e1, physical page17, Eqs39/40. The [author-hosted paper](https://duruofei.com/papers/Guleryuz_SandwichedCompression-RepurposingStandardCodecsWithNeuralNetworkWrappers_2024.pdf) and [official implementation](https://github.com/google/sandwiched_compression) provide primary online references; no official implementation code is copied here.

## Methods compared against this repo

| Source / pages read | Relevant method | Consequence for this round |
|---|---|---|
| Zhao, A Preprocessing Framework, pp4-6 | Frozen machine analyzers; task, distortion and rate loss; spatial/temporal preprocessing; differentiable motion/transform/quantization proxy with a factorized entropy prior | Repo already has the common task loss, spatial/temporal edits and motion option. Replacing the architecture would invalidate the fixed-source comparison. The entropy modeling difference deserves attention. |
| Lu, Preprocessing Enhanced, p5 and training-method passages; local NeurIPS text | Real BPG forward values with differentiable proxy gradients, plus separate proxy fine-tuning | Real-forward STE already exists. Proxy gradients still need scrutiny: substituting the true forward scalar alone cannot establish a faithful rate gradient. Their image/BPG training is not this AR/H264 experiment. |
| Sandwiched Compression, p17 visually checked | Sum of logarithmic transform-coefficient costs; scalar fitting to real JPEG rate to weight derivatives | Supplies the two small V17 interventions below. The paper uses per-image calibration; the video implementation uses the existing batch-mean bpp. PRE-only scope excludes the paper's postprocessor. |
| GOP-based Deep Preprocessing, pp2-3 visually checked | GOP-aligned proxy training, relative rate fitting and distortion feedback | Supports auditing proxy/real scale, but its learned proxy, controller and perceptual objective would be a larger experiment. V17 does not claim to reproduce its GOP model. |
| Semantic Preprocessor, p2 | Semantic priors, shared preprocessing, rate/task/feature terms | V16 already tested raw-codec feature fidelity at weight0.5: partial recovery after pixel release, no gate pass. Avoid repeating a semantic-loss weight without a new mechanism. |
| Task-Switchable Pre-Processor, pp3-4 | Shared preprocessing trunk with task-conditioned modulation | Fits a future common OD/AR pipeline, but does not prove one unconditional shared checkpoint. Requires aligned task adapters/data and a separately registered architecture experiment. |
| Feature-Based RDO, pp1-2 | Feature distortion changes standard encoder partition/RDO decisions | Encoder intervention falls outside the current fixed-medium, fixed-QP preprocessing comparison. Do not alter encoder decisions to claim this gate. |
| Symmetric Entropy-Constrained VCM, pp1,3-5 | Bidirectional conditional entropy between codec semantic features and a vision backbone, built on a learned video codec | Useful conceptual motivation for retaining task information, but requires learned context/entropy decoding. It is not a drop-in loss for frozen H264/H265 preprocessing. |
| AccMPEG, abstract and quality-selection passages | DNN-guided macroblock quality allocation | Changes encoding quality spatially; defer under the fixed-QP protocol. |
| RPP, pp3-4 method passages | Adaptive DCT/high-frequency perceptual preprocessing | Repo already has adaptive DCT regularization. Human perception results do not establish AR efficiency; no blind increase of DCT weight. |
| Adaptive High-Frequency, p2 visually checked | Parameterized smoothing/sharpening with rate/perception supervision | Not selected: lacks this task-regret guarantee, and source architecture remains fixed. |

Published BD-rate gains use different tasks, codec settings, populations and training budgets. They are not predicted repo outcomes and are not acceptance evidence. No reported paper gain is transferred to these screens.

## Existing rate-model lineage

`src/models/entropy_codec.py` already contains D8 `LearnedRateCodec`, a per-frequency Laplacian mixture prior, with its own optimizer in the common engine and optional prior freezing. `configs/additive_attn_ar.yaml` combines it with an attentive architecture and motion compensation. That configuration selects `codec.kind: entropy`, so it is proxy-only; it does not put H264/H265 in the training forward pass. Its comments discuss historical learning failures, but no independently audited result for that exact configuration was located in the current curated docs. Do not elevate comments to fresh experimental evidence.

V17 leaves D8 in place. It adds a parameter-free logarithmic option to the already deployed VirtualCodec+STE route, retaining the source architecture and one existing real encode per call. This separates the rate-cost and derivative-scale questions without adding a jointly trained prior or changing forward reconstruction.

## Selected intervention and limits

The current Gaussian-power cost aggregates normalized coefficient power by channel before logarithmic costing. It includes the DC coefficient; a missing-DC diagnosis would be incorrect. The alternative costs each normalized coefficient `y = coeff / step` as `log2(1 + abs(y))`, sums over coefficients, and retains the existing bpp normalization. Reconstruction, quantization, prediction, chroma sampling, motion settings and RNG consumption stay identical. This smooth magnitude proxy is not an entropy coder or an exact rate derivative; the abs derivative is zero exactly at zero.

For calibration, compute `a = (real_bpp / max(proxy_bpp.detach(), 1e-8)).detach()`. Scale only the proxy rate derivative by `a`; preserve the real codec's rate forward value and the existing reconstruction STE. Detaching the entire multiplier is essential: differentiating through the quotient would cancel the intended derivative when the denominator equals the proxy. The denominator floor is fixed before the run, with no hidden multiplier cap; logs expose floor activation, scale, proxy bpp and real bpp. Non-finite or negative rate inputs fail clearly. The ratio is batch-mean video bpp, unlike the paper's per-image JPEG choice, and it remains content-dependent. Scalar value fitting cannot prove correct gradient direction across clips or codec modes.

The common codec/engine change is task-independent and available through the existing task API. V17 screens AR only. OD remains frozen; this is neither a joint OD/AR performance result nor a new OD feature adapter.

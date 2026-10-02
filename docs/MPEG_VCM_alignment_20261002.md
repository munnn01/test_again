# MPEG VCM source alignment and limits

Recorded 2026-10-02. These are short source summaries and an explicit local
adaptation, not a transcription of standards, a claim to have read the final
TR full text, a full MPEG CTC experiment or a standard-conformance report.

## Sources inspected

The official [ISO catalogue for ISO/IEC TR 23888-3:2026](https://www.iso.org/standard/89045.html)
lists its first edition as published in June 2026. Its metadata/abstract
concern encoder and receiver optimization for machine analysis. Only the
public catalogue was read; the final paid full text was not accessed. The
local older committee draft below must not be represented as that final text.

The public [w24254 CDTR package](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/147_Sapporo/w24254.zip)
contains the older 2024 draft of TR23888-3, read via local tr_draft.md.
Its section7.5 describes content denoising as an option to limit wasted
rate and preserve machine cues, with detail-preserving filter examples and
cautions about uniform filtering. It neither supplies our residual-Gaussian
algorithm nor establishes an AR result. Verified document SHA-256:
b3426c94851662fc991c8c2f9c89859d24d5a1ae7d1ee33b2777d29133de1563
for WG5_N0307_JVET-AI2030-v1.docx. This is a document hash, not the zip hash.

The public [MPEG155 CTC package w26718](https://www.mpeg.org/wp-content/uploads/mpeg_meetings/155_Geneva/w26718.zip)
was read via local ctc.md (WG04N0870, serial 26718, internal date
2026-09-07, recommended for the 155th and following meetings). Its stale
July 2026 header is not treated as an exact publication date. It specifies
matched reference/anchor conditions,
task-specific datasets and networks, six quality points, reporting and
separate training/test material for MPEG technology comparisons. It covers
object detection, tracking and optional segmentation with VVC-based
reference configurations. Verified WG04N0870_VCM_CTC.docx SHA-256:
00e10ee3c0ae18571f420270b2e3580c72aea763732b1a542125d5e91edb4f02.
Again this identifies the document, not its containing archive.

Original document bytes remain local under
D:/STUDY/LAB/bao_1/output/mpeg_vcm_20261002/. Only our compact paraphrases,
source identities and separately registered experimental design enter the
code/config package; original documents are not uploaded.

## Current scope differs from CTC

| Aspect | Inspected CTC | Current learned-AR screen |
|---|---|---|
| Task/data | Detection, tracking, optional segmentation; SFU-HW/TVD/Panda and listed image data | AR on existing Kinetics clips; this screen supplies no new OD evidence |
| Codec/configuration | VVC-based reference/anchor and RA/LD/AI conditions | Shared H.264/H.265 preprocessing pipeline; these two screens use real H.264 medium rather than a complete VVC configuration |
| Quality grid | Six anchor points, video sequence-dependent starting QPs and intervals | Fixed five QPs30/35/40/45/50; scorer and grid remain unchanged |
| BD reporting | Curve fitting then middle four points of the six-point grid | Existing all-five-QP aggregate curve/scorer unchanged; CTC exclusions are not imported |
| Task/rate reporting | Task-specific metrics, video kbps and reporting templates | Held-out Top1, bpp-based aggregate curves and repo paired evidence |
| Test exposure | Separate prescribed train/test materials | Fixed dev208 and reserved802/current-search complement; historical validation exposure disclosed |
| Runtime | Encoder/decoder and individual-module timing reports | Full CTC runtime accounting is absent; separate local CPU preprocessor latency measures only that component |

For some CTC end-to-end proposals, a per-sequence ±10% anchor-rate window
serves point matching. That is a different quantity from our user acceptance
criterion BD-rate <= -10%; it is not permission to relax, reinterpret or
import a different BD calculation into this screen. Required CTC overlap
handling, extra points and templates also do not retrospectively change our
fixed registered QP grid or scorer. A future CTC study needs its own design,
reference settings, task datasets, kbps measurement and complete runtime
reports. Our canonical bpp uses existing elementary-bitstream bytes and
constant clip geometry; no container accounting change is introduced.
Physical kbps must not be inferred from unchecked sampled-Kinetics FPS.
The separate local CPU preprocessor measurement does not establish encoder,
decoder, full-pipeline or standards runtime performance.

## Our independent adaptation

[V19 design](RUN_DESIGN_autonomous_ar_v19.md) tests fixed spatial Gaussian
smoothing of the signed learned RGB residual, followed by source-plus-edit
and the existing clamp, with sigma 0.5 versus 1.0. It retains the algebraic
source path and uses no new learned weights, source denoising or temporal
filter. The draft's general denoising concept motivates a bounded test;
it does not endorse these sigmas or guarantee preserved recognition.
Useful edit cues can be attenuated, compensated by learned amplitude, or
changed by clamping. Lower residual HF energy alone is not measured rate
or machine-task evidence.

The repository already studied residual-spectrum cost and gamma_res TV
in [RUN_DESIGN_additive.md](RUN_DESIGN_additive.md) sections5.3/5.4. V19's
explicit structural filter on the V18A calibrated-log recipe is distinct
from those historical sweeps and from frozen OD background/decoder blur.
That connection does not establish bit savings, OD transfer or standards
conformance. Only the original real-codec dev gate,
and then immutable-checkpoint reserved802/full1010 confirmation, can support
the requested improvement. Account/data/document permissions and the
existing prohibition on new goals, recurring checks and GitHub pushes
remain independent of these source documents.

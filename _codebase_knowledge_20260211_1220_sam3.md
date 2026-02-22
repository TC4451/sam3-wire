# Codebase Knowledge - sam3

## Session and repo info
- Generated: 2026-02-11 12:20 (local)
- Repo root: `/home/daizi/sam3`
- Scope: full deep read with emphasis on call flow, data/state mutation, math/algorithm mapping, and failure/performance risks.
- Core package: `sam3/` plus train/eval/agent stacks.

## Repo purpose (high confidence)
- SAM 3 is implemented here as a unified promptable segmentation stack for image and video, supporting text + geometric prompts and detector/tracker composition.
- README explicitly states: unified model for image/video, open-vocabulary segmentation with text/exemplar prompts, presence-token-based discrimination, and decoupled detector-tracker design.
- The model section states detector + tracker with shared vision encoder and SAM2-style tracker backbone.

Evidence:
- `README.md :: README (overview) [L49-L52]`
- `README.md :: README (model section) [L186-L189]`
- `pyproject.toml :: project metadata [L5-L10]`

## Repo map (directory -> role)
- `sam3/model`: core model code (builder, detector, tracker, video orchestration, encoders/decoders, geometry, segmentation heads).
- `sam3/train`: hydra/submitit training entrypoint, trainer runtime, losses/matchers, dataset loaders/collator.
- `sam3/eval`: postprocessing and COCO-style online evaluator.
- `sam3/perflib`: optional accelerated kernels (NMS/CC/association/compile wrappers) with CPU/triton fallbacks.
- `sam3/agent`: agentic orchestration around SAM3 service calls + iterative mask filtering.
- `examples`, `scripts`, `assets`: notebooks, scripts, and benchmark/demo assets.

Evidence:
- `sam3/model_builder.py :: build_sam3_image_model [L560-L643]`
- `sam3/train/train.py :: main [L140-L338]`
- `sam3/eval/postprocessors.py :: PostProcessImage [L32-L324]`
- `sam3/perflib/nms.py :: nms_masks [L24-L55]`
- `sam3/agent/agent_core.py :: agent_inference [L124-L565]`

## Backbone workflows

### 1) Image grounding path
1. `build_sam3_image_model` assembles backbone, text encoder, transformer, scoring head, geometry encoder, and optional segmentation/instance-interactivity.
2. `Sam3Processor._forward_grounding` drives inference and thresholding on `pred_logits` (with optional presence gating) and rescales boxes/masks to original image size.
3. `Sam3Image.forward_grounding` orchestrates prompt encoding -> encoder -> decoder -> segmentation heads.
4. Decoder outputs include boxes/logits/presence features; segmentation head returns masks + semantic/presence side outputs.

Evidence:
- `sam3/model_builder.py :: build_sam3_image_model [L560-L643]`
- `sam3/model/sam3_image_processor.py :: Sam3Processor._forward_grounding [L183-L222]`
- `sam3/model/sam3_image.py :: Sam3Image.forward_grounding [L439-L491]`
- `sam3/model/sam3_image.py :: Sam3Image._encode_prompt [L166-L210]`
- `sam3/model/sam3_image.py :: Sam3Image._run_encoder [L211-L250]`
- `sam3/model/sam3_image.py :: Sam3Image._run_decoder [L251-L298]`
- `sam3/model/sam3_image.py :: Sam3Image._update_scores_and_boxes [L299-L384]`
- `sam3/model/sam3_image.py :: Sam3Image._run_segmentation_heads [L385-L424]`
- `sam3/model/maskformer_segmentation.py :: UniversalSegmentationHead.forward [L272-L327]`

### 2) Video propagation path (detector + tracker orchestration)
1. `build_sam3_video_model` creates detector (Sam3ImageOnVideoMultiGPU) + tracker (SAM2-like predictor) and wraps in `Sam3VideoInferenceWithInstanceInteractivity`.
2. `Sam3VideoInference.init_state` builds inference session state (frames, metadata, caches).
3. `propagate_in_video` loops frame-by-frame; each frame calls `_run_single_frame_inference`.
4. `_run_single_frame_inference` calls `_det_track_one_frame` (in `Sam3VideoBase`) which executes:
   - backbone + detector (with chunked multigpu prefetch buffer),
   - tracker propagation,
   - planning heuristics (association/recondition/suppression/hotstart/limits),
   - execution phase (add/remove tracker objects),
   - output assembly.
5. `_postprocess_output` converts masks/scores/boxes into API format and applies filtering/hiding rules.

Evidence:
- `sam3/model_builder.py :: build_sam3_video_model [L653-L793]`
- `sam3/model/sam3_video_inference.py :: init_state [L55-L91]`
- `sam3/model/sam3_video_inference.py :: propagate_in_video [L251-L357]`
- `sam3/model/sam3_video_inference.py :: _run_single_frame_inference [L358-L429]`
- `sam3/model/sam3_video_base.py :: Sam3VideoBase._det_track_one_frame [L152-L293]`
- `sam3/model/sam3_video_base.py :: run_backbone_and_detection [L313-L401]`
- `sam3/model/sam3_video_base.py :: run_tracker_propagation [L402-L453]`
- `sam3/model/sam3_video_base.py :: run_tracker_update_planning_phase [L506-L819]`
- `sam3/model/sam3_video_base.py :: run_tracker_update_execution_phase [L893-L935]`
- `sam3/model/sam3_video_base.py :: build_outputs [L936-L1015]`
- `sam3/model/sam3_video_inference.py :: _postprocess_output [L430-L523]`

### 3) Multi-GPU detector chunking
- `forward_video_grounding_multigpu` maintains a chunk cache keyed by frame index, reads current chunk, evicts previous chunk, prebuilds next chunk.
- `_build_multigpu_buffer_next_chunk` computes local frame, optionally runs mask NMS, and all-gathers detection tensors (and optional SAM2 backbone features) into per-frame cache.

Evidence:
- `sam3/model/sam3_image.py :: Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu [L698-L788]`
- `sam3/model/sam3_image.py :: Sam3ImageOnVideoMultiGPU._build_multigpu_buffer_next_chunk [L789-L869]`
- `sam3/model/sam3_image.py :: Sam3ImageOnVideoMultiGPU._gather_tensor [L870-L880]`

### 4) Tracker memory mechanics
- `track_step` fuses current features with memory, runs SAM heads, and optionally encodes new memory.
- `_prepare_memory_conditioned_features` builds temporal memory prompt from conditioning/non-conditioning frames and object pointers with temporal encodings.
- `_encode_new_memory` encodes high-res masks into memory features and injects no-object embedding based on object score logits.

Evidence:
- `sam3/model/sam3_tracker_base.py :: track_step [L929-L1066]`
- `sam3/model/sam3_tracker_base.py :: _prepare_memory_conditioned_features [L559-L795]`
- `sam3/model/sam3_tracker_base.py :: _encode_new_memory [L796-L850]`

### 5) Instance interactivity extension
- `Sam3VideoInferenceWithInstanceInteractivity.propagate_in_video` chooses among full propagation, partial tracker propagation, or fetch-only based on action history.
- `add_tracker_new_points` supports add/refine flows, object-to-GPU assignment, rank broadcast of refined masks/scores, and cache update.

Evidence:
- `sam3/model/sam3_video_inference.py :: Sam3VideoInferenceWithInstanceInteractivity.propagate_in_video [L996-L1190]`
- `sam3/model/sam3_video_inference.py :: parse_action_history_for_propagation [L1222-L1279]`
- `sam3/model/sam3_video_inference.py :: add_tracker_new_points [L1400-L1611]`

### 6) Predictor service orchestration
- `Sam3VideoPredictor.handle_request` dispatches session/add/remove/reset/close request types.
- `Sam3VideoPredictorMultiGPU` spawns worker processes, starts NCCL process group, dispatches requests/stream requests to workers, and joins by barrier.

Evidence:
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictor.handle_request [L57-L89]`
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictor.propagate_in_video [L184-L228]`
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictorMultiGPU._start_worker_processes [L374-L412]`
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictorMultiGPU._start_nccl_process_group [L413-L435]`
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictorMultiGPU._worker_process_command_loop [L447-L509]`

### 7) Training and evaluation pipeline
- `sam3/train/train.py` is hydra + submitit launcher for local/cluster runs.
- `Trainer` manages env setup, DDP wrap, dataloaders, step loop, checkpoint save/load, val loop.
- Loss path: `Sam3LossWrapper.compute_loss` applies core/o2m loss composition and normalization.
- Matching path: `BinaryHungarianMatcherV2.forward` computes class/L1/GIoU costs with optional validity masking.
- Data path: `collate_fn_api`, image/video dataset loaders, COCO JSON adapters.
- Eval path: postprocessors convert outputs to eval format; `CocoEvaluator` incrementally updates/synchronizes/summarizes metrics.

Evidence:
- `README_TRAIN.md :: training overview [L3-L25]`
- `sam3/train/train.py :: main [L140-L338]`
- `sam3/train/trainer.py :: Trainer.__init__ [L148-L238]`
- `sam3/train/trainer.py :: Trainer._setup_ddp_distributed_training [L299-L321]`
- `sam3/train/trainer.py :: Trainer._run_step [L901-L967]`
- `sam3/train/trainer.py :: Trainer._save_checkpoint [L377-L395]`
- `sam3/train/loss/sam3_loss.py :: Sam3LossWrapper.compute_loss [L83-L160]`
- `sam3/train/matcher.py :: BinaryHungarianMatcherV2.forward [L483-L671]`
- `sam3/train/data/collator.py :: collate_fn_api [L137-L361]`
- `sam3/train/data/sam3_image_dataset.py :: CustomCocoDetectionAPI.load_queries [L262-L432]`
- `sam3/train/data/sam3_video_dataset.py :: VideoGroundingDataset._load_datapoint [L93-L161]`
- `sam3/train/data/sam3_video_dataset.py :: VideoGroundingDataset._sample_stage_ids [L162-L189]`
- `sam3/train/data/sam3_video_dataset.py :: VideoGroundingDataset._tile_single_image_data [L235-L300]`
- `sam3/train/data/sam3_video_dataset.py :: VideoGroundingDataset._subsample_queries [L301-L327]`
- `sam3/train/data/coco_json_loaders.py :: COCO_FROM_JSON.loadQueriesAndAnnotationsFromDatapoint [L153-L254]`
- `sam3/eval/postprocessors.py :: PostProcessImage.forward [L62-L152]`
- `sam3/eval/postprocessors.py :: PostProcessImage.process_results [L257-L324]`
- `sam3/eval/postprocessors.py :: PostProcessAPIVideo.process_results [L364-L546]`
- `sam3/eval/coco_eval.py :: CocoEvaluator.update [L166-L198]`
- `sam3/eval/coco_eval.py :: CocoEvaluator.synchronize_between_processes [L220-L250]`
- `sam3/eval/coco_eval.py :: CocoEvaluator.summarize [L283-L346]`

## Key symbols to know
- `sam3/model_builder.py :: build_sam3_image_model [L560-L643]` - image model assembly.
- `sam3/model_builder.py :: build_sam3_video_model [L653-L793]` - video model assembly with heuristics.
- `sam3/model_builder.py :: build_tracker [L434-L488]` - SAM2-derived tracker instantiation.
- `sam3/model/sam3_image_processor.py :: Sam3Processor._forward_grounding [L183-L222]` - image API scoring/filtering + rescale.
- `sam3/model/sam3_image.py :: Sam3Image.forward_grounding [L439-L491]` - detector forward orchestration.
- `sam3/model/decoder.py :: TransformerDecoder.forward [L407-L610]` - core decoder loop with iterative ref boxes and presence logits.
- `sam3/model/encoder.py :: TransformerEncoderFusion.forward [L515-L580]` - image/text fusion encoder.
- `sam3/model/geometry_encoders.py :: SequenceGeometryEncoder.forward [L717-L838]` - geometric prompt encoding.
- `sam3/model/maskformer_segmentation.py :: UniversalSegmentationHead.forward [L272-L327]` - mask + semantic + presence outputs.
- `sam3/model/sam3_image.py :: Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu [L698-L788]` - chunked multigpu detection caching.
- `sam3/model/sam3_video_base.py :: Sam3VideoBase._det_track_one_frame [L152-L293]` - per-frame detector-tracker execution.
- `sam3/model/sam3_video_base.py :: Sam3VideoBase.run_tracker_update_planning_phase [L506-L819]` - heuristics/association/reconditioning pipeline.
- `sam3/model/sam3_video_base.py :: Sam3VideoBase._associate_det_trk [L1161-L1298]` - detection-tracker association logic.
- `sam3/model/sam3_video_base.py :: Sam3VideoBase._process_hotstart [L1312-L1437]` - delayed output / early pruning.
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference.propagate_in_video [L251-L357]` - main user-facing propagation generator.
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference._compile_model [L574-L653]` - torch.compile integration and component-level wrapping.
- `sam3/model/sam3_video_inference.py :: Sam3VideoInferenceWithInstanceInteractivity.propagate_in_video [L996-L1190]` - partial propagation merge path.
- `sam3/model/sam3_tracker_base.py :: Sam3TrackerBase.track_step [L929-L1066]` - SAM2-like step + memory encoding.
- `sam3/model/sam3_tracking_predictor.py :: Sam3TrackerPredictor.add_new_points_or_box [L180-L341]` - interactive point/box flow.
- `sam3/model/sam3_tracking_predictor.py :: Sam3TrackerPredictor.propagate_in_video [L790-L874]` - batched object tracker propagation.
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictor.handle_request [L57-L89]` - request dispatch API.
- `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictorMultiGPU._worker_process_command_loop [L447-L509]` - worker command runtime.
- `sam3/train/train.py :: main [L140-L338]` - train launcher.
- `sam3/train/trainer.py :: Trainer._run_step [L901-L967]` - train forward/backward microstep.
- `sam3/train/loss/sam3_loss.py :: Sam3LossWrapper.compute_loss [L83-L160]` - loss aggregation.
- `sam3/train/matcher.py :: BinaryHungarianMatcherV2.forward [L483-L671]` - matching core.
- `sam3/eval/postprocessors.py :: PostProcessImage.forward [L62-L152]` - detection/mask postprocess.
- `sam3/eval/coco_eval.py :: CocoEvaluator.update [L166-L198]` - streaming metric update.

## Data/state model and mutation points

### Global session state (video inference)
- `inference_state` stores immutable metadata (`num_frames`, image size), mutable tracker metadata, caches (`feature_cache`, `cached_frame_outputs`), and action history.
- Mutation is concentrated in:
  - `init_state` and `_construct_initial_input_batch` (initialization),
  - `_run_single_frame_inference` (per-frame updates of tracker states/metadata),
  - `_cache_frame_outputs` (cache writes),
  - interactive methods (`add_prompt`, `add_tracker_new_points`, `remove_object`).

Evidence:
- `sam3/model/sam3_video_inference.py :: init_state [L55-L91]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference._run_single_frame_inference [L358-L429]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference._cache_frame_outputs [L524-L550]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInferenceWithInstanceInteractivity.add_tracker_new_points [L1400-L1611]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInferenceWithInstanceInteractivity.remove_object [L1280-L1317]`

### Tracker metadata ownership
- `tracker_metadata` is updated in planning phase and consumed by execution/output phases.
- Includes object IDs per GPU, global IDs, object scores, per-frame tracker scores, and rank0 suppression/removal metadata.
- Rank synchronization uses object broadcast (`broadcast_object_list`) and collective ops (`all_gather`, `barrier`).

Evidence:
- `sam3/model/sam3_video_base.py :: run_tracker_update_planning_phase [L506-L819]`
- `sam3/model/sam3_video_base.py :: run_tracker_update_execution_phase [L893-L935]`
- `sam3/model/sam3_video_base.py :: build_outputs [L936-L1015]`
- `sam3/model/sam3_video_base.py :: broadcast_python_obj_cpu [L147-L151]`

### Prompt and target data model
- Training/inference stage payloads are dataclasses (`FindStage`, `BatchedFindTarget`, `BatchedDatapoint`) with conversion helpers.
- Collator packs variable-length per-query inputs into padded tensors and builds `boxes_padded` for matching.

Evidence:
- `sam3/model/data_misc.py :: FindStage [L61-L83]`
- `sam3/model/data_misc.py :: BatchedFindTarget [L85-L123]`
- `sam3/model/data_misc.py :: BatchedDatapoint [L160-L167]`
- `sam3/train/data/collator.py :: collate_fn_api [L137-L361]`

## Math/algorithm notes

### Supported (strong code evidence)
- DETR-style decoder with iterative reference box refinement and query-level class/box heads.
- Presence-logit branch integrated into scoring in image processor/eval postprocessor.
- Hungarian matching over class + L1 + GIoU costs for supervision.
- Tracker memory bank with temporal conditioning and object pointers.
- NMS over mask IoU matrix for detector deduplication.

Evidence:
- `sam3/model/sam3_image.py :: Sam3Image._update_scores_and_boxes [L299-L384]`
- `sam3/model/decoder.py :: TransformerDecoder.forward [L407-L610]`
- `sam3/model/sam3_image_processor.py :: Sam3Processor._forward_grounding [L194-L197]`
- `sam3/eval/postprocessors.py :: PostProcessImage.forward [L101-L105]`
- `sam3/train/matcher.py :: BinaryHungarianMatcherV2.forward [L571-L607]`
- `sam3/model/sam3_tracker_base.py :: Sam3TrackerBase._prepare_memory_conditioned_features [L559-L795]`
- `sam3/perflib/nms.py :: nms_masks [L24-L55]`

### Likely (moderate evidence)
- Detector/tracker decoupling is used to reduce interference and allow richer heuristic scheduling in video.
- Temporal disambiguation is implemented via memory selection/score heuristics in tracker state and planning pipeline.

Evidence:
- `README.md :: README (overview) [L51-L52]`
- `sam3/model_builder.py :: build_sam3_video_model [L719-L771]`
- `sam3/model/sam3_video_base.py :: run_tracker_update_planning_phase [L506-L819]`
- `sam3/model/sam3_tracker_base.py :: Sam3TrackerBase.track_step [L929-L1066]`

### Hypothesis (needs targeted validation)
- Exact calibration impact of detector score gating by presence may vary across tasks and may interact with postprocessor `use_presence` defaults.
- Relative quality/speed tradeoff between perflib backends (CUDA extension vs triton vs CPU fallback) likely depends strongly on object count and frame resolution.

Evidence:
- `sam3/model/sam3_image_processor.py :: Sam3Processor._forward_grounding [L194-L199]`
- `sam3/eval/postprocessors.py :: PostProcessImage.__init__ [L35-L60]`
- `sam3/perflib/nms.py :: generic_nms [L56-L74]`
- `sam3/perflib/connected_components.py :: connected_components [L55-L86]`

## Stability, performance, and pitfalls

### Confirmed risks / pitfalls
- Potential bug: presence-logit clamp appears non-effective because `clamp` result is not assigned/inplace.
  - `intermediate_layer_presence_logits.clamp(...)` is called without assignment.
- `geo_encoder_use_img_cross_attn` exists in `build_sam3_video_model` signature but is not used in body.
- Multi-GPU synchronization can bottleneck on CPU object broadcasts in partial refinement and predictor worker orchestration.
- Heavy compile path with multiple component-level `torch.compile` wrappers can have long warm-up and shape-cache sensitivity.
- Perflib behavior depends on optional external packages (`torch_generic_nms`, `cc_torch`), with fallback paths that can materially change performance.

Evidence:
- `sam3/model/decoder.py :: TransformerDecoder.forward [L583-L588]`
- `sam3/model_builder.py :: build_sam3_video_model [L653-L793]`
- `sam3/model/sam3_video_base.py :: broadcast_python_obj_cpu [L147-L151]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInferenceWithInstanceInteractivity.propagate_in_video [L1116-L1135]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference._compile_model [L574-L653]`
- `sam3/model/sam3_video_inference.py :: Sam3VideoInference.warm_up_compilation [L800-L839]`
- `sam3/perflib/nms.py :: generic_nms [L64-L73]`
- `sam3/perflib/connected_components.py :: connected_components [L74-L86]`

### Practical performance notes
- Detector chunk prefetch + async all-gather is explicitly designed to overlap compute and communication.
- Image/text backbone outputs are cached and reused in video propagation (`feature_cache`, text cache keyed by prompt tuple).
- Tracker supports optional offload of frame/state outputs to CPU for long-video memory pressure control.

Evidence:
- `sam3/model/sam3_image.py :: Sam3ImageOnVideoMultiGPU.forward_video_grounding_multigpu [L762-L785]`
- `sam3/model/sam3_video_base.py :: run_backbone_and_detection [L323-L333]`
- `sam3/model/sam3_video_base.py :: run_backbone_and_detection [L394-L400]`
- `sam3/model/sam3_tracking_predictor.py :: Sam3TrackerPredictor.init_state [L70-L83]`
- `sam3/model/sam3_tracker_base.py :: Sam3TrackerBase.track_step [L1045-L1062]`

## External dependencies and uncertain behavior
- Core runtime: PyTorch; package metadata lists `timm`, `numpy`, `tqdm`, `ftfy`, `regex`, `iopath`, `huggingface_hub`.
- Train extras add `hydra-core`, `submitit`, `tensorboard`, `scipy`, `torchmetrics`, `fvcore`, `fairscale`.
- Eval relies on `pycocotools` API behavior.
- perflib acceleration paths depend on optional third-party extensions (`torch_generic_nms`, `cc_torch`) and triton implementations.

Evidence:
- `pyproject.toml :: project dependencies [L27-L36]`
- `pyproject.toml :: optional train dependencies [L68-L79]`
- `sam3/eval/coco_eval.py :: imports [L24-L35]`
- `sam3/perflib/nms.py :: import fallback [L12-L21]`
- `sam3/perflib/connected_components.py :: import fallback [L8-L17]`

## Agent stack notes
- Agent loop repeatedly calls LLM (`send_generate_request`) and SAM service (`call_sam_service`), tracks used prompts, and prunes message history.
- Output artifacts include json + rendered images; multiple control paths for `segment_phrase`, `examine_each_mask`, `select_masks_and_return`, `report_no_mask`.

Evidence:
- `sam3/agent/agent_core.py :: agent_inference [L124-L565]`
- `sam3/agent/inference.py :: run_single_image_inference [L11-L67]`
- `sam3/agent/client_sam3.py :: call_sam_service [L51-L139]`

## Open questions
- Is the non-assigned clamp in decoder presence logits intentional (relying on downstream clamps), or a bug?
  - `sam3/model/decoder.py :: TransformerDecoder.forward [L583-L588]`
- Should `geo_encoder_use_img_cross_attn` be wired into geometry encoder construction, or removed from public API?
  - `sam3/model_builder.py :: build_sam3_video_model [L653-L793]`
- What is the expected contract between `presence_logit_dec` and postprocessors across all tasks (`use_presence` defaults differ by class)?
  - `sam3/eval/postprocessors.py :: PostProcessImage.__init__ [L35-L60]`
  - `sam3/eval/postprocessors.py :: PostProcessAPIVideo.__init__ [L330-L363]`
- Which perflib backend matrix is expected in production environments, and what are minimum performance baselines for fallback paths?
  - `sam3/perflib/nms.py :: generic_nms [L56-L74]`
  - `sam3/perflib/connected_components.py :: connected_components [L55-L86]`
- For long sessions, what is intended lifecycle/eviction policy for `_ALL_INFERENCE_STATES` at service level beyond explicit close/shutdown?
  - `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictor._ALL_INFERENCE_STATES [L25-L27]`
  - `sam3/model/sam3_video_predictor.py :: Sam3VideoPredictor.close_session [L237-L253]`

## Quick grep anchors (for fast future navigation)
- Image API: `sam3/model/sam3_image_processor.py`, `sam3/model/sam3_image.py`
- Video API: `sam3/model/sam3_video_inference.py`, `sam3/model/sam3_video_base.py`
- Tracker core: `sam3/model/sam3_tracker_base.py`, `sam3/model/sam3_tracking_predictor.py`
- Build graph: `sam3/model_builder.py`
- Train: `sam3/train/train.py`, `sam3/train/trainer.py`, `sam3/train/loss/sam3_loss.py`, `sam3/train/matcher.py`
- Eval: `sam3/eval/postprocessors.py`, `sam3/eval/coco_eval.py`
- Service: `sam3/model/sam3_video_predictor.py`


# Copilot instructions for InternEvo

## Big picture (read this first)
- This repo is a distributed LLM training framework; runtime entry is [../train.py](../train.py), not notebooks or ad-hoc scripts.
- Training boot flow is: CLI args -> distributed init -> model/data builder -> trainer fit:
  - args from [../internlm/utils/common.py](../internlm/utils/common.py)
  - distributed launch/config sanity in [../internlm/initialize/launch.py](../internlm/initialize/launch.py)
  - model creation via [../internlm/model/builder.py](../internlm/model/builder.py)
  - trainer orchestration via [../internlm/core/trainer_builder.py](../internlm/core/trainer_builder.py)
  - core train pipeline logic in [../internlm/train/pipeline.py](../internlm/train/pipeline.py)
- Keep changes aligned with the architecture split in [../doc/structure.md](../doc/structure.md): `core` (parallel context/scheduler), `data`, `model`, `solver`, `initialize`, `utils`.

## Project-specific configuration conventions
- Config files are executable Python (example: [../configs/7B_sft.py](../configs/7B_sft.py)), not YAML.
- Pattern: define uppercase constants first (`SEQ_LEN`, `NUM_LAYER`, etc.), then compose dicts (`data`, `model`, `parallel`, `ckpt`).
- `parallel.pipeline.mode` is validated/uppercased in launch and must be one of `1F1B`, `ZBH1`, `ZBV`, `UNIFIED`, `HYDRA`.
- `data.packed_length` is derived as `seq_len * micro_bsz`; do not hardcode contradictory values.
- Checkpoint paths use scheme prefixes (e.g., `local:...`, `boto3:s3://...`), and `auto_resume=True` overrides naive expectations of `load_ckpt_info`.

## Data and integration points
- Two primary dataset flows:
  - tokenized bin/meta (use [../tools/tokenizer.py](../tools/tokenizer.py), format documented in [../tools/README.md](../tools/README.md))
  - HuggingFace streaming (`data.type="streaming"` + `tokenizer_path`), documented in [../README.md](../README.md) and [../doc/usage.md](../doc/usage.md)
- Optional performance dependencies are real integration boundaries:
  - flash-attn (`v2.2.1`), Apex, megablocks, plus local C++ extensions in [../csrc/rotary](../csrc/rotary) and [../csrc/xentropy](../csrc/xentropy).

## Developer workflows that match this repo
- Typical local multi-GPU run:
  - `torchrun --nnodes=1 --nproc_per_node=8 train.py --config ./configs/7B_sft.py --launcher torch`
- Slurm run is first-class and default parser launcher is `slurm`; if using torchrun, pass `--launcher torch` explicitly.
- Before running tests/scripts locally, set `PYTHONPATH=$PWD:$PYTHONPATH` (mirrors CI).
- CI uses `pytest` heavily with GPU markers (see [../tests/test_training/test_loss.py](../tests/test_training/test_loss.py)); many tests assume 4/8/16 GPUs and/or Slurm.
- Lint contract (from workflow): flake8 + isort + black + pylint with max line length 120; preserve existing disable lists in [workflows/lint_check.yaml](workflows/lint_check.yaml).

## Guardrails for code changes
- Prefer editing inside `internlm/*` modules over changing top-level scripts unless entry behavior is intentionally modified.
- When changing parallelism behavior, verify interactions among `zero1`, `tensor`, `pipeline`, `weight`, and `expert` settings in [../internlm/initialize/launch.py](../internlm/initialize/launch.py).
- Keep `PipelineSimulator/` changes isolated; it is a separate simulation toolchain and not the training runtime path in [../train.py](../train.py).
- Reuse existing model/optimizer/scheduler factories and registries rather than introducing one-off construction paths.

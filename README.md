# knowledge_inv

Study how engineering background knowledge, including function–behavior–structure (FBS) information, changes a vision-language model’s CAD-editing decisions.

This repository is the first milestone: load a verified request–image pair from neuralCAD-Edit, run Qwen3-VL-8B-Instruct on GPU, capture selected prefill activations, and compare three controlled prompting conditions.

## Environment

Work in WSL, not Windows Python.

```bash
cd /home/adml/Research/knowledge_inv
source .venv/bin/activate
export PYTHONPATH=src
```

The project virtualenv currently uses **Python 3.14.6** from Miniconda and a working **PyTorch 2.11.0+cu128** build. A shell may also show `(base)`; always check:

```bash
which python
python -c "import sys; print(sys.executable); print(sys.version)"
```

Use `.venv/bin/python` for all project commands. Do not install the official neuralCAD-Edit `environment.yml` on top of this environment. That stack includes CadQuery/vLLM and is for later CAD execution in a separate env.

Pinned packages are in `requirements.txt`. If you need to recreate the env later:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt
```

CUDA 12.8 was verified on this machine (RTX 5090, driver 610, 31.84 GB VRAM, BF16 supported). Recheck `nvidia-smi` and `scripts/verify_environment.py` before changing the PyTorch build.

## Dataset

Downloaded and extracted under `data/edit_192_external`. This is the full 192-request / 384-expert-edit release, not the 48-case hackathon slice. The 48 text-modality requests are one of four modality groups inside that full set.

Official Autodesk code is a git submodule at `external/neuralCAD-Edit` (pinned to `68d1acceda0743e364f45fb7823f7cf39c530ba5`). That is the [`src/`](https://github.com/AutodeskAILab/neuralCAD-Edit/tree/main/src) tree (harnesses, VLMs, preprocess, configs). After `git clone`, run:

```bash
git submodule update --init --recursive
```

Official `storage_dir` is the dataset root that contains `mongita_db/`. The dataset itself is not in this repo.

Do not run official cleanup, ingestion, or benchmark scripts on the source copy.

## Commands

```bash
cd /home/adml/Research/knowledge_inv
source .venv/bin/activate
export PYTHONPATH=src

python scripts/verify_environment.py
python scripts/inspect_dataset.py
python scripts/build_manifest.py --limit 8
python scripts/run_qwen_example.py
python scripts/run_pilot.py --conditions direct plan fbs
```

`run_qwen_example.py` uses the first verified manifest record, a modest-resolution original isometric view, and does not receive edited images, expert solutions, or ratings.

`run_pilot.py` repeats the same image and request under:

1. Direct decision
2. Generic plan, then decision
3. Generated FBS analysis, then decision

Analysis generation and decision are logged as separate stages. Generated analysis is labeled model-inferred, not supplied expert fact.

## Outputs

- `reports/environment.md`
- `reports/dataset_inspection.md`
- `manifests/pilot_text_v1.jsonl`
- `runs/qwen_example/<request_id>/`
- `runs/pilot/<request_id>/{direct,plan,fbs}/`

Activation tensors are stored separately from metadata (`activations.pt` + `activations.json`). Large assets and run files stay out of Git.

## First milestone status

Completed on this machine:

- Environment and CUDA matmul verified (`reports/environment.md`)
- Dataset inspected from Mongita without mutating it (`reports/dataset_inspection.md`)
- Pilot manifest of 8 verified text requests (`manifests/pilot_text_v1.jsonl`)
- Qwen3-VL-8B-Instruct GPU run on `ZK22J6VYRKQ2RTFD_1758875163.609787`
- Prefill activations saved for language layers 8, 17, and 35
- Three-condition pilot logged under `runs/pilot/`

`test_qwen_v1.py` is a leftover smoke script. It still points at the nonexistent path `data/images/part.png`. Use `scripts/run_qwen_example.py` instead.

## License

The dataset is CC BY-NC 4.0. Keep the license files in `data/edit_192_external`.

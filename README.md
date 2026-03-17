<!--
Tip: Keep this README focused on setup + usage. Put longer notes in `references/`.
-->

# Multimodal Emotion Recognition (Audio + Video)

Train a multimodal emotion classifier on **RAVDESS-style** samples by fusing:

- **Audio** embeddings from **HuBERT** (`facebook/hubert-base-ls960`)
- **Video** embeddings from **ViViT** (`google/vivit-b-16x2-kinetics400`)
- A **bi-directional cross-attention** fusion layer (`src/models/fusion_model.py`)

## Repository Layout

- `src/data/load_dataset.py`: dataset + augmentations + `collate_fn` (pads audio and builds an attention mask)
- `src/models/audio_model.py`: HuBERT feature extractor
- `src/models/video_model.py`: ViViT feature extractor
- `src/models/fusion_model.py`: cross-attention fusion classifier
- `environment.yml` / `requirements.txt`: environment snapshots

## Prerequisites

- Linux/macOS recommended (Windows should work via WSL)
- Python 3.10+ (this repo’s exported environment uses Python 3.11)
- (Optional) NVIDIA GPU + CUDA if you want GPU training

Notes:
- First run will download Hugging Face pretrained weights (HuBERT/ViViT) into your local cache (`~/.cache/huggingface/`).
- This project uses `wandb` for experiment tracking; you can disable it via `WANDB_MODE=disabled`.

## Installation

### Option A (Recommended): Conda environment

Create the environment from the exported YAML:

```bash
conda env create -f environment.yml
conda activate iopenv
```

If you edit dependencies and want to re-export:

```bash
conda env export --no-builds > environment.yml
```

### Option B: venv + pip (best-effort)

The current `requirements.txt` may include machine-specific entries (e.g., local `file://...` wheels from an export),
so treat this route as “best-effort”:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `torch/torchaudio/torchvision` fail to install via `requirements.txt`, install PyTorch first using the official
PyTorch selector for your OS/CUDA, then re-run the command above for the remaining deps.

## Data Format

The dataloader expects **paired `.npy` files** in two folders with the **same filename**:

```
data/processed/audio/<sample_id>.npy
data/processed/video/<sample_id>.npy
```

### Filenames (RAVDESS convention)

Filenames are parsed like:

`01-01-03-01-02-02-19.npy`

Where the 3rd field (`03`) is the **emotion** label and the last field (`19`) is the **actor id**.
Labels are converted from `1..8` to `0..7` in `src/data/load_dataset.py`.

### Train/Val/Test split

The dataset splits by actor id in `src/data/load_dataset.py`:

- `actor_id <= 21` → train
- `22..23` → val
- `24` → test

### Tensor shapes

- Audio: 1D waveform stored as `.npy` (variable length allowed; it is padded per-batch in `collate_fn`)
- Video: `.npy` with frames shaped like `(T, H, W, C)`; it is transposed to `(T, C, H, W)` and resized to `224x224`

## Usage

### Dataset sanity check

Runs a quick batch-load to validate shapes and padding logic:

```bash
python src/data/load_dataset.py
```

### Training

If you have a training entrypoint (for example `train.py`), you can run:

```bash
WANDB_MODE=disabled python train.py --epochs 30 --batch_size 4 --lr 1e-4
```

By default, the dataloader looks for:

- `--audio_dir ./data/processed/audio`
- `--vid_dir ./data/processed/video`

## Configuration & Secrets

- Avoid committing secrets (API keys, tokens).
- If you use Weights & Biases, prefer exporting `WANDB_API_KEY` in your shell environment instead of committing it.

## Contributing

Contributions are welcome—bug fixes, cleanup, training/eval scripts, docs, and experiments.

- Create a branch from `main`
- Keep PRs small and focused
- Add a short description of the change + how you validated it
- Do not commit large datasets or generated artifacts (models, caches, logs)

### Development checklist

- `python -m compileall .` (quick syntax check)
- Run the dataset sanity check: `python src/data/load_dataset.py`

## Legal / “Contributing Rules”

- Only contribute code/data that you have the legal right to share.
- If you add datasets or pretrained weights references, include the source and the applicable license/terms.
- Don’t upload sensitive personal data or private recordings.

## License

See `LICENCE.txt` (MIT).

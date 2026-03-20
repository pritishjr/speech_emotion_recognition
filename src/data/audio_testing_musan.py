"""
Create noisy audio test data by mixing MUSAN noise with processed clean waveforms.

Example:
python src/data/audio_testing_musan.py \
    --clean_audio_dir ./data/processed/audio \
    --noise_dir ./data/raw/MUSAN/noise/sound-bible \
    --output_dir ./data/processed/audio-test \
    --split test
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

try:
    import librosa
except ImportError:
    librosa = None


def parse_args():
    parser = argparse.ArgumentParser("Build noisy test data from MUSAN noise")

    parser.add_argument("--clean_audio_dir", type=str, default="./data/processed/audio", help="Directory with clean .npy waveforms.")
    parser.add_argument("--noise_dir", type=str, default="./data/raw/MUSAN/noise/sound-bible", help="Directory with MUSAN noise .wav files.")
    parser.add_argument("--output_dir", type=str, default="./data/processed/audio-test", help="Directory to save noisy .npy waveforms.")
    parser.add_argument("--metadata_csv", type=str, default="./data/processed/audio-test/metadata.csv", help="CSV file that logs clean/noise/SNR mapping.")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"], help="Which actor-based split from clean_audio_dir to convert.")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate when loading MUSAN noise.")
    parser.add_argument("--snr_db", type=float, nargs="+", default=[0.0, 5.0, 10.0, 15.0], help="Candidate SNR levels in dB. One is sampled per file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap for number of files to process.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing files in output_dir.")

    return parser.parse_args()


def actor_id_from_filename(filename):
    parts = filename.split("-")
    if len(parts) != 7:
        return None
    try:
        return int(parts[6].split(".")[0])
    except ValueError:
        return None


def keep_for_split(filename, split):
    if split == "all":
        return True

    actor_id = actor_id_from_filename(filename)
    if actor_id is None:
        return False

    if split == "train":
        return actor_id <= 20
    if split == "val":
        return 21 <= actor_id <= 22
    if split == "test":
        return actor_id > 22
    return False


def match_length(noise, target_len, rng):
    if len(noise) == 0:
        raise ValueError("Noise sample is empty after loading.")

    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)

    max_start = len(noise) - target_len
    start_idx = rng.randint(0, max_start) if max_start > 0 else 0
    return noise[start_idx : start_idx + target_len]


def mix_with_snr(clean, noise, snr_db):
    clean = clean.astype(np.float32)
    noise = noise.astype(np.float32)

    clean_rms = np.sqrt(np.mean(clean**2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise**2) + 1e-12)

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    noise_scale = desired_noise_rms / max(noise_rms, 1e-12)

    mixed = clean + (noise * noise_scale)

    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def collect_clean_files(clean_audio_dir, split):
    clean_dir = Path(clean_audio_dir)
    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Clean audio directory not found: {clean_audio_dir}")

    files = sorted(clean_dir.glob("*.npy"))
    return [f for f in files if keep_for_split(f.name, split)]


def collect_noise_files(noise_dir):
    noise_root = Path(noise_dir)
    if not noise_root.is_dir():
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

    noise_files = sorted(noise_root.rglob("*.wav"))
    if not noise_files:
        raise ValueError(f"No .wav files found under: {noise_dir}")
    return noise_files


def load_noise_waveform(noise_path, target_sr):
    if librosa is not None:
        noise_wave, _ = librosa.load(str(noise_path), sr=target_sr, mono=True)
        return noise_wave.astype(np.float32)

    src_sr, noise_wave = wavfile.read(str(noise_path))

    if noise_wave.ndim > 1:
        noise_wave = noise_wave.mean(axis=1)

    if np.issubdtype(noise_wave.dtype, np.integer):
        info = np.iinfo(noise_wave.dtype)
        scale = max(abs(info.min), abs(info.max))
        noise_wave = noise_wave.astype(np.float32) / float(scale)
    else:
        noise_wave = noise_wave.astype(np.float32)

    if src_sr != target_sr:
        noise_wave = signal.resample_poly(noise_wave, target_sr, src_sr).astype(np.float32)

    return noise_wave


def build_noisy_dataset(
    clean_audio_dir,
    noise_dir,
    output_dir,
    metadata_csv=None,
    split="test",
    target_sr=16000,
    snr_db_values=None,
    seed=42,
    max_files=None,
    overwrite=False,
    quiet=False,
):
    if snr_db_values is None or len(snr_db_values) == 0:
        snr_db_values = [0.0, 5.0, 10.0, 15.0]

    rng = random.Random(seed)

    clean_files = collect_clean_files(clean_audio_dir, split)
    noise_files = collect_noise_files(noise_dir)

    if max_files is not None:
        clean_files = clean_files[:max_files]

    if not clean_files:
        raise ValueError(f"No clean .npy files selected for split='{split}' in {clean_audio_dir}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if metadata_csv is None:
        metadata_csv = output_dir / "metadata.csv"
    metadata_path = Path(metadata_csv)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    noise_cache = {}
    metadata_rows = []
    written_count = 0
    skipped_count = 0

    if not quiet:
        print(f"Selected clean files: {len(clean_files)}")
        print(f"Available MUSAN noise files: {len(noise_files)}")
        print(f"Writing noisy outputs to: {output_dir}")

    for clean_path in tqdm(clean_files, desc="Mixing MUSAN noise", disable=quiet):
        out_path = output_dir / clean_path.name
        if out_path.exists() and not overwrite:
            skipped_count += 1
            continue

        clean_wave = np.load(clean_path).astype(np.float32).flatten()
        if len(clean_wave) == 0:
            skipped_count += 1
            continue

        noise_path = rng.choice(noise_files)
        if noise_path not in noise_cache:
            noise_cache[noise_path] = load_noise_waveform(noise_path, target_sr)
        noise_wave = noise_cache[noise_path]

        noise_segment = match_length(noise_wave, len(clean_wave), rng)
        snr_db = float(rng.choice(snr_db_values))
        noisy_wave = mix_with_snr(clean_wave, noise_segment, snr_db)

        np.save(out_path, noisy_wave)
        written_count += 1

        metadata_rows.append(
            {
                "clean_file": str(clean_path),
                "noise_file": str(noise_path),
                "snr_db": snr_db,
                "output_file": str(out_path),
                "num_samples": int(len(noisy_wave)),
            }
        )

    with metadata_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["clean_file", "noise_file", "snr_db", "output_file", "num_samples"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    summary = {
        "split": split,
        "clean_files_selected": len(clean_files),
        "written_files": written_count,
        "skipped_files": skipped_count,
        "noise_files_available": len(noise_files),
        "output_dir": str(output_dir),
        "metadata_csv": str(metadata_path),
        "snr_db_values": [float(x) for x in snr_db_values],
    }

    if not quiet:
        print("Noisy test set creation complete.")
        print(f"Written files: {written_count}")
        print(f"Skipped files: {skipped_count}")
        print(f"Metadata CSV: {metadata_path}")

    return summary


def main():
    args = parse_args()
    build_noisy_dataset(
        clean_audio_dir=args.clean_audio_dir,
        noise_dir=args.noise_dir,
        output_dir=args.output_dir,
        metadata_csv=args.metadata_csv,
        split=args.split,
        target_sr=args.target_sr,
        snr_db_values=args.snr_db,
        seed=args.seed,
        max_files=args.max_files,
        overwrite=args.overwrite,
        quiet=False,
    )


if __name__ == "__main__":
    main()

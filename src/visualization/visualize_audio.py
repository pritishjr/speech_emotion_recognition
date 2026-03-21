import argparse
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
DATA_ROOT = SRC_ROOT / "data"
VISUALIZATION_ROOT = SRC_ROOT / "visualization"

for module_root in (DATA_ROOT, VISUALIZATION_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from src.data.audio_testing_musan import collect_clean_files
from plot_settings import global_plot_settings


DEFAULT_CLEAN_AUDIO_DIR = REPO_ROOT / "data/processed/audio"
DEFAULT_NOISY_AUDIO_DIR = REPO_ROOT / "data/processed/audio-test"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "reports/figures/random_test_audio_snr_comparison.png"
DEFAULT_WAV_OUTPUT_DIR = REPO_ROOT / "reports/audio/random_test_audio_snr_comparison"
DEFAULT_SAMPLE_RATE = 16000
SNR_DIR_PATTERN = re.compile(r"^snr_(?P<value>-?\d+(?:\.\d+)?)db$")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select one clean test waveform and its matching noisy copies across SNR folders."
    )
    parser.add_argument(
        "--clean_audio_dir",
        type=str,
        default=str(DEFAULT_CLEAN_AUDIO_DIR),
        help="Directory containing clean processed .npy audio files.",
    )
    parser.add_argument(
        "--noisy_audio_dir",
        type=str,
        default=str(DEFAULT_NOISY_AUDIO_DIR),
        help="Root directory that contains noisy SNR folders such as snr_0db.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "all"],
        help="Dataset split to sample the clean file from.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional clean filename to use instead of random selection.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Where to save the waveform comparison figure.",
    )
    parser.add_argument(
        "--wav_output_dir",
        type=str,
        default=str(DEFAULT_WAV_OUTPUT_DIR),
        help="Directory where the selected clean and noisy .wav files will be saved.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate to use when exporting the selected waveforms to .wav.",
    )
    parser.add_argument(
        "--save_wav",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the selected clean and noisy waveforms as .wav files.",
    )
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display the figure after saving it.",
    )
    return parser.parse_args()


def collect_snr_directories(noisy_audio_dir):
    noisy_root = Path(noisy_audio_dir)
    if not noisy_root.is_dir():
        raise FileNotFoundError(f"Noisy audio directory not found: {noisy_audio_dir}")

    snr_dirs = []
    for path in sorted(noisy_root.iterdir()):
        match = SNR_DIR_PATTERN.match(path.name)
        if path.is_dir() and match:
            snr_dirs.append((float(match.group("value")), path))

    if not snr_dirs:
        raise ValueError(f"No SNR directories found under: {noisy_audio_dir}")

    return snr_dirs


def noisy_variants_for_clean(clean_path, snr_directories):
    matches = {}
    for snr_value, snr_dir in snr_directories:
        noisy_path = snr_dir / clean_path.name
        if noisy_path.is_file():
            matches[snr_value] = noisy_path
    return matches


def choose_clean_file(clean_audio_dir, noisy_audio_dir, split="test", seed=None, filename=None):
    clean_files = collect_clean_files(clean_audio_dir, split)
    if not clean_files:
        raise ValueError(f"No clean files found for split='{split}' in {clean_audio_dir}")

    snr_directories = collect_snr_directories(noisy_audio_dir)

    if filename is not None:
        clean_lookup = {path.name: path for path in clean_files}
        if filename not in clean_lookup:
            raise FileNotFoundError(
                f"Requested filename '{filename}' is not available in split='{split}'."
            )
        clean_path = clean_lookup[filename]
        noisy_paths = noisy_variants_for_clean(clean_path, snr_directories)
        if len(noisy_paths) != len(snr_directories):
            raise FileNotFoundError(
                f"Missing one or more noisy SNR variants for clean file '{filename}'."
            )
        return clean_path, noisy_paths

    eligible_clean_files = []
    for clean_path in clean_files:
        noisy_paths = noisy_variants_for_clean(clean_path, snr_directories)
        if len(noisy_paths) == len(snr_directories):
            eligible_clean_files.append((clean_path, noisy_paths))

    if not eligible_clean_files:
        raise ValueError(
            "No clean files have matching noisy variants across every SNR directory."
        )

    rng = random.Random(seed)
    return rng.choice(eligible_clean_files)


def load_waveforms(clean_path, noisy_paths):
    loaded_noisy = {
        snr_value: np.load(path).astype(np.float32) for snr_value, path in noisy_paths.items()
    }
    return np.load(clean_path).astype(np.float32), loaded_noisy


def extract_random_test_audio(
    clean_audio_dir=DEFAULT_CLEAN_AUDIO_DIR,
    noisy_audio_dir=DEFAULT_NOISY_AUDIO_DIR,
    split="test",
    seed=None,
    filename=None,
):
    clean_path, noisy_paths = choose_clean_file(
        clean_audio_dir=clean_audio_dir,
        noisy_audio_dir=noisy_audio_dir,
        split=split,
        seed=seed,
        filename=filename,
    )
    clean_waveform, noisy_waveforms = load_waveforms(clean_path, noisy_paths)
    return {
        "clean_file": clean_path,
        "clean_waveform": clean_waveform,
        "noisy_files": noisy_paths,
        "noisy_waveforms": noisy_waveforms,
    }


def plot_audio_comparison(sample_bundle, save_path=None, show=False):
    global_plot_settings()

    noisy_items = sorted(sample_bundle["noisy_waveforms"].items())
    total_plots = 1 + len(noisy_items)

    fig, axes = plt.subplots(total_plots, 1, figsize=(12, 2.8 * total_plots), sharex=True)
    if total_plots == 1:
        axes = [axes]

    clean_waveform = sample_bundle["clean_waveform"]
    clean_name = sample_bundle["clean_file"].name

    axes[0].plot(clean_waveform, color="black", linewidth=1.0)
    axes[0].set_title(f"Clean test audio: {clean_name}")
    axes[0].set_ylabel("Amplitude")

    for axis, (snr_value, waveform) in zip(axes[1:], noisy_items):
        axis.plot(waveform, linewidth=1.0)
        axis.set_title(f"Noisy version at {snr_value:g} dB")
        axis.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Sample Index")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)


def waveform_to_int16(waveform):
    waveform = np.asarray(waveform, dtype=np.float32).flatten()
    waveform = np.clip(waveform, -1.0, 1.0)
    return (waveform * 32767.0).astype(np.int16)


def save_waveform_as_wav(waveform, output_path, sample_rate=DEFAULT_SAMPLE_RATE):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(output_path, sample_rate, waveform_to_int16(waveform))
    return output_path


def export_sample_bundle_as_wav(sample_bundle, output_dir=DEFAULT_WAV_OUTPUT_DIR, sample_rate=DEFAULT_SAMPLE_RATE):
    output_dir = Path(output_dir)
    stem = sample_bundle["clean_file"].stem

    exported_files = {}
    exported_files["clean"] = save_waveform_as_wav(
        sample_bundle["clean_waveform"],
        output_dir / f"{stem}_clean.wav",
        sample_rate=sample_rate,
    )

    noisy_exports = {}
    for snr_value, waveform in sorted(sample_bundle["noisy_waveforms"].items()):
        snr_label = f"{snr_value:g}".replace("-", "neg").replace(".", "p")
        noisy_exports[snr_value] = save_waveform_as_wav(
            waveform,
            output_dir / f"{stem}_snr_{snr_label}db.wav",
            sample_rate=sample_rate,
        )

    exported_files["noisy"] = noisy_exports
    return exported_files


def print_sample_summary(sample_bundle):
    print(f"Clean file: {sample_bundle['clean_file']}")
    for snr_value, noisy_path in sorted(sample_bundle["noisy_files"].items()):
        print(f"{snr_value:g} dB noisy file: {noisy_path}")


def print_wav_summary(exported_files):
    print(f"Saved clean wav: {exported_files['clean']}")
    for snr_value, wav_path in sorted(exported_files["noisy"].items()):
        print(f"Saved {snr_value:g} dB wav: {wav_path}")


def main():
    args = parse_args()
    sample_bundle = extract_random_test_audio(
        clean_audio_dir=args.clean_audio_dir,
        noisy_audio_dir=args.noisy_audio_dir,
        split=args.split,
        seed=args.seed,
        filename=args.filename,
    )
    print_sample_summary(sample_bundle)
    plot_audio_comparison(sample_bundle, save_path=args.save_path, show=args.show)
    if args.save_path:
        print(f"Saved comparison figure to: {Path(args.save_path)}")
    if args.save_wav:
        exported_files = export_sample_bundle_as_wav(
            sample_bundle,
            output_dir=args.wav_output_dir,
            sample_rate=args.sample_rate,
        )
        print_wav_summary(exported_files)


if __name__ == "__main__":
    main()

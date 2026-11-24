# -*- coding: utf-8 -*-
"""Interactive microphone voice enrollment script for spoken digits 0-9."""
import argparse
import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
from utils import trim_silence


SAMPLE_RATE = 22050


def record_clip(duration=1.5, sr=SAMPLE_RATE):
    """Record a mono audio clip from the default microphone."""
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.squeeze(axis=1)


def next_sample_index(digit_dir):
    """Return next 1-based sample index in digit_dir so existing recordings are preserved."""
    if not os.path.isdir(digit_dir):
        return 1
    existing = [f for f in os.listdir(digit_dir) if f.lower().endswith('.wav')]
    return len(existing) + 1


def enroll_digit(digit, count, duration, raw_root, train_root, test_root):
    """Record `count` valid samples for a single digit and split 80/20 into train/test."""
    raw_dir = os.path.join(raw_root, f"digit_{digit}")
    train_dir = os.path.join(train_root, f"digit_{digit}")
    test_dir = os.path.join(test_root, f"digit_{digit}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    start_idx = next_sample_index(raw_dir)
    test_count = max(1, count // 5) if count >= 5 else 0
    recorded = 0

    print(f"\n=== Digit {digit} ({count} samples) ===")
    while recorded < count:
        idx = start_idx + recorded
        cmd = input(f"[{digit}] Sample {recorded + 1}/{count} — Press Enter to record (or 's' to skip, 'q' to quit): ").strip().lower()
        if cmd == 'q':
            return False
        if cmd == 's':
            break

        print("  Speak now...", end='', flush=True)
        time.sleep(0.15)
        y = record_clip(duration=duration, sr=SAMPLE_RATE)
        trimmed, is_valid = trim_silence(y, SAMPLE_RATE)
        peak = float(np.max(np.abs(y))) if y.size else 0.0

        if not is_valid:
            print(f" too quiet or too short (peak={peak:.4f}, trimmed={len(trimmed)/SAMPLE_RATE:.2f}s). Try again.")
            continue

        print(f" captured {len(trimmed)/SAMPLE_RATE:.2f}s (peak={peak:.3f}).")
        raw_path = os.path.join(raw_dir, f"{idx}_{digit}.wav")
        sf.write(raw_path, y, SAMPLE_RATE)

        is_test = recorded >= (count - test_count)
        target_dir = test_dir if is_test else train_dir
        proc_path = os.path.join(target_dir, f"{idx}_{digit}.wav")
        sf.write(proc_path, trimmed, SAMPLE_RATE)
        recorded += 1

    return True


def main():
    parser = argparse.ArgumentParser(description="Record your own spoken digits (0-9) for voice enrollment.")
    parser.add_argument("--digits", type=int, nargs="+", default=list(range(10)), help="Digits to record (default: 0-9)")
    parser.add_argument("--count", type=int, default=10, help="Samples per digit (default: 10)")
    parser.add_argument("--duration", type=float, default=1.5, help="Seconds per recording (default: 1.5)")
    parser.add_argument("--raw-dir", default="./my_records", help="Directory for raw recordings")
    parser.add_argument("--train-dir", default="./my_train_records", help="Directory for trimmed training clips")
    parser.add_argument("--test-dir", default="./my_test_records", help="Directory for trimmed test clips")
    args = parser.parse_args()

    print("Microphone Voice Enrollment")
    print(f"Sample rate: {SAMPLE_RATE} Hz | Clip duration: {args.duration}s | Samples/digit: {args.count}")
    print("Tip: Speak clearly and naturally, leaving a tiny pause before/after the digit.")

    for d in args.digits:
        if d < 0 or d > 9:
            print(f"Skipping invalid digit {d} (must be 0-9).")
            continue
        ok = enroll_digit(d, args.count, args.duration, args.raw_dir, args.train_dir, args.test_dir)
        if not ok:
            print("\nStopped early.")
            break

    print("\nDone! Run `.venv\\Scripts\\python.exe train_all.py` to train models on your voice.")


if __name__ == "__main__":
    main()

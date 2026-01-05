#!/usr/bin/env python3
"""
train_composer_models.py

Train per-composer LSTM models from folders of MIDIs.

Usage:
    python train_composer_models.py --dataset ./dataset --out_models ./models --steps_per_quarter 4 --seq_len 64 --epochs 40

Requirements:
    pip install tensorflow pretty_midi numpy tqdm
Notes:
    - Dataset folder should contain subfolders named by composer.
    - Use public-domain MIDI files for training (e.g. Bach, Beethoven).
"""

import os
import glob
import json
import argparse
from collections import Counter
from tqdm import tqdm
import numpy as np
import pretty_midi
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------
# Parameters (defaults)
# ----------------------
DEFAULT_STEPS_PER_QUARTER = 4
DEFAULT_SEQ_LEN = 64
DEFAULT_MIN_TOKEN_FREQ = 2
DEFAULT_EMBED = 256
DEFAULT_LSTM_UNITS = 512
DEFAULT_BATCH = 64

# ----------------------
# Utilities
# ----------------------
def midi_to_time_steps(pm: pretty_midi.PrettyMIDI, steps_per_quarter=4, min_pitch=21, max_pitch=108):
    # derive step_seconds from median beat spacing
    beats = pm.get_beats()
    if len(beats) > 1:
        median_beat = np.median(np.diff(beats))
    else:
        # fallback tempo ~120 bpm -> beat 0.5s
        median_beat = 0.5
    step_seconds = median_beat / steps_per_quarter
    total_steps = int(np.ceil(pm.get_end_time() / step_seconds)) + 1
    time_steps = [set() for _ in range(total_steps)]
    for inst in pm.instruments:
        for n in inst.notes:
            if n.pitch < min_pitch or n.pitch > max_pitch:
                continue
            start = int(np.floor(n.start / step_seconds))
            end = int(np.ceil(n.end / step_seconds))
            for t in range(start, end):
                if 0 <= t < total_steps:
                    time_steps[t].add(n.pitch)
    print("MIDI to Time-Steps COMPLETE")
    return time_steps

def chord_to_token(chord_set):
    if not chord_set:
        return "rest"
    return ",".join(map(str, sorted(chord_set)))

# ----------------------
# Build vocab & dataset
# ----------------------
def build_vocab_and_sequences(midi_files, steps_per_quarter, min_token_freq, seq_len):
    token_counts = Counter()
    all_time_sequences = []  # list of token lists per file
    print("Parsing MIDI files and counting tokens...")
    for f in tqdm(midi_files):
        try:
            pm = pretty_midi.PrettyMIDI(f)
            tsteps = midi_to_time_steps(pm, steps_per_quarter=steps_per_quarter)
            tokens = [chord_to_token(s) for s in tsteps]
            for tok in tokens:
                token_counts[tok] += 1
            all_time_sequences.append(tokens)
        except Exception as e:
            print(f"Warning: cannot parse {f}: {e}")
    # filter tokens
    common_tokens = [t for t,c in token_counts.items() if c >= min_token_freq]
    common_tokens = sorted(common_tokens)
    token_to_id = {t: i+1 for i,t in enumerate(common_tokens)}  # reserve 0 for PAD/UNK
    token_to_id['<UNK>'] = 0
    id_to_token = {i: t for t,i in token_to_id.items()}
    # create X,Y arrays
    X = []
    Y = []
    for tokens in all_time_sequences:
        ids = [token_to_id.get(t, 0) for t in tokens]
        if len(ids) <= seq_len:
            continue
        for i in range(0, len(ids) - seq_len):
            X.append(ids[i:i+seq_len])
            Y.append(ids[i+seq_len])
    if len(X) == 0:
        return None, None, token_to_id, id_to_token
    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)
    print("Build vocab and sequences COMPLETE")
    return X, Y, token_to_id, id_to_token

# ----------------------
# Model builder
# ----------------------
def build_lstm_model(vocab_size, seq_len, embed_dim=256, lstm_units=512):
    model = models.Sequential([
        layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=seq_len, mask_zero=True),
        layers.LSTM(lstm_units, return_sequences=True),
        layers.LSTM(lstm_units//2),
        layers.Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ----------------------
# Main
# ----------------------
def train_composer(midi_dir, out_models_dir, composer_name, args):
    midi_files = glob.glob(os.path.join(midi_dir, "*.mid")) + glob.glob(os.path.join(midi_dir, "*.midi"))
    if not midi_files:
        print(f"No MIDIs found for {composer_name} in {midi_dir}")
        return
    X, Y, token_to_id, id_to_token = build_vocab_and_sequences(
        midi_files,
        steps_per_quarter=args.steps_per_quarter,
        min_token_freq=args.min_token_freq,
        seq_len=args.seq_len
    )
    if X is None:
        print(f"Not enough sequences for {composer_name}, skipping.")
        return
    vocab_size = len(token_to_id)
    print(f"Training {composer_name}: {len(X)} sequences, vocab size {vocab_size}")
    model = build_lstm_model(vocab_size=vocab_size, seq_len=args.seq_len,
                             embed_dim=args.embed_dim, lstm_units=args.lstm_units)
    # callbacks: save best
    composer_out = os.path.join(out_models_dir, composer_name)
    os.makedirs(composer_out, exist_ok=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(composer_out, "best_model.h5"),
                                                    save_best_only=True, monitor="loss", verbose=1)
    model.fit(X, Y, batch_size=args.batch, epochs=args.epochs, callbacks=[checkpoint], validation_split=0.05)
    # save final model and vocab
    model.save(os.path.join(composer_out, "final_model.h5"))
    meta = {
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "steps_per_quarter": args.steps_per_quarter,
        "seq_len": args.seq_len
    }
    with open(os.path.join(composer_out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved model and vocab to {composer_out}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Root dataset folder with subfolders per composer")
    parser.add_argument("--out_models", default="./models", help="Output models folder")
    parser.add_argument("--steps_per_quarter", type=int, default=DEFAULT_STEPS_PER_QUARTER)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--min_token_freq", type=int, default=DEFAULT_MIN_TOKEN_FREQ)
    parser.add_argument("--embed_dim", type=int, default=DEFAULT_EMBED)
    parser.add_argument("--lstm_units", type=int, default=DEFAULT_LSTM_UNITS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    # iterate composers
    for composer_folder in sorted(os.listdir(args.dataset)):
        composer_path = os.path.join(args.dataset, composer_folder)
        if not os.path.isdir(composer_path):
            continue
        print("Processing composer:", composer_folder)
        train_composer(composer_path, args.out_models, composer_folder, args)

if __name__ == "__main__":
    main()

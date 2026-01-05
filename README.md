# Realtime_AI-Composer-Models_Playback
AI model training with MIDI-datasets of classical composers in Python --> Realtime playback of MIDI-notes with trained models in python --> Realtime playback of said MIDI-notes with the VST-plugin "LABS" from Steinberg in PureData

## Overview
**NeuralMIDI-Streamer** is a real-time generative music system that bridges the gap between Deep Learning and live audio synthesis. It uses **LSTM (Long Short-Term Memory)** neural networks trained on classical composer datasets to generate polyphonic MIDI streams on the fly.

Unlike static MIDI generation, this system acts as a **virtual instrument**: it streams notes via **OSC (Open Sound Control)** to a dedicated audio engine (PureData hosting VST plugins). The system features a custom state-machine to handle note polyphony, velocity smoothing, and allows the user to modulate generation parameters (Temperature, Density, Intensity) live during performance.

This project serves as a research prototype for exploring **Human-AI Interaction** in musical improvisation.

## System Architecture

The system consists of three main modules:

1.  **Training Module (`train_composer_models.py`):** Parses raw MIDI datasets, tokenizes polyphonic chords, and trains composer-specific LSTM models.
2.  **Inference Engine (`realtime_osc_streamer.py`):** Loads models, predicts next tokens in real-time, and manages a complex state machine to translate tokens into musical events (Note On/Off/Velocity).
3.  **Audio Synthesis (PureData):** Receives OSC messages and triggers high-quality samples via the LABS VST plugin using `vstplugin~`.

## Key Features

* **Polyphonic Generation:** Uses chord-based tokenization to generate harmonies, not just melodies.
* **Live Parameter Modulation:**
    * **Temperature:** Controls the "creativity" or randomness of the model.
    * **Density:** Controls the sparsity of notes (stochastic gating).
    * **Intensity:** Controls velocity dynamics and expression.
* **Smooth Transitions:** Implements parameter smoothing (interpolation) to avoid abrupt jumps in musical dynamics.
* **State-Based Note Logic:** A robust dispatcher handles note lifecycles, ensuring no "stuck notes" and allowing for smooth polyphonic voice allocation.
* **Hot-Swapping:** Switch between different composer models (e.g., from Bach to Beethoven) in real-time without stopping the audio engine.

## Installation

### Prerequisites

* Python 3.8+
* [PureData (Pd)](https://puredata.info/)
* [LABS VST Plugin](https://labs.spitfireaudio.com/) (or any VST instrument)
* [deken](https://github.com/pure-data/deken) (to install `vstplugin~` within Pd)

### Python Dependencies

* tensorflow
* numpy
* pretty_midi
* python-osc
* tqdm

pip install tensorflow numpy pretty_midi python-osc tqdm

## Usage

1. Data Preparation & Training
* Place your MIDI files in a folder structure organized by composer:

```
dataset/
    ├── Bach/
    │   ├── fugue1.mid
    │   └── ...
    ├── Beethoven/
    │   └── sonata.mid
    │   └── ...
    └── ...
```

* Run the training script to generate models:

python train_composer_models.py \
  --dataset ./dataset \
  --out_models ./models_v1 \
  --epochs 40 \
  --seq_len 64

* The script will create an `.h5` model file and a `vocab.json` for each composer.

2. Audio Engine Setup (PureData)

* Open the provided PureData patch (audio_engine.pd).

* Ensure the vstplugin~ object is loaded.

* Click the "Open" message in Pd to load your desired VST (e.g., Spitfire LABS).

* Ensure audio processing is turned on in Pd.

3. Running the Streamer

* Launch the real-time generator:

`python realtime_osc_streamer.py --models_dir ./models_v1 --osc_port 56120`

## Live Controls

Once the script is running, use the following keyboard shortcuts to interact with the AI:

| Key | Function | Description |
| :--- | :--- | :--- |
| **T / G** | Temperature +/- | Increases/Decreases prediction randomness (Entropy). |
| **D / F** | Density +/- | Probability gate for note generation (High = busy, Low = sparse). |
| **I / K** | Intensity +/- | Scales the output velocity (Dynamics). |
| **C** | Change Composer | Opens a menu to hot-swap the underlying AI model. |
| **Q** | Quit | Gracefully stops the engine and sends Note-Offs. |

## Technical Implementation Details

### Tokenization Strategy

To handle polyphony with a sequential LSTM, time is quantized into steps (default: 16th notes). Simultaneous notes are grouped into a set, sorted, and stringified into a single token (e.g., "60,64,67" for a C-Major chord).

### The Inference Loop

The `RealTimeOSCStreamer` class runs a continuous loop:
1. **Predict:** The model receives the last $N$ tokens and predicts the probability distribution for the next token.
2. **Sample:** A token is sampled based on the current Temperature.
3. **Gate:** The Density parameter stochastically decides if the token creates sound or silence.
4. **Dispatch:** The list of pitches is sent to the OSC manager.

### OSC State Machine

Sending raw MIDI data over UDP/OSC can lead to stuck notes if packets are lost or logic is flawed. This system implements a state tracker (`self.active` sets) that compares the target pitches with the currently sounding pitches.

* **New Notes:** Trigger `/note_on` and fade-in velocity.

* **Released Notes:** Trigger `/note_off`.

* **Sustained Notes:** Optionally receive velocity updates `/note_vel` for dynamic swelling without re-triggering envelopes.

## Licence

This project is licensed under the **MIT License**.

__Created by Malte Mittrowann__
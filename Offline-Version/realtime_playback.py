#!/usr/bin/env python3
import os, time, json, argparse, numpy as np, threading, random, sys
from pythonosc import udp_client
import tensorflow as tf

# ============================================================
# Utility helpers
# ============================================================

def list_models(models_dir):
    return sorted([d for d in os.listdir(models_dir)
                   if os.path.isdir(os.path.join(models_dir, d))])

def load_vocab_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def sample_with_temp(probs, temp=1.0):
    probs = np.asarray(probs).astype('float64')
    if temp <= 0:
        return int(np.argmax(probs))
    preds = np.log(probs + 1e-9) / temp
    preds = np.exp(preds) / np.sum(np.exp(preds))
    return int(np.random.choice(len(preds), p=preds))

def token_to_pitches(token):
    if not token or token in ("rest", "<UNK>"):
        return []
    try:
        return [int(x) for x in token.split(",")]
    except:
        return []

# ============================================================
# Realtime OSC generator (state-based note logic)
# ============================================================

class RealTimeOSCStreamer:
    def __init__(self, composer_path, osc_client,
                 step_seconds=0.125,
                 temperature=1.0, density=1.0, intensity=1.0,
                 smoothing=0.06,
                 max_polyphony=6):

        # OSC
        self.osc = osc_client
        self.step_seconds = step_seconds

        # smoothed params (values and targets)
        self.temperature = float(temperature)
        self.density = float(density)
        self.intensity = float(intensity)
        self.temperature_target = float(temperature)
        self.density_target = float(density)
        self.intensity_target = float(intensity)
        self.smoothing = float(smoothing)

        # polyphony + velocity smoothing
        self.max_polyphony = int(max_polyphony)
        # note_velocities stores smoothed velocity for notes we track
        # KEY invariant: presence in note_velocities does NOT mean the note is "active" (sent as on)
        self.note_velocities = {}   # pitch -> float velocity (0..127)
        # active set stores which notes are currently considered sounding (we sent note_on and haven't sent note_off)
        self.active = set()

        # model state
        self.model = None
        self.seq = []
        self.seq_len = 64
        self.id_to_token = {}
        self.token_to_id = {}

        self.model_lock = threading.Lock()

        # load first composer synchronously
        self.load_composer(composer_path, blocking=True)

    # --------------------------------------------------------

    def find_model_file(self, path):
        for f in ["best_model.h5", "final_model.h5"]:
            p = os.path.join(path, f)
            if os.path.exists(p):
                return p
        p = os.path.join(path, "model")
        if os.path.isdir(p):
            return p
        return None

    # --------------------------------------------------------

    def load_composer(self, composer_path, blocking=False):
        if blocking:
            self._load_composer_internal(composer_path)
        else:
            threading.Thread(target=self._load_composer_internal,
                             args=(composer_path,), daemon=True).start()

    # --------------------------------------------------------

    def _load_composer_internal(self, composer_path):

        vocab_path = os.path.join(composer_path, "vocab.json")
        if not os.path.exists(vocab_path):
            print(f"[ERROR] Missing vocab.json in {composer_path}")
            return

        vocab = load_vocab_json(vocab_path)

        # token_to_id maps token -> id (values might be strings in JSON)
        token_to_id = {token: int(idx) for token, idx in vocab["token_to_id"].items()}
        id_to_token = {int(idx): token for token, idx in vocab["token_to_id"].items()}
        seq_len = int(vocab.get("seq_len", 64))

        model_file = self.find_model_file(composer_path)
        if model_file is None:
            print(f"[ERROR] No model in {composer_path}")
            return

        print(f"[loader] Loading model from: {model_file}")
        new_model = tf.keras.models.load_model(model_file)
        print("[loader] Loaded.")

        # atomar replace model and reset sequence; clear state to avoid mismatches
        with self.model_lock:
            self.model = new_model
            self.id_to_token = id_to_token
            self.token_to_id = token_to_id
            self.seq_len = seq_len
            # reset sequence and note state on model switch to avoid stale indexes causing bad tokens
            self.seq = [0] * self.seq_len
            # when switching models we should also ensure no stuck notes remain:
            # send note_off for any active notes and clear state
            for p in list(self.active):
                try:
                    self.osc.send_message("/note_off", [int(p)])
                except:
                    pass
            self.active.clear()
            self.note_velocities.clear()

        print("[loader] Model switch complete. Now playing:", os.path.basename(composer_path))

    # --------------------------------------------------------

    def set_temperature(self, t): self.temperature_target = float(t)
    def set_density(self, d): self.density_target = float(d)
    def set_intensity(self, i): self.intensity_target = float(i)

    # --------------------------------------------------------

    def step(self):

        with self.model_lock:
            if self.model is None:
                return

        # smooth parameters
        self.temperature += (self.temperature_target - self.temperature) * self.smoothing
        self.density += (self.density_target - self.density) * self.smoothing
        self.intensity += (self.intensity_target - self.intensity) * self.smoothing

        # prepare input window
        window = self.seq[-self.seq_len:]
        if len(window) < self.seq_len:
            window = [0] * (self.seq_len - len(window)) + window

        inp = np.array([window], dtype=np.int32)

        with self.model_lock:
            preds = self.model.predict(inp, verbose=0)[0]

        next_id = sample_with_temp(preds, temp=max(0.01, self.temperature))
        self.seq.append(next_id)

        token = self.id_to_token.get(next_id, "<UNK>")
        pitches = token_to_pitches(token)

        # density gating
        if random.random() > self.density:
            pitches = []

        # enforce polyphony limit
        if len(pitches) > self.max_polyphony:
            pitches = pitches[:self.max_polyphony]

        # send OSC notes using stateful logic (fixed)
        self.send_osc_notes(pitches)

        # send debug params
        try:
            self.osc.send_message("/gen/token", token)
            self.osc.send_message("/gen/active", len(self.active))
            self.osc.send_message("/gen/params",
                                  [float(self.temperature),
                                   float(self.density),
                                   float(self.intensity)])
        except:
            pass

    # --------------------------------------------------------

    def send_osc_notes(self, target_pitches):
        """
        State-based note dispatcher:
         - send /note_on only when a pitch transitions from not-active -> active
         - send /note_off only when a pitch transitions from active -> not-active
         - internally smooth velocities, but do NOT resend /note_on repeatedly
         - optionally send /note_vel updates for continuing notes (commented)
        """
        target = set(target_pitches)
        prev = set(self.active)

        # Determine transitions
        to_on = target - prev      # newly requested notes -> send one note_on each
        to_off = prev - target     # notes to release -> send one note_off each
        continuing = target & prev  # notes that stay sounding

        # Initialize velocities for newly requested notes (start from 0 to allow fade-in)
        for p in to_on:
            if p not in self.note_velocities:
                self.note_velocities[p] = 0.0

        # Ensure we have velocity entries for continuing notes as well
        for p in continuing:
            self.note_velocities.setdefault(p, 0.0)

        # Also make sure any lingering note_velocities for notes not targetted will be handled and eventually removed

        # Velocity smoothing params
        fade_factor = 0.2
        vel_target = max(1.0, min(127.0, 100.0 * float(self.intensity)))  # target velocity for active notes

        # 1) Send note_off for to_off (do this first so PD/Instrument releases quickly)
        for p in to_off:
            try:
                self.osc.send_message("/note_off", [int(p)])
            except:
                pass
            # remove velocity tracking for released notes
            if p in self.note_velocities:
                self.note_velocities.pop(p, None)

        # 2) Update velocities & send note_on only when crossing threshold from zero to on
        newly_active = set()
        for p in list(self.note_velocities.keys()):
            v = self.note_velocities.get(p, 0.0)
            if p in target:
                # fade up toward vel_target
                new_v = v + (vel_target - v) * fade_factor
            else:
                # fade down toward 0
                new_v = v * (1.0 - fade_factor)

            # clamp
            new_v = max(0.0, min(127.0, new_v))
            self.note_velocities[p] = new_v

            # If this pitch is intended to be sounding (in target) and wasn't previously in active,
            # and we now have a usable velocity, send ONE note_on and register as active.
            if p in to_on:
                # Only send note_on once when it transitions from velocity 0 -> >0
                if new_v >= 1.0:
                    try:
                        self.osc.send_message("/note_on", [int(p), int(round(new_v))])
                    except:
                        pass
                    newly_active.add(p)
                # else: velocity still ramping, do not send yet (wait until >=1)
            elif p in continuing:
                # already active: we may optionally send velocity update messages
                # (some synths/Pd patches may implement a /note_vel message)
                # We'll send a /note_vel so PD can update amplitude without re-triggering.
                try:
                    self.osc.send_message("/note_vel", [int(p), int(round(new_v))])
                except:
                    pass
                newly_active.add(p)
            else:
                # p neither in target nor continuing -> it's fading out; if new_v < 1 it will be removed below
                pass

            # If velocity dropped below 1 and the note was not actively sounding, ensure cleanup
            if new_v < 1.0 and p not in newly_active:
                # If it somehow remained in active set, send note_off
                if p in self.active:
                    try:
                        self.osc.send_message("/note_off", [int(p)])
                    except:
                        pass
                    # remove from active
                    # will be updated later
                # remove from tracking
                try:
                    self.note_velocities.pop(p, None)
                except KeyError:
                    pass

        # 3) Update active set: union of previous continuing notes that stay above threshold and newly_active
        # Build new_active carefully: include those in 'continuing' with velocity >=1, plus newly_active
        new_active = set()
        for p in continuing:
            v = self.note_velocities.get(p, 0.0)
            if v >= 1.0:
                new_active.add(p)
            else:
                # velocity dropped -> ensure note_off was sent above
                try:
                    self.osc.send_message("/note_off", [int(p)])
                except:
                    pass
                # cleanup
                try:
                    self.note_velocities.pop(p, None)
                except KeyError:
                    pass

        new_active.update(newly_active)

        # assign new active set
        self.active = new_active

    # --------------------------------------------------------

    def stop_all(self):
        # request graceful fade
        self.set_density(0)
        self.set_intensity(0)
        for _ in range(12):
            self.step()
            time.sleep(self.step_seconds)
        # force send offs if any remain
        for p in list(self.active):
            try:
                self.osc.send_message("/note_off", [p])
            except:
                pass
        self.active.clear()
        self.note_velocities.clear()

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="./models")
    parser.add_argument("--osc_host", default="127.0.0.1")
    parser.add_argument("--osc_port", type=int, default=56120)
    parser.add_argument("--step_seconds", type=float, default=0.125)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--density", type=float, default=0.8)
    parser.add_argument("--intensity", type=float, default=0.8)
    parser.add_argument("--smoothing", type=float, default=0.06)
    parser.add_argument("--max_polyphony", type=int, default=6)
    args = parser.parse_args()

    osc = udp_client.SimpleUDPClient(args.osc_host, args.osc_port)

    composers = list_models(args.models_dir)
    if not composers:
        print("No models found.")
        return

    print("Available composers:")
    for i, c in enumerate(composers):
        print(f"  [{i}] {c}")

    idx = int(input("Choose index: "))
    composer_path = os.path.join(args.models_dir, composers[idx])

    streamer = RealTimeOSCStreamer(
        composer_path, osc,
        step_seconds=args.step_seconds,
        temperature=args.temperature,
        density=args.density,
        intensity=args.intensity,
        smoothing=args.smoothing,
        max_polyphony=args.max_polyphony
    )

    streamer.composers = composers
    streamer.models_dir = args.models_dir

    print("""
Controls:
 t/g = temperature up/down
 d/f = density up/down
 i/k = intensity up/down
 c   = change composer
 q   = quit
""")

    try:
        while True:

            key = None
            if os.name == "nt":
                import msvcrt
                if msvcrt.kbhit():
                    try: key = msvcrt.getch().decode()
                    except: pass
            else:
                import select
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                if dr:
                    key = sys.stdin.read(1)

            if key:
                key = key.lower()
                if key == "t":
                    streamer.set_temperature(min(streamer.temperature_target + 0.1, 10.0))
                elif key == "g":
                    streamer.set_temperature(max(0.01, streamer.temperature_target - 0.1))
                elif key == "d":
                    streamer.set_density(min(1, streamer.density_target + 0.1))
                elif key == "f":
                    streamer.set_density(max(0, streamer.density_target - 0.1))
                elif key == "i":
                    streamer.set_intensity(min(1.5, streamer.intensity_target + 0.1))
                elif key == "k":
                    streamer.set_intensity(max(0.1, streamer.intensity_target - 0.1))
                elif key == "c":
                    print("Choose composer:")
                    for i, c in enumerate(composers):
                        print(f"[{i}] {c}")
                    try:
                        idx = int(input("Index: "))
                        new_path = os.path.join(streamer.models_dir, composers[idx])
                        streamer.load_composer(new_path, blocking=False)
                        print("Switched composer (background load).")
                    except:
                        print("Invalid index.")
                elif key == "q":
                    break

                # print current targets and smooth values
                print(f"""
--- CURRENT GENERATOR STATE ---
Temperature (target): {streamer.temperature_target:.2f}
Density     (target): {streamer.density_target:.2f}
Intensity   (target): {streamer.intensity_target:.2f}

Temperature (smooth): {streamer.temperature:.2f}
Density     (smooth): {streamer.density:.2f}
Intensity   (smooth): {streamer.intensity:.2f}

Active notes: {len(streamer.active)}
Polyphony   : {streamer.max_polyphony}
------------------------------
""")

            streamer.step()
            time.sleep(args.step_seconds)

    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop_all()
        print("Bye.")


if __name__ == "__main__":
    main()

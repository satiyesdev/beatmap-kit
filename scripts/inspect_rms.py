# -*- coding: utf-8 -*-
"""Print short-time RMS dB for a time range (diagnostics)."""
import sys
import librosa
import numpy as np

wav, a, b = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
y, sr = librosa.load(wav, sr=44100, mono=True)
hop = 110  # 2.5 ms
rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
db = 20 * np.log10(np.maximum(rms, 1e-8))
for i, v in enumerate(db):
    t = i * hop / sr
    if a <= t <= b:
        print(f"{t:6.3f} {v:6.1f} " + "#" * max(0, int((v + 50))))
# -*- coding: utf-8 -*-
"""Quick vocal-presence check: compare mix vs Demucs vocal-stem RMS."""
import sys
import librosa
import numpy as np

mix_path, voc_path = sys.argv[1], sys.argv[2]
y, sr = librosa.load(mix_path, sr=44100, mono=True)
v, _ = librosa.load(voc_path, sr=44100, mono=True)
n = min(len(y), len(v))
y, v = y[:n], v[:n]
hop = 1024
rm = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
rv = librosa.feature.rms(y=v, frame_length=2048, hop_length=hop)[0]
dm = 20 * np.log10(np.maximum(rm, 1e-8))
dv = 20 * np.log10(np.maximum(rv, 1e-8))
print("mix_rms_db", round(float(20*np.log10(max(np.sqrt(np.mean(y**2)),1e-8))),2))
print("voc_rms_db", round(float(20*np.log10(max(np.sqrt(np.mean(v**2)),1e-8))),2))
print("vocal_minus_mix_db", round(float(20*np.log10(max(np.sqrt(np.mean(v**2)),1e-8))-20*np.log10(max(np.sqrt(np.mean(y**2)),1e-8))),2))
print("frame_mix_median", round(float(np.median(dm)),2), "frame_voc_median", round(float(np.median(dv)),2))
print("frac_voc_gt_mix_minus_15", round(float(np.mean(dv > dm - 15)),3))
print("frac_voc_gt_abs_minus_40", round(float(np.mean(dv > -40)),3))
print("corr", round(float(np.corrcoef(dm, dv)[0,1]),3))
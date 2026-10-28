# beatmap-kit

**Drop in a song, get one JSON beatmap for beat-synced editing.**

One command analyses a track and writes a single JSON with the tempo, every beat,
accented onsets, sustained loudness jumps, vocal onsets, per-kana Japanese lyric
timing and the section structure — everything you need to cut picture on the beat.

```bat
beatmap.bat "music\song.mp3"
```

→ `runs\20261028-143000-song\beats-song.json`

```
BPM 124.5 · 532 beats · 1623 onsets · 42 loudness jumps · 5 sections
```

---

## Features

| Output | What it is | How it is computed |
|---|---|---|
| **BPM / beats** | tempo + every beat pulse with strength | `librosa.beat.beat_track` (tightness 400) |
| **onsets / accents** | transient starts, normalised strength, accent flag | spectral-flux onset detection on the mix |
| **drum onsets** | onsets on the percussive component only | HPSS → onset detection |
| **loudness jumps** | sustained level steps (≥2.5 dB, held) | RMS → 0.5 s smoothing → before/after comparison |
| **vocal onsets** | syllable-attack candidates on the vocal stem | Demucs `htdemucs` separation → onset strength |
| **lyrics (Japanese)** | per-**kana** timing with a confidence score | faster-whisper ASR → WhisperX CTC alignment (wav2vec2 XLSR-53 JA) |
| **structure** | verse / chorus / bridge-ish sections with energy | librosa laplacian segmentation (allin1 optional) |
| **energy / density / jumps** | per-beat RMS, per-bar onset density, ≥4 dB beat drops | derived from the beats above |

All times are in seconds and rounded to 0.01 s. `t` is relative to the analysed
segment, `abs` is relative to the start of the song.

## Requirements

- Windows 10/11 (the batch files are Windows-only)
- Python **3.10 – 3.13** (tested on 3.12.13). If Python is not on `PATH`, run
  `set BEATMAP_PYTHON=C:\path\to\python.exe` before `setup.bat`.
- ~10 GB of disk: two virtualenvs (~7 GB) + ~2.7 GB of models
- NVIDIA GPU is optional. `setup.bat` installs the CUDA 12.8 build of torch;
  `setup.bat --cpu` installs the CPU build and everything still works (slower).
- `ffmpeg` on PATH **only** if you feed `.m4a` / `.aac` files

## Quick start

```bat
git clone https://github.com/satiyesdev/beatmap-kit.git
cd beatmap-kit

setup.bat            :: GPU machine (CUDA 12.8)
:: setup.bat --cpu   :: no NVIDIA GPU

beatmap.bat "music\song.mp3"
```

The first run downloads the models into `models\` (~2.7 GB):
faster-whisper-medium, wav2vec2-large-xlsr-53-japanese, and Demucs htdemucs.
They are cached — later runs are offline.

If the download stalls, point the Hugging Face client at a mirror:

```bat
set HF_ENDPOINT=https://hf-mirror.com
beatmap.bat "music\song.mp3"
```

### Analysing a segment instead of the whole song

```bat
beatmap.bat "music\song.mp3" 0:21 0:56
```

### With a lyrics file

```bat
beatmap.bat "music\song.mp3" --lyrics lyrics\song.txt --title "Song" --artist "Artist"
```

### Instrumental track (skips Demucs/ASR/lyrics)

```bat
beatmap.bat "music\instrumental.flac" --instrumental
```

## Command line options

| Option | Meaning |
|---|---|
| `input` | audio file: `.mp3` `.wav` `.flac` `.ogg` (`.m4a` `.aac` need ffmpeg) |
| `start` `end` | optional segment, `0:21` / `21` / `1:02:03`; default = whole song |
| `--lyrics PATH` | lyric file (see below); without it the ASR transcript is aligned |
| `--title` / `--artist` | default: guessed from the `Artist - Title` file name |
| `--slug` | output name, `beats-<slug>.json`; default = title |
| `--device auto\|cpu\|cuda` | `auto` uses CUDA when available, falls back to CPU |
| `--instrumental` | skip Demucs, vocal onsets, ASR and lyrics |
| `--align-mode auto\|per-line\|global\|asr` | how lyrics are matched to ASR segments |
| `--structure-method auto\|allin1\|librosa` | section segmentation backend |
| `--structure-k N` | number of laplacian clusters; `0` = auto (~one per 38 s) |
| `--demucs-shifts N` | `0` (deterministic, default) or `1` for random-shift ensembling |
| `--run-dir PATH` | force the output folder |

## Lyrics file

One lyric line per line, only the lines actually sung in the analysed segment.
Blank lines and `#` comments are ignored. An optional kana reading can follow `|`:

```text
# optional comment
震える心 疼いてる体 | ふるえるこころついてるからな
裏腹くわばら
結婚しましょう そうしましょう
```

Without `|`, the reading comes from `pykakasi`. If the number of lyric lines
equals the number of ASR segments, each line is aligned inside its own segment
(`per-line`), otherwise all kana are aligned in one pass (`global`).

## Output JSON

`examples/beats-rain.json` is a full instrumental run (坂本龍一 – Rain, 4:16) you
can compare against.

| Key | Contents |
|---|---|
| `schema` | `beatmap-kit/v1` |
| `source` | input name, sha256, segment wav, conversion info — **paths are relative to the run folder** |
| `segment` | start/end/duration of the analysed segment |
| `tempo` | `bpm`, `bpm_from_beats`, beat count, full-track tempo, tempogram candidates |
| `beats[]` | `i`, `t`, `abs`, `strength`, `nearest_onset_t`, `nearest_onset_dist` |
| `onsets[]` | `i`, `t`, `abs`, `strength`, `strength_norm`, `accent`, `on_beat`, `nearest_beat` |
| `drum_onsets[]` | same shape, percussive component only |
| `loudness.changes[]` | `t`, `abs`, `kind` (rise/drop), `delta_db`, `before_db`, `after_db`, `held_sec` |
| `lyrics_lines[]` | `i`, `text`, `kana`, `t`/`end_t`, `abs`/`end_abs`, `score_mean`, `confidence` |
| `vocals.syllable_onsets[]` | vocal onset candidates with `energy_rise_db` |
| `vocals.whisperx.kana_timeline[]` | one entry per kana: `kana`, `t`, `end_t`, `dur`, `score`, `confidence`, `line` |
| `vocals.whisperx.asr` | raw faster-whisper segments |
| `structure` | `method`, `segments[]` (`label`, `start_abs`, `end_abs`, `rms_db`, `onset_density_per_sec`), `segment_overlap` |
| `energy.per_beat[]` | RMS dBFS for every beat window |
| `density.bars[]` | onsets per 4-beat bar |
| `jumps.events[]` | beat-to-beat level jumps above 4 dB |
| `counts` | quick totals |
| `summary_zh` / `notes_zh` | human-readable Chinese summary + caveats |

## How it works

```
song ── prepare ──> segment.wav
                     ├─ librosa mix analysis ....... BPM, beats, onsets, loudness
                     ├─ Demucs htdemucs ............ vocals.wav
                     │   ├─ vocal onset detection
                     │   ├─ faster-whisper (JA ASR)
                     │   └─ WhisperX CTC alignment . kana + char timeline
                     ├─ build_json ................. beats-<slug>.json
                     └─ structure + energy ......... sections, per-beat energy,
                                                     per-bar density, drops
```

Two virtualenvs keep the memory footprint sane: `venv` (torch + Demucs, GPU when
available) and `venv_asr` (faster-whisper + WhisperX, CPU on purpose so Demucs
keeps the VRAM). Each stage runs in its own process.

## Helper scripts

| Script | Purpose |
|---|---|
| `scripts\check_run.py beats-x.json` | print counts, tempo, lyric lines |
| `scripts\validate_json.py beats-x.json` | check for NaN and time fields with >2 decimals |
| `scripts\show_extra.py extra.json` | print structure / energy / density / jumps |
| `scripts\inspect_region.py beats-x.json 12.5 18.0` | kana / onsets / lines in a time range |
| `scripts\check_vocal_energy.py segment.wav vocals.wav` | decide whether a track is instrumental |

## Limitations

- Windows only for now (the code is portable, the `.bat` files are not).
- Lyric alignment is **Japanese only** (kana); other languages fall back to the
  ASR transcript (`--align-mode asr`).
- ASR is tuned for singing: overlapping harmonies, heavy autotune or rap can
  produce low-confidence kana — the `confidence` field tells you which.
- `allin1` sections are optional and currently **not installable on Windows**
  (NATTEN has no Windows wheels); the librosa laplacian backend is the default.
- Only one song per run, no batch mode yet.

## Thanks

Built on the shoulders of [librosa](https://librosa.org/),
[Demucs](https://github.com/adefossez/demucs), [allin1](https://github.com/mir-aidj/all-in-one),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper),
[WhisperX](https://github.com/m-bain/whisperX), [pykakasi](https://github.com/elzup/pykakasi),
scikit-learn and Hugging Face.

## License

[MIT](LICENSE) © Satiyes

---

# 中文说明

**丢进一首歌，吐出一份 JSON**：BPM、每一拍、重音 onset、音量突变、人声起落、
日语歌词逐字（假名）时间轴、段落结构 —— 卡点剪辑需要的东西都在里面。

```bat
git clone https://github.com/satiyesdev/beatmap-kit.git
cd beatmap-kit
setup.bat            :: 有 NVIDIA 显卡；没有就 setup.bat --cpu
beatmap.bat "music\song.mp3"
```

输出在 `runs\<时间>-<slug>\beats-<slug>.json`。第一次运行会把约 2.7 GB 的模型
（faster-whisper-medium、wav2vec2-large-xlsr-53-japanese、htdemucs）下到 `models\`；
下载卡住的话先 `set HF_ENDPOINT=https://hf-mirror.com`。

只分析片段：`beatmap.bat song.mp3 0:21 0:56`
带歌词：`beatmap.bat song.mp3 --lyrics lyrics\song.txt`
纯音乐：`beatmap.bat song.flac --instrumental`（跳过 Demucs / ASR / 歌词）

歌词文件一行一句，只写这段里唱到的行；每行可用 `|` 附上假名读法：

```text
震える心 疼いてる体 | ふるえるこころついてるからな
裏腹くわばら
```

字段说明见上表；`examples/beats-rain.json` 是一份完整的纯音乐输出示例。
JSON 里的路径都相对于本次运行目录，方便直接分享。

目前只支持 Windows；歌词对齐只做日语。MIT 协议，署名 Satiyes。
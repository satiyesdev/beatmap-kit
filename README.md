# beatmap-kit

**Drop in a song, get one JSON beatmap for beat-synced editing.**
一键解析鼓点、歌词时间轴和段落，给卡点剪辑用

![beatmap-kit demo](assets/demo.gif)

*Every cut in this video was timed with beatmap-kit. / 这支视频里每一刀都是用 beatmap-kit 卡的。*

[English](#english) · [中文](#中文)

---

## English

beatmap-kit analyses one track and writes a single JSON with the tempo, every
beat, accents, loudness jumps, vocal onsets, per-kana Japanese lyric timing and
the section structure — everything you need to cut picture on the beat.

### Features

- **BPM** — tempo of the selected segment and of the full track
- **Beats** — every beat pulse with its strength
- **Onsets** — transient starts on the mix, plus a separate drum-only list
- **Accents** — onsets flagged as strong hits (normalised strength ≥ 0.60) and whether they land on a beat
- **Loudness jumps** — sustained level steps (≥ 2.5 dB, held ≥ 1 s) and per-beat drops/rises above 4 dB
- **Vocal onsets / offsets** — syllable-attack candidates on the Demucs vocal stem
- **Japanese lyric alignment** — per-kana timing with a confidence score (faster-whisper ASR → WhisperX alignment)
- **Section structure** — verse / chorus / bridge-style sections with energy; uses [allin1](https://github.com/mir-aidj/all-in-one) when it is installed, otherwise falls back to librosa laplacian segmentation (see [Limitations](#limitations))

### Requirements

- **Windows 10 / 11 only** for now — the `.bat` launchers are Windows-only
- Python **3.10 – 3.13** (tested on 3.12). If it is not on `PATH`, run
  `set BEATMAP_PYTHON=C:\path\to\python.exe` before `setup.bat`
- About 10 GB of disk: two virtualenvs (~7 GB) + ~2.7 GB of models
- An NVIDIA GPU is optional — `setup.bat --cpu` installs the CPU build and everything still works, just slower
- `ffmpeg` on `PATH`, only if you feed `.m4a` / `.aac` files

### Quick start

```bat
git clone https://github.com/satiyesdev/beatmap-kit.git
cd beatmap-kit

setup.bat
beatmap.bat song.mp3
```

`setup.bat` creates `venv\` and `venv_asr\` inside the project folder and installs
the CUDA 12.8 build of torch; use `setup.bat --cpu` on a machine without an NVIDIA GPU.

The first run downloads the models (~2.7 GB: faster-whisper-medium,
wav2vec2-large-xlsr-53-japanese, Demucs htdemucs) into `models\`. After that it
runs offline. If the download stalls, set a Hugging Face mirror first:

```bat
set HF_ENDPOINT=https://hf-mirror.com
```

**Where the output goes.** Every run gets its own folder under the project:

```
runs\<YYYYMMDD-HHMMSS>-<slug>\beats-<slug>.json
```

`<slug>` is the song title, taken from the file name (`Artist - Title.mp3` → `Title`)
or from `--title` / `--slug`. The same folder also holds the intermediate files
(`segment.wav`, `mix.json`, `extra.json`, `manifest.json`, `logs\`, and `demucs\` for vocal runs).
Use `--run-dir PATH` to choose the folder yourself. The path of the finished JSON is
printed at the end of the run.

More examples:

```bat
:: only analyse 0:21 – 0:56
beatmap.bat song.mp3 0:21 0:56

:: align your own lyrics instead of the ASR transcript
beatmap.bat song.mp3 --lyrics lyrics\song.txt --title "Song" --artist "Artist"

:: instrumental track: skip Demucs / ASR / lyrics
beatmap.bat instrumental.flac --instrumental
```

### Command line options

| Option | Meaning |
|---|---|
| `input` | audio file: `.mp3` `.wav` `.flac` `.ogg` (`.m4a` `.aac` need ffmpeg) |
| `start` `end` | optional segment, `0:21` / `21` / `1:02:03`; default = whole song |
| `--lyrics PATH` | lyric file (format below); without it the ASR transcript is aligned |
| `--title` / `--artist` | default: guessed from an `Artist - Title` file name |
| `--slug` | output name, `beats-<slug>.json`; default = title |
| `--device auto\|cpu\|cuda` | `auto` uses CUDA when available and falls back to CPU |
| `--instrumental` | skip Demucs, vocal onsets, ASR and lyrics |
| `--align-mode auto\|per-line\|global\|asr` | how lyrics are matched to ASR segments |
| `--structure-method auto\|allin1\|librosa` | section segmentation backend |
| `--structure-k N` | number of laplacian clusters; `0` = auto (about one per 38 s) |
| `--demucs-shifts N` | `0` (deterministic, default) or `1` for random-shift ensembling |
| `--run-dir PATH` | force the output folder |

### Lyrics file

One lyric line per line, only the lines actually sung in the analysed segment.
Blank lines and `#` comments are ignored. An optional kana reading can follow `|`:

```text
# optional comment
震える心 疼いてる体 | ふるえるこころついてるからな
裏腹くわばら
結婚しましょう そうしましょう
```

Without `|`, the reading comes from `pykakasi`. If the number of lyric lines equals
the number of ASR segments, each line is aligned inside its own segment (`per-line`);
otherwise all kana are aligned in one pass (`global`).

### Output JSON

[`examples/beats-rain.json`](examples/beats-rain.json) is a full instrumental run
(Rain, 4:16) to compare against. All times are in seconds, rounded to 0.01 s:
`t` is relative to the start of the analysed segment, `abs` is relative to the start of the song.

| Key | Contents |
|---|---|
| `schema` | `beatmap-kit/v1` |
| `analysis_mode` | `full` or `instrumental` |
| `title` / `artist` / `generated_at` | track info and run time |
| `time_convention` | what `t` and `abs` mean, plus the segment start/end |
| `source` | input name and type, source audio, duration, segment wav, conversion info — paths are relative to the run folder |
| `segment` | `start_sec`, `end_sec`, `duration_sec`, labels, `sample_rate`, `samples` |
| `tempo` | `bpm`, `bpm_from_beats`, `beat_count`, `full_track_bpm`, `tempogram_candidates`, `beat_interval_median_sec`, `bar_length_sec_4_4`, `time_signature_assumed` |
| `beats[]` | `i`, `t`, `abs`, `strength`, `frame`, `nearest_onset_t`, `nearest_onset_dist` |
| `onsets[]` | `i`, `t`, `abs`, `strength`, `strength_norm`, `accent` (`strength_norm ≥ 0.60`), `on_beat` (within 0.05 s of a beat), `nearest_beat`, `frame` |
| `drum_onsets[]` | `i`, `t`, `abs`, `strength`, `frame` — percussive component only |
| `loudness.changes[]` | sustained steps: `t`, `abs`, `kind` (`sustained_rise` / `sustained_drop`), `direction`, `delta_db`, `before_db`, `after_db`, `held_sec` |
| `loudness.profile_0p5s[]` | level curve: `t`, `abs`, `db` every 0.5 s |
| `vocals.syllable_onsets[]` | vocal onset candidates: `t`, `abs`, `strength`, `strength_norm`, `energy_rise_db` |
| `vocals.energy_onsets[]` | vocal energy-rise points |
| `vocals.whisperx.kana_timeline[]` | one entry per kana: `kana`, `t`, `end_t`, `dur`, `score`, `confidence` (`high` / `medium` / `low`), `line` |
| `vocals.whisperx.char_timeline[]` | per-character timing |
| `vocals.whisperx.asr` | raw faster-whisper segments |
| `lyrics_lines[]` | one entry per lyric line: `i`, `text`, `kana`, `t`, `end_t`, `abs`, `end_abs`, `dur`, `score_mean`, `confidence` |
| `structure` | `method` (`allin1` or `librosa_laplacian`), `segments[]` (`label`, `start_abs`, `end_abs`, `duration_sec`, `rms_db`, `onset_density_per_sec`, `label_confidence`), `segment_overlap[]` (sections clipped to the analysed segment) |
| `energy.per_beat[]` | RMS dBFS for every beat window: `i`, `t`, `abs`, `end_t`, `rms_db` |
| `density.bars[]` | onsets per 4-beat bar: `bar`, `t`, `onset_count`, `drum_onset_count`, `onsets_per_sec` |
| `jumps.events[]` | beat-to-beat level jumps above 4 dB: `from_beat`, `to_beat`, `t`, `delta_db`, `direction` |
| `counts` | quick totals |
| `summary_zh` / `notes_zh` | human-readable Chinese summary and caveats |

### How it works

```
song ── prepare ──> segment.wav
                     ├─ librosa mix analysis ....... BPM, beats, onsets, loudness
                     ├─ Demucs htdemucs ............ vocals.wav
                     │   ├─ vocal onset detection
                     │   ├─ faster-whisper (JA ASR)
                     │   └─ WhisperX CTC alignment . kana + char timeline
                     ├─ build_json ................. beats-<slug>.json
                     └─ structure + energy ......... sections, per-beat energy,
                                                     per-bar density, jumps
```

Two virtualenvs keep memory use sane: `venv` (torch + Demucs, GPU when available)
and `venv_asr` (faster-whisper + WhisperX, CPU on purpose so Demucs keeps the VRAM).
Each stage runs in its own process.

### Helper scripts

| Script | Purpose |
|---|---|
| `scripts\check_run.py beats-x.json` | print counts, tempo, lyric lines |
| `scripts\validate_json.py beats-x.json` | check for NaN and time fields with more than 2 decimals |
| `scripts\show_extra.py extra.json` | print structure / energy / density / jumps |
| `scripts\inspect_region.py beats-x.json 12.5 18.0` | kana / onsets / lines in a time range |
| `scripts\check_vocal_energy.py segment.wav vocals.wav` | decide whether a track is instrumental |

### Limitations

- Windows only for now (the Python code is portable, the `.bat` files are not).
- Lyric alignment is **Japanese only** (kana); other languages fall back to the ASR transcript (`--align-mode asr`).
- ASR is tuned for singing: overlapping harmonies, heavy autotune or rap can produce
  low-confidence kana — the `confidence` field tells you which.
- **allin1 is optional and currently cannot be installed on Windows** (NATTEN ships no
  Windows wheels), so on Windows the section structure comes from the librosa
  laplacian backend. Labels from that backend are heuristic; `structure.method` tells you which one ran.
- One song per run, no batch mode yet.

### Credits

Built on [librosa](https://librosa.org/), [Demucs](https://github.com/adefossez/demucs),
[allin1](https://github.com/mir-aidj/all-in-one),
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) and
[WhisperX](https://github.com/m-bain/whisperX), with
[pykakasi](https://github.com/miurahr/pykakasi), scikit-learn and Hugging Face along the way.

### License

[MIT](LICENSE)

---

## 中文

**一键解析鼓点、歌词时间轴和段落，给卡点剪辑用。**

丢进一首歌，输出一份 JSON：BPM、每一拍、重音、音量突变、人声起落、日语歌词逐字
（假名）时间、段落结构 —— 卡点剪辑要用的东西都在里面。

*这支视频里每一刀都是用 beatmap-kit 卡的（动图见页面顶部）。*

### 功能

- **BPM** —— 所选片段与整首歌的速度
- **beat** —— 每一拍的位置和强度
- **onset** —— 整体混音的瞬态起点，另有只含鼓组的 onset 列表
- **重音** —— 强度归一化 ≥ 0.60 的 onset 标为重音，并标明是否落在拍点上
- **音量突变** —— 持续型电平台阶（≥ 2.5 dB，持续 ≥ 1 s），以及逐拍间超过 4 dB 的起伏
- **人声起落** —— 基于 Demucs 人声轨的音节起音候选
- **日语歌词逐字对齐（假名）** —— 每个假名的时间 + 置信度（faster-whisper 识别 → WhisperX 对齐）
- **段落结构（allin1）** —— verse / chorus / bridge 等段落及能量；装了
  [allin1](https://github.com/mir-aidj/all-in-one) 就用它，否则退回 librosa laplacian 分段（见[局限](#局限)）

### 环境要求

- **目前只支持 Windows 10 / 11**（启动脚本是 `.bat`）
- Python **3.10 – 3.13**（在 3.12 上测试）。Python 不在 `PATH` 里的话，运行
  `setup.bat` 前先 `set BEATMAP_PYTHON=C:\path\to\python.exe`
- 约 10 GB 磁盘：两个虚拟环境（约 7 GB）+ 约 2.7 GB 模型
- NVIDIA 显卡可选：`setup.bat --cpu` 装 CPU 版，一样能跑，只是慢一些
- 只有输入 `.m4a` / `.aac` 时才需要 `ffmpeg`

### 快速开始

```bat
git clone https://github.com/satiyesdev/beatmap-kit.git
cd beatmap-kit

setup.bat
beatmap.bat song.mp3
```

`setup.bat` 会在项目目录里建 `venv\` 和 `venv_asr\`，并安装 CUDA 12.8 版 torch；
没有 NVIDIA 显卡就用 `setup.bat --cpu`。

第一次运行会把约 2.7 GB 的模型（faster-whisper-medium、wav2vec2-large-xlsr-53-japanese、
Demucs htdemucs）下载到 `models\`，之后完全离线。下载卡住的话先设置镜像：

```bat
set HF_ENDPOINT=https://hf-mirror.com
```

**输出在哪里**：每次运行都会在项目目录下新建一个文件夹：

```
runs\<YYYYMMDD-HHMMSS>-<slug>\beats-<slug>.json
```

`<slug>` 是歌名：取自文件名（`歌手 - 歌名.mp3` → `歌名`），或由 `--title` / `--slug` 指定。
同一个文件夹里还有中间产物（`segment.wav`、`mix.json`、`extra.json`、`manifest.json`、
`logs\`，带人声的歌还有 `demucs\`）。想自己指定输出目录用 `--run-dir PATH`。
运行结束时会打印最终 JSON 的路径。

更多用法：

```bat
:: 只分析 0:21 – 0:56
beatmap.bat song.mp3 0:21 0:56

:: 用自己的歌词对齐，而不是 ASR 识别结果
beatmap.bat song.mp3 --lyrics lyrics\song.txt --title "Song" --artist "Artist"

:: 纯音乐：跳过 Demucs / ASR / 歌词
beatmap.bat instrumental.flac --instrumental
```

命令行参数见上方英文部分的 *Command line options* 表。

### 歌词文件

一行一句，只写这段里实际唱到的行；空行和以 `#` 开头的行会被忽略。每行可以用 `|`
附上假名读法：

```text
# 注释
震える心 疼いてる体 | ふるえるこころついてるからな
裏腹くわばら
結婚しましょう そうしましょう
```

不写 `|` 时读法由 `pykakasi` 生成。歌词行数等于 ASR 段数时每行在各自的段内对齐
（`per-line`），否则所有假名一次性整体对齐（`global`）。

### 输出 JSON 字段说明

[`examples/beats-rain.json`](examples/beats-rain.json) 是一份完整的纯音乐输出（Rain，4:16），可以对照着看。
所有时间单位为秒，四舍五入到 0.01 s：`t` 相对所选片段起点，`abs` 相对整首歌开头。

| 字段 | 内容 |
|---|---|
| `schema` | `beatmap-kit/v1` |
| `analysis_mode` | `full`（带人声）或 `instrumental`（纯音乐） |
| `title` / `artist` / `generated_at` | 曲目信息与生成时间 |
| `time_convention` | `t` / `abs` 的含义，以及片段起止 |
| `source` | 输入文件名与类型、源音频、时长、片段 wav、转换信息 —— 路径都相对于本次运行目录 |
| `segment` | `start_sec`、`end_sec`、`duration_sec`、标签、`sample_rate`、`samples` |
| `tempo` | `bpm`、`bpm_from_beats`、`beat_count`、`full_track_bpm`、`tempogram_candidates`、`beat_interval_median_sec`、`bar_length_sec_4_4`、`time_signature_assumed` |
| `beats[]` | `i`、`t`、`abs`、`strength`、`frame`、`nearest_onset_t`、`nearest_onset_dist` |
| `onsets[]` | `i`、`t`、`abs`、`strength`、`strength_norm`、`accent`（`strength_norm ≥ 0.60`）、`on_beat`（离某一拍 ≤ 0.05 s）、`nearest_beat`、`frame` |
| `drum_onsets[]` | `i`、`t`、`abs`、`strength`、`frame` —— 仅打击乐分量 |
| `loudness.changes[]` | 持续型音量突变：`t`、`abs`、`kind`（`sustained_rise` / `sustained_drop`）、`direction`、`delta_db`、`before_db`、`after_db`、`held_sec` |
| `loudness.profile_0p5s[]` | 每 0.5 s 一个点的电平曲线：`t`、`abs`、`db` |
| `vocals.syllable_onsets[]` | 人声起音候选：`t`、`abs`、`strength`、`strength_norm`、`energy_rise_db` |
| `vocals.energy_onsets[]` | 人声能量上升点 |
| `vocals.whisperx.kana_timeline[]` | 每个假名一条：`kana`、`t`、`end_t`、`dur`、`score`、`confidence`（`high` / `medium` / `low`）、`line` |
| `vocals.whisperx.char_timeline[]` | 逐字符时间 |
| `vocals.whisperx.asr` | faster-whisper 原始识别段 |
| `lyrics_lines[]` | 每句歌词一条：`i`、`text`、`kana`、`t`、`end_t`、`abs`、`end_abs`、`dur`、`score_mean`、`confidence` |
| `structure` | `method`（`allin1` 或 `librosa_laplacian`）、`segments[]`（`label`、`start_abs`、`end_abs`、`duration_sec`、`rms_db`、`onset_density_per_sec`、`label_confidence`）、`segment_overlap[]`（裁到所选片段内的段落） |
| `energy.per_beat[]` | 每一拍窗口的 RMS dBFS：`i`、`t`、`abs`、`end_t`、`rms_db` |
| `density.bars[]` | 每 4 拍一小节的 onset 数：`bar`、`t`、`onset_count`、`drum_onset_count`、`onsets_per_sec` |
| `jumps.events[]` | 相邻拍之间超过 4 dB 的电平跳变：`from_beat`、`to_beat`、`t`、`delta_db`、`direction` |
| `counts` | 各项数量汇总 |
| `summary_zh` / `notes_zh` | 中文摘要与注意事项 |

### 局限

- 目前只支持 Windows（Python 代码本身可移植，`.bat` 不行）。
- 歌词对齐只做**日语**（假名）；其他语言退回 ASR 文本（`--align-mode asr`）。
- ASR 针对歌声做过调整，但重叠和声、重度修音、说唱仍可能给出低置信度假名 ——
  用 `confidence` 字段筛掉即可。
- **allin1 是可选依赖，目前在 Windows 上装不上**（NATTEN 没有 Windows 轮子），所以 Windows 下段落结构
  走的是 librosa laplacian 方案，标签是启发式的；`structure.method` 会告诉你实际用了哪个。
- 一次只处理一首歌，暂无批处理。

### 致谢

基于 [librosa](https://librosa.org/)、[Demucs](https://github.com/adefossez/demucs)、
[allin1](https://github.com/mir-aidj/all-in-one)、
[faster-whisper](https://github.com/SYSTRAN/faster-whisper)、
[WhisperX](https://github.com/m-bain/whisperX)，
以及 [pykakasi](https://github.com/miurahr/pykakasi)、scikit-learn 和 Hugging Face。

### 许可

[MIT](LICENSE)

# -*- coding: utf-8 -*-
"""Step 6: merge all step outputs into the beatmap-kit/v1 JSON."""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json, r2, write_json  # noqa: E402

TIME_KEYS = {
    "t", "abs", "end_t", "end_abs", "start", "end", "dur",
    "held_sec", "nearest_onset_t", "nearest_syllable_onset_t",
    "offset", "seconds", "start_sec", "end_sec", "duration_sec",
    "beat_interval_median_sec", "bar_length_sec_4_4",
    "source_duration_sec", "start_label", "end_label",
}


def round_times(x):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in TIME_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = round(float(v), 2)
            else:
                out[k] = round_times(v)
        return out
    if isinstance(x, list):
        return [round_times(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--instrumental", action="store_true",
                    help="skip vocals/ASR/lyrics and emit an empty vocals block")
    args = ap.parse_args()
    run = Path(args.run)

    meta = read_json(run / "meta.json")
    mix = read_json(run / "mix.json")
    if args.instrumental:
        voc = {"method": None, "syllable_onsets": [], "energy_onsets": []}
        demucs = {"tool": "skipped (instrumental)", "device": "skipped"}
        asr = {"status": "skipped", "segments": [], "text": ""}
        ali = {"status": "skipped", "mode": "instrumental", "lyrics_lines": [],
               "kana_timeline": [], "char_timeline": [], "alignment": None}
    else:
        voc = read_json(run / "vocals.json")
        demucs = read_json(run / "demucs.json")
        asr = read_json(run / "asr.json")
        ali = read_json(run / "alignment.json")

    lines = ali.get("lyrics_lines", [])
    kana = ali.get("kana_timeline", [])
    tempo = dict(mix["tempo"])
    if tempo.get("bpm"):
        ibi = 60.0 / float(tempo["bpm"])
        tempo["beat_interval_median_sec"] = round(ibi, 3)
        tempo["bar_length_sec_4_4"] = round(ibi * 4, 3)
        tempo["time_signature_assumed"] = "4/4"
    changes = mix.get("loudness_changes", [])
    change_text = "；".join(
        f"{c['t']:.2f}s {c['direction']} {abs(c['delta_db']):.2f} dB" for c in changes
    ) or "无"

    summary_base = (
        f"{mix['segment']['start_label']}–{mix['segment']['end_label']}"
        f"（{mix['segment']['duration_sec']:.2f} s）速度 {mix['tempo']['bpm']:.2f} BPM，"
        f"检测到 {len(mix['beats'])} 个 beat、{len(mix['onsets'])} 个 onset、"
        f"{len(mix['drum_onsets'])} 个鼓点 onset。"
        f"持续音量突变：{change_text}。"
    )
    if args.instrumental:
        summary = summary_base + " 纯音乐：已跳过 Demucs 人声分离、ASR、歌词/假名。"
    else:
        summary = summary_base + (
            f" Demucs 人声轨上检测到 {len(voc.get('syllable_onsets', []))} 个音节起始候选。"
            f" WhisperX 对齐得到 {len(kana)} 个假名、{len(lines)} 行歌词。"
        )

    notes = [
        "t 是相对所选片段起点的秒数，abs 是相对歌曲开头的秒数；所有时间四舍五入到 0.01 s。",
        "beat 是节拍脉冲（librosa beat_track, tightness=400）；onset 是瞬态起始；drum_onsets 是 HPSS 打击乐分量上的 onset。",
        "音量突变只列持续型平均电平变化（0.5 s 平滑，前后 0.40 s 比较，>=2.5 dB），鼓点造成的瞬时 RMS 尖峰不算。",
    ]
    if args.instrumental:
        notes.append("纯音乐：Demucs/ASR/WhisperX 全部跳过；vocals 字段保留为空结构。")
    else:
        notes += [
            "人声轨由 Demucs htdemucs 分出；syllable_onsets 是频谱通量 onset 候选，可能包含辅音、气声或颤音造成的额外点。",
            "lyrics_lines 的起止时间取该行第一个/最后一个对齐假名，不再用 ASR 段边界；t/end_t 即该行实际演唱区间。",
            "kana_timeline 每行是一个假名，score/confidence 可直接筛低置信度项；wav2vec2 帧移约 20 ms，0.01 s 是输出精度，不是真实分辨率。",
            "演唱、和声、ASR 错误与歌词文件不匹配都可能造成时间偏移；低置信度行建议人工复核。",
        ]
        if ali.get("mode") == "global":
            notes.append("本次为全局歌词对齐：歌词文件应只包含所选时间段内唱到的行。")
        if ali.get("mode") == "per-line":
            notes.append("本次为逐行对齐：歌词行数与 ASR 段数一致，每行在对应 ASR 段时间窗内对齐。")

    result = {
        "schema": "beatmap-kit/v1",
        "analysis_mode": "instrumental" if args.instrumental else "full",
        "title": meta.get("title", ""),
        "artist": meta.get("artist", ""),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "time_convention": {
            "t": "seconds from segment start, rounded to 0.01 s",
            "abs": "seconds from the beginning of the song, rounded to 0.01 s",
            "segment_start_sec": meta["start_sec"],
            "segment_end_sec": meta["end_sec"],
        },
        "source": {
            "input": meta["input"],
            "input_type": meta["input_type"],
            "input_sha256": meta.get("input_sha256"),
            "source_audio": Path(meta["source_audio"]).name,
            "source_audio_sha256": meta.get("source_audio_sha256"),
            "source_duration_sec": meta.get("source_duration_sec"),
            "segment_wav": meta["wav"],
            "conversion": meta.get("conversion"),
        },
        "segment": mix["segment"],
        "tempo": tempo,
        "beats": mix["beats"],
        "onsets": mix["onsets"],
        "drum_onsets": mix["drum_onsets"],
        "loudness": {
            "method": mix.get("loudness_method"),
            "changes": changes,
            "profile_0p5s": mix.get("loudness_profile_0p5s", []),
            "note": ("changes are sustained mix-level steps (>=2.5 dB, >=1 s apart); "
                     "single drum hits / short transients are intentionally not listed here"),
        },
        "lyrics_lines": lines,
        "vocals": {
            "status": "skipped" if args.instrumental else "analyzed",
            "reason": ("instrumental track: Demucs/ASR/WhisperX skipped"
                       if args.instrumental else None),
            "separation": {
                "tool": demucs.get("tool", "Demucs htdemucs"),
                "device": demucs.get("device"),
                "input": "segment.wav",
                "vocals": demucs.get("vocals"),
                "no_vocals": demucs.get("no_vocals"),
                "skipped": bool(args.instrumental),
                "duration_sec": mix["segment"]["duration_sec"],
            },
            "onset_method": voc.get("method"),
            "syllable_onsets": voc.get("syllable_onsets", []),
            "energy_onsets": voc.get("energy_onsets", []),
            "whisperx": {
                "status": ali.get("status", "ok"),
                "mode": ali.get("mode"),
                "text_source": ali.get("text_source"),
                "asr": asr,
                "alignment": ali.get("alignment"),
                "kana_timeline": kana,
                "char_timeline": ali.get("char_timeline", []),
                "lines": lines,
            },
        },
        "counts": {
            "beats": len(mix["beats"]),
            "onsets": len(mix["onsets"]),
            "drum_onsets": len(mix["drum_onsets"]),
            "loudness_changes": len(changes),
            "vocal_syllable_onsets": len(voc.get("syllable_onsets", [])),
            "vocal_energy_onsets": len(voc.get("energy_onsets", [])),
            "kana": len(kana),
            "lyrics_lines": len(lines),
            "aligned_chars": len(ali.get("char_timeline", [])),
        },
        "summary_zh": summary,
        "notes_zh": notes,
    }
    result = round_times(result)
    write_json(args.out, result)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size} bytes)")
    print(summary)
    print(json.dumps(result["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
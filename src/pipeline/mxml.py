from __future__ import annotations

import bisect
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import music21 as m21

DEFAULT_BPM = 60.0
LEGACY_MERGE_SAME_PITCH_LEGATO = False

# ============================================================
# 输出文件开关：只改 True / False 即可
# ============================================================
# 默认只生成一个大表：*_score_final.csv
OUTPUT_SCORE_FINAL = True

# 以下四项默认关闭。需要调试时再改成 True。
OUTPUT_SCORE_NOTES = False
OUTPUT_SCORE_RELATIONS = False
OUTPUT_SCORE_DIRECTIONS = False
OUTPUT_SCORE_META = False
# ============================================================

LEGACY_COLUMNS = [
    "累计时间(s)",
    "音高频率(Hz)",
    "持续时间(s)",
    "音名",
    "累计时长(s)",
    "连奏信息",
    "BPM",
    "装饰音判定",
]



# Only map words that are explicit enough. Unknown words remain in 谱面文字.


def find_musescore_executable() -> Path:
    """Locate MuseScore for the optional MSCZ-to-MusicXML conversion."""
    configured = os.environ.get("MUSESCORE_EXE", "").strip()
    command_candidates = [
        configured,
        "MuseScore4.exe",
        "MuseScore3.exe",
        "mscore.exe",
        "musescore.exe",
    ]
    for command in command_candidates:
        if not command:
            continue
        resolved = shutil.which(command)
        if resolved:
            return Path(resolved).resolve()
        candidate = Path(command)
        if candidate.is_file():
            return candidate.resolve()

    if os.name == "nt":
        for candidate in [
        ]:
            if candidate.is_file():
                return candidate.resolve()
    raise RuntimeError(
        "发现 .mscz，但未找到 MuseScore。请安装 MuseScore，或通过 "
        "MUSESCORE_EXE 指定可执行文件。"
    )


def _run_musescore_export(
    executable: Path,
    input_path: Path,
    output_path: Path,
    *,
    isolated_runtime: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    runtime_context = None
    if isolated_runtime:
        runtime_context = tempfile.TemporaryDirectory(prefix="erhu_musescore_runtime_")
        runtime_dir = runtime_context.name
        env["APPDATA"] = runtime_dir
        env["LOCALAPPDATA"] = runtime_dir
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        return subprocess.run(
            [str(executable), "-o", str(output_path), str(input_path)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            env=env,
        )
    finally:
        if runtime_context is not None:
            runtime_context.cleanup()


def convert_mscz_to_musicxml(input_path: Path, output_path: Path) -> Path:
    """Export one MuseScore project as plain MusicXML and validate it."""
    executable = find_musescore_executable()
    failures = []
    for isolated_runtime in (False, True):
        if output_path.exists():
            output_path.unlink()
        try:
            result = _run_musescore_export(
                executable,
                input_path,
                output_path,
                isolated_runtime=isolated_runtime,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append(f"timeout: {exc}")
            continue
        if result.returncode == 0 and output_path.is_file():
            try:
                ET.parse(output_path)
            except Exception as exc:
                failures.append(f"invalid exported MusicXML: {exc}")
                continue
            print(f"Converted MSCZ to MusicXML: {input_path.name} -> {output_path.name}")
            return output_path
        details = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
        failures.append(
            f"return_code={result.returncode}, isolated_runtime={isolated_runtime}"
            + (f"\n{details}" if details else "")
        )
    raise RuntimeError(
        f"MuseScore 无法转换 {input_path.name}：\n" + "\n--- retry ---\n".join(failures)
    )


def choose_input_file(base_dir: Path) -> Path:
    """Use MusicXML when present; otherwise convert the only MSCZ project."""
    candidates = sorted(
        p for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".musicxml", ".xml"}
    )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(
            "当前目录存在多个 MusicXML/XML 文件，请只保留一个后重新运行：\n"
            f"{names}"
        )
    if candidates:
        return candidates[0]

    mscz_candidates = sorted(
        p for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".mscz"
    )
    if not mscz_candidates:
        raise FileNotFoundError(
            f"当前目录没有 .musicxml/.xml/.mscz 文件：{base_dir}"
        )
    if len(mscz_candidates) > 1:
        names = ", ".join(p.name for p in mscz_candidates)
        raise RuntimeError(
            "当前目录存在多个 MSCZ 文件，请只保留一个后重新运行：\n"
            f"{names}"
        )
    mscz_path = mscz_candidates[0]
    output_path = mscz_path.with_suffix(".musicxml")
    return convert_mscz_to_musicxml(mscz_path, output_path)

def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clean_token(value: Any, fallback: str) -> str:
    text = str(value if value not in (None, "") else fallback)
    text = re.sub(r"[^0-9A-Za-z_-]+", "_", text.strip())
    return text or fallback


def element_offset_in_part(element: Any, part: m21.stream.Part) -> float:
    try:
        return float(element.getOffsetInHierarchy(part))
    except Exception:
        return float(getattr(element, "offset", 0.0))


def get_voice_id(element: Any) -> str:
    try:
        voice = element.getContextByClass(m21.stream.Voice)
        if voice is not None:
            return clean_token(getattr(voice, "id", None), "1")
    except Exception:
        pass
    return "1"


def get_measure_number(element: Any) -> str:
    value = getattr(element, "measureNumber", None)
    if value is None:
        try:
            measure = element.getContextByClass(m21.stream.Measure)
            value = getattr(measure, "number", None)
        except Exception:
            value = None
    return clean_token(value, "0")


def get_time_signature(element: Any) -> str:
    try:
        ts = element.getContextByClass(m21.meter.TimeSignature)
        return ts.ratioString if ts is not None else ""
    except Exception:
        return ""


def get_key_fifths(element: Any) -> int | None:
    try:
        key_sig = element.getContextByClass(m21.key.KeySignature)
        return int(key_sig.sharps) if key_sig is not None else None
    except Exception:
        return None


def xml_local_name(tag: str) -> str:
    return str(tag).split("}")[-1]


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if xml_local_name(child.tag) == name:
            return child
    return None


def direct_child_text(element: ET.Element, name: str, default: str = "") -> str:
    child = direct_child(element, name)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def collect_wedge_intervals(
    input_path: Path,
    score: m21.stream.Score,
) -> dict[int, list[dict[str, Any]]]:
    """Read crescendo/diminuendo directions from raw MusicXML with exact offsets."""
    tree = ET.parse(input_path)
    root = tree.getroot()
    raw_parts = [child for child in list(root) if xml_local_name(child.tag) == "part"]
    result: dict[int, list[dict[str, Any]]] = {}

    for part_index, raw_part in enumerate(raw_parts, start=1):
        if part_index > len(score.parts):
            break
        music_part = score.parts[part_index - 1]
        music_measures = list(music_part.getElementsByClass(m21.stream.Measure))
        raw_measures = [child for child in list(raw_part) if xml_local_name(child.tag) == "measure"]

        divisions = 1.0
        events: list[dict[str, Any]] = []
        sequence = 0

        for measure_index, raw_measure in enumerate(raw_measures):
            if measure_index < len(music_measures):
                measure_start = element_offset_in_part(music_measures[measure_index], music_part)
            elif music_measures:
                last = music_measures[-1]
                measure_start = element_offset_in_part(last, music_part) + float(last.barDuration.quarterLength)
            else:
                measure_start = 0.0

            cursor_q = 0.0
            for child in list(raw_measure):
                tag = xml_local_name(child.tag)

                if tag == "attributes":
                    value = direct_child_text(child, "divisions")
                    if value:
                        parsed = safe_float(value)
                        if math.isfinite(parsed) and parsed > 0:
                            divisions = parsed
                    continue

                if tag == "direction":
                    offset_div = safe_float(direct_child_text(child, "offset"), 0.0)
                    if not math.isfinite(offset_div):
                        offset_div = 0.0
                    event_q = measure_start + cursor_q + offset_div / divisions
                    for descendant in child.iter():
                        if xml_local_name(descendant.tag) != "wedge":
                            continue
                        wedge_type = str(descendant.attrib.get("type", "")).strip().lower()
                        number = str(descendant.attrib.get("number", "1") or "1")
                        if wedge_type in {"crescendo", "diminuendo", "stop"}:
                            sequence += 1
                            events.append(
                                {
                                    "offset_quarter": float(event_q),
                                    "type": wedge_type,
                                    "number": number,
                                    "sequence": sequence,
                                }
                            )
                    continue

                if tag in {"backup", "forward"}:
                    duration_div = safe_float(direct_child_text(child, "duration"), 0.0)
                    duration_q = duration_div / divisions if math.isfinite(duration_div) else 0.0
                    cursor_q += -duration_q if tag == "backup" else duration_q
                    continue

                if tag == "note":
                    is_chord = direct_child(child, "chord") is not None
                    is_grace = direct_child(child, "grace") is not None
                    duration_div = safe_float(direct_child_text(child, "duration"), 0.0)
                    duration_q = duration_div / divisions if math.isfinite(duration_div) else 0.0
                    if not is_chord and not is_grace:
                        cursor_q += duration_q

        active: dict[str, tuple[str, float]] = {}
        intervals: list[dict[str, Any]] = []
        for event in sorted(events, key=lambda row: (row["offset_quarter"], row["sequence"])):
            event_type = event["type"]
            number = event["number"]
            offset_q = float(event["offset_quarter"])
            if event_type in {"crescendo", "diminuendo"}:
                active[number] = (event_type, offset_q)
            elif event_type == "stop" and number in active:
                start_type, start_q = active.pop(number)
                intervals.append(
                    {
                        "kind": start_type,
                        "number": number,
                        "start_quarter": float(start_q),
                        "stop_quarter": float(offset_q),
                    }
                )

        # Preserve unmatched starts as exact start markers rather than inventing an end.
        for number, (start_type, start_q) in active.items():
            intervals.append(
                {
                    "kind": start_type,
                    "number": number,
                    "start_quarter": float(start_q),
                    "stop_quarter": math.nan,
                }
            )

        result[part_index] = intervals

    return result


def dynamic_change_at_offset(intervals: list[dict[str, Any]], offset: float) -> str:
    labels: list[str] = []
    eps = 1e-7
    for interval in intervals:
        kind = str(interval.get("kind", ""))
        prefix = "渐强" if kind == "crescendo" else "渐弱" if kind == "diminuendo" else ""
        if not prefix:
            continue
        start = safe_float(interval.get("start_quarter"))
        stop = safe_float(interval.get("stop_quarter"))
        if not math.isfinite(start):
            continue

        if math.isfinite(stop):
            if abs(stop - start) <= eps and abs(offset - start) <= eps:
                labels.append(f"{prefix}起止点")
            elif abs(offset - start) <= eps:
                labels.append(f"{prefix}起点")
            elif abs(offset - stop) <= eps:
                labels.append(f"{prefix}结束")
            elif start < offset < stop:
                labels.append(f"{prefix}中")
        elif abs(offset - start) <= eps:
            labels.append(f"{prefix}起点")

    return "|".join(dict.fromkeys(labels))


def build_tempo_map(score: m21.stream.Score) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for mark in score.flatten().getElementsByClass(m21.tempo.MetronomeMark):
        bpm = safe_float(mark.number)
        if not math.isfinite(bpm) or bpm <= 0:
            continue
        points.append((float(mark.offset), bpm))

    if not points:
        return [(0.0, DEFAULT_BPM)]

    points.sort(key=lambda item: item[0])
    collapsed: list[tuple[float, float]] = []
    for offset, bpm in points:
        if collapsed and abs(collapsed[-1][0] - offset) < 1e-9:
            collapsed[-1] = (offset, bpm)
        else:
            collapsed.append((offset, bpm))

    if collapsed[0][0] > 0:
        collapsed.insert(0, (0.0, collapsed[0][1]))
    return collapsed


def bpm_at_quarter(tempo_map: list[tuple[float, float]], quarter_offset: float) -> float:
    offsets = [point[0] for point in tempo_map]
    index = bisect.bisect_right(offsets, quarter_offset) - 1
    if index < 0:
        index = 0
    return float(tempo_map[index][1])


def quarter_to_seconds(tempo_map: list[tuple[float, float]], quarter_offset: float) -> float:
    target = max(0.0, float(quarter_offset))
    seconds = 0.0

    for index, (start, bpm) in enumerate(tempo_map):
        if target <= start:
            break
        end = tempo_map[index + 1][0] if index + 1 < len(tempo_map) else target
        segment_end = min(target, end)
        if segment_end > start:
            seconds += (segment_end - start) * 60.0 / bpm
        if target <= end:
            break

    return seconds


def spanner_state(element: Any, class_name: str) -> tuple[str, str]:
    start = False
    stop = False
    matched = False
    numbers: list[str] = []
    try:
        sites = element.getSpannerSites()
    except Exception:
        sites = []

    for site in sites:
        if class_name not in getattr(site, "classes", []):
            continue
        matched = True
        number = getattr(site, "idLocal", None)
        if number not in (None, ""):
            numbers.append(str(number))
        try:
            start = start or bool(site.isFirst(element))
            stop = stop or bool(site.isLast(element))
        except Exception:
            pass

    if start and stop:
        state = "start_stop"
    elif start:
        state = "start"
    elif stop:
        state = "stop"
    elif matched:
        state = "continue"
    else:
        state = ""

    return state, "|".join(sorted(set(numbers)))


def tie_type(element: Any) -> str:
    tie = getattr(element, "tie", None)
    return str(getattr(tie, "type", "") or "").lower()


def articulation_names(element: Any) -> str:
    values = [obj.__class__.__name__ for obj in (getattr(element, "articulations", []) or [])]
    return "|".join(values)


def expression_names(element: Any) -> str:
    values = [obj.__class__.__name__ for obj in (getattr(element, "expressions", []) or [])]
    return "|".join(values)


def span_state_cn(state: str) -> str:
    return {
        "start": "起点",
        "continue": "中",
        "stop": "结束",
        "start_stop": "起止点",
    }.get(str(state), "")


def dynamic_change_label(crescendo_state: str, diminuendo_state: str) -> str:
    values: list[str] = []
    if crescendo_state:
        values.append(f"渐强{span_state_cn(crescendo_state)}")
    if diminuendo_state:
        values.append(f"渐弱{span_state_cn(diminuendo_state)}")
    return "|".join(values)








def build_measure_ranges(part: m21.stream.Part) -> list[tuple[float, float, str]]:
    ranges: list[tuple[float, float, str]] = []
    measures = list(part.getElementsByClass(m21.stream.Measure))
    for measure in measures:
        start = element_offset_in_part(measure, part)
        end = start + float(measure.barDuration.quarterLength)
        ranges.append((start, end, clean_token(getattr(measure, "number", None), "0")))
    return ranges


def measure_number_at_offset(
    measure_ranges: list[tuple[float, float, str]],
    offset: float,
) -> str:
    for start, end, number in measure_ranges:
        if start - 1e-7 <= offset < end - 1e-7:
            return number
    if measure_ranges and abs(offset - measure_ranges[-1][1]) < 1e-7:
        return measure_ranges[-1][2]
    return ""


def collect_part_directions(
    part: m21.stream.Part,
    tempo_map: list[tuple[float, float]],
    part_id: str,
) -> tuple[pd.DataFrame, list[tuple[float, str]], list[tuple[float, str]]]:
    rows: list[dict[str, Any]] = []
    words: list[tuple[float, str]] = []
    dynamics: list[tuple[float, str]] = []
    direction_index = 0
    measure_ranges = build_measure_ranges(part)

    def append_row(offset: float, direction_type: str, value: str, number: str = "") -> None:
        nonlocal direction_index
        direction_index += 1
        measure = measure_number_at_offset(measure_ranges, offset)
        rows.append(
            {
                "direction_id": f"{part_id}_D{direction_index:04d}",
                "part_id": part_id,
                "offset_quarter": round(offset, 6),
                "offset_sec": round(quarter_to_seconds(tempo_map, offset), 6),
                "measure_number": measure,
                "direction_type": direction_type,
                "value": value,
                "number": number,
            }
        )

    for expr in part.recurse().getElementsByClass(m21.expressions.TextExpression):
        offset = element_offset_in_part(expr, part)
        content = str(getattr(expr, "content", "") or "").strip()
        if content:
            words.append((offset, content))
            append_row(offset, "words", content)

    for dyn in part.recurse().getElementsByClass(m21.dynamics.Dynamic):
        offset = element_offset_in_part(dyn, part)
        value = str(getattr(dyn, "value", "") or "").strip()
        if value:
            dynamics.append((offset, value))
            append_row(offset, "dynamic", value)

    for mark in part.recurse().getElementsByClass(m21.tempo.MetronomeMark):
        offset = element_offset_in_part(mark, part)
        bpm = safe_float(mark.number)
        if math.isfinite(bpm) and bpm > 0:
            append_row(offset, "tempo", f"{bpm:g}")

    words.sort(key=lambda item: item[0])
    dynamics.sort(key=lambda item: item[0])
    return pd.DataFrame(rows), words, dynamics


def values_at_exact_offset(events: list[tuple[float, str]], offset: float) -> str:
    return "|".join(
        value for event_offset, value in events
        if abs(event_offset - offset) < 1e-7
    )


def latest_value(events: list[tuple[float, str]], offset: float) -> str:
    result = ""
    for event_offset, value in events:
        if event_offset <= offset + 1e-7:
            result = value
        else:
            break
    return result


def make_note_rows(
    score: m21.stream.Score,
    tempo_map: list[tuple[float, float]],
    wedge_intervals_by_part: dict[int, list[dict[str, Any]]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, str]]:
    detailed_rows: list[dict[str, Any]] = []
    all_direction_frames: list[pd.DataFrame] = []
    object_to_note_id: dict[int, str] = {}

    for part_index, part in enumerate(score.parts, start=1):
        part_id = f"P{part_index:02d}"
        part_source_id = clean_token(getattr(part, "id", None), part_id)
        directions_df, words, dynamic_events = collect_part_directions(part, tempo_map, part_id)
        all_direction_frames.append(directions_df)
        part_wedge_intervals = wedge_intervals_by_part.get(part_index, [])

        # At a barline, a grace note at the end of the previous measure can
        # share the same absolute offset as the first event of the next
        # measure.  `note_index_in_measure` is local to each measure, so it is
        # not a valid cross-measure tie-breaker.  Preserve the MusicXML measure
        # sequence explicitly and use it before the per-measure note index.
        part_measures = list(part.getElementsByClass(m21.stream.Measure))
        measure_sequence_by_id = {
            id(measure): sequence
            for sequence, measure in enumerate(part_measures)
        }
        measure_sequence_by_number: dict[str, int] = {}
        for sequence, measure in enumerate(part_measures):
            measure_sequence_by_number.setdefault(
                clean_token(getattr(measure, "number", None), "0"), sequence
            )

        counters: defaultdict[tuple[str, str], int] = defaultdict(int)
        elements = list(part.flatten().notesAndRests)

        for element in elements:
            if isinstance(element, m21.note.Rest):
                continue

            measure_number = get_measure_number(element)
            context_measure = element.getContextByClass(m21.stream.Measure)
            measure_number_token = clean_token(measure_number, "0")
            context_number_token = clean_token(
                getattr(context_measure, "number", None), "0"
            )
            # music21 can resolve a previous-measure grace note exactly on a
            # barline to the following measure's context.  The note's own
            # MusicXML measure identity is authoritative in that case.
            measure_sequence = measure_sequence_by_number.get(measure_number_token)
            if measure_sequence is None and context_number_token == measure_number_token:
                measure_sequence = measure_sequence_by_id.get(id(context_measure))
            if measure_sequence is None:
                measure_sequence = len(part_measures)
            voice_id = get_voice_id(element)
            counter_key = (measure_number, voice_id)
            counters[counter_key] += 1
            note_index = counters[counter_key]

            measure_token = clean_token(measure_number, "0")
            measure_id_token = (
                f"{int(measure_token):03d}" if measure_token.isdigit() else measure_token
            )
            event_id = (
                f"P{part_index:02d}_M{measure_id_token}"
                f"_V{clean_token(voice_id, '1')}_N{note_index:04d}"
            )

            onset_quarter = element_offset_in_part(element, part)
            duration_quarter = float(element.quarterLength)
            is_grace = bool(getattr(getattr(element, "duration", None), "isGrace", False))
            if is_grace:
                duration_quarter = 0.0
            end_quarter = onset_quarter + duration_quarter

            onset_sec = quarter_to_seconds(tempo_map, onset_quarter)
            end_sec = quarter_to_seconds(tempo_map, end_quarter)
            duration_sec = max(0.0, end_sec - onset_sec)
            bpm = bpm_at_quarter(tempo_map, onset_quarter)

            slur_state, slur_number = spanner_state(element, "Slur")
            glissando_state, glissando_number = spanner_state(element, "Glissando")
            tie = tie_type(element)

            words_at_onset = values_at_exact_offset(words, onset_quarter)
            dynamic_at_onset = values_at_exact_offset(dynamic_events, onset_quarter)
            active_dynamic = latest_value(dynamic_events, onset_quarter)
            dynamic_change = dynamic_change_at_offset(part_wedge_intervals, onset_quarter)

            pitches = list(element.pitches) if isinstance(element, m21.chord.Chord) else [element.pitch]
            is_chord = len(pitches) > 1
            first_score_note_id = ""

            for tone_index, pitch in enumerate(pitches, start=1):
                score_note_id = event_id if not is_chord else f"{event_id}_C{tone_index:02d}"
                if not first_score_note_id:
                    first_score_note_id = score_note_id

                detailed_rows.append(
                    {
                        "score_note_id": score_note_id,
                        "event_id": event_id,
                        "part_id": part_id,
                        "part_source_id": part_source_id,
                        "part_index": part_index,
                        "_measure_sequence": measure_sequence,
                        "measure_number": measure_number,
                        "voice": voice_id,
                        "note_index_in_measure": note_index,
                        "chord_tone_index": tone_index,
                        "is_chord": is_chord,
                        "onset_quarter": round(onset_quarter, 6),
                        "duration_quarter": round(duration_quarter, 6),
                        "end_quarter": round(end_quarter, 6),
                        "onset_sec": round(onset_sec, 6),
                        "duration_sec": round(duration_sec, 6),
                        "end_sec": round(end_sec, 6),
                        "pitch_name": pitch.nameWithOctave,
                        "pitch_midi": int(pitch.midi),
                        "pitch_hz": round(float(pitch.frequency), 6),
                        "bpm": round(bpm, 6),
                        "is_grace": is_grace,
                        "grace_type": "grace" if is_grace else "",
                        "tie_type": tie,
                        "slur_state": slur_state,
                        "slur_number": slur_number,
                        "glissando_state": glissando_state,
                        "glissando_number": glissando_number,
                        "articulations": articulation_names(element),
                        "expressions": expression_names(element),
                        "direction_words_at_onset": words_at_onset,
                        "dynamic_at_onset": dynamic_at_onset,
                        "active_dynamic": active_dynamic,
                        "dynamic_change": dynamic_change,
                        "time_signature": get_time_signature(element),
                        "key_fifths": get_key_fifths(element),
                    }
                )

            object_to_note_id[id(element)] = first_score_note_id

    notes_df = pd.DataFrame(detailed_rows)
    if not notes_df.empty:
        notes_df = notes_df.sort_values(
            [
                "onset_quarter",
                "part_index",
                "_measure_sequence",
                "voice",
                "note_index_in_measure",
                "chord_tone_index",
            ],
            kind="stable",
        ).reset_index(drop=True)
        notes_df = notes_df.drop(columns=["_measure_sequence"])

    notes_df = add_alignment_event_fields(notes_df)

    directions_df = (
        pd.concat(all_direction_frames, ignore_index=True)
        if all_direction_frames else pd.DataFrame()
    )
    if not directions_df.empty:
        directions_df = directions_df.sort_values(
            ["offset_quarter", "part_id", "direction_id"],
            kind="stable",
        ).reset_index(drop=True)

    return notes_df, directions_df, object_to_note_id


def add_alignment_event_fields(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the bridge between raw MusicXML notes and performed onset events.

    Every MusicXML note remains in the table, but tie continuation/stop notes do
    not request a new acoustic onset. The first note of a tie chain carries the
    chain's total logical duration.
    """
    if notes_df.empty:
        return notes_df.copy()

    df = notes_df.copy().reset_index(drop=True)
    active_ties: dict[tuple[str, str, str], str] = {}
    group_rows: defaultdict[str, list[int]] = defaultdict(list)
    anchor_by_group: dict[str, int] = {}

    flags: list[str] = []
    groups: list[str] = []
    anchors: list[str] = []

    for idx, row in df.iterrows():
        note_id = str(row["score_note_id"])
        key = (
            str(row.get("part_id", "")),
            str(row.get("voice", "")),
            str(row.get("pitch_name", "")),
        )
        tie = str(row.get("tie_type", "") or "").strip().lower()

        if tie == "start":
            group_id = f"TIE_{note_id}"
            active_ties[key] = group_id
            participate = "是"
            anchor_by_group[group_id] = idx
        elif tie in {"continue", "stop"} and key in active_ties:
            group_id = active_ties[key]
            participate = "否"
            if tie == "stop":
                active_ties.pop(key, None)
        else:
            # Malformed orphan continue/stop is kept alignable instead of being
            # silently discarded. Ordinary notes each form their own group.
            group_id = note_id
            participate = "是"
            anchor_by_group[group_id] = idx

        group_rows[group_id].append(idx)
        flags.append(participate)
        groups.append(group_id)
        anchor_idx = anchor_by_group.get(group_id, idx)
        anchors.append(str(df.at[anchor_idx, "score_note_id"]))

    df["参与起音对齐"] = flags
    df["对齐组ID"] = groups
    df["对齐锚点ID"] = anchors
    df["逻辑持续时间(s)"] = 0.0
    df["逻辑持续时间(拍)"] = 0.0
    df["逻辑结束时间(s)"] = df["onset_sec"].astype(float)

    for group_id, indices in group_rows.items():
        anchor_idx = anchor_by_group[group_id]
        start_sec = float(df.at[anchor_idx, "onset_sec"])
        end_sec = max(float(df.at[i, "end_sec"]) for i in indices)
        start_q = float(df.at[anchor_idx, "onset_quarter"])
        end_q = max(float(df.at[i, "end_quarter"]) for i in indices)
        df.at[anchor_idx, "逻辑持续时间(s)"] = max(0.0, end_sec - start_sec)
        df.at[anchor_idx, "逻辑持续时间(拍)"] = max(0.0, end_q - start_q)
        df.at[anchor_idx, "逻辑结束时间(s)"] = end_sec
        for i in indices:
            df.at[i, "对齐锚点ID"] = str(df.at[anchor_idx, "score_note_id"])
            if i != anchor_idx:
                df.at[i, "逻辑结束时间(s)"] = end_sec

    return df


def build_relations(
    score: m21.stream.Score,
    object_to_note_id: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    relation_index = 0

    def append_relation(
        relation_type: str,
        from_id: str,
        to_id: str,
        number: str = "",
        source: str = "musicxml",
    ) -> None:
        nonlocal relation_index
        if not from_id or not to_id:
            return
        relation_index += 1
        rows.append(
            {
                "relation_id": f"R{relation_index:05d}",
                "relation_type": relation_type,
                "from_score_note_id": from_id,
                "to_score_note_id": to_id,
                "number": number,
                "source": source,
            }
        )

    for sp in score.spannerBundle:
        classes = set(getattr(sp, "classes", []))
        if "Slur" in classes:
            relation_type = "slur"
        elif "Glissando" in classes:
            relation_type = "glissando"
        elif "Crescendo" in classes:
            relation_type = "crescendo"
        elif "Diminuendo" in classes:
            relation_type = "diminuendo"
        else:
            continue

        elements = list(sp.getSpannedElements())
        if not elements:
            continue
        from_id = object_to_note_id.get(id(elements[0]), "")
        to_id = object_to_note_id.get(id(elements[-1]), "")
        if relation_type in {"crescendo", "diminuendo"} and from_id == to_id:
            continue
        append_relation(
            relation_type,
            from_id,
            to_id,
            str(getattr(sp, "idLocal", "") or ""),
        )

    # Tie relations are not spanners in music21. Pair start/continue/stop by part, voice and pitch.
    for part in score.parts:
        active: dict[tuple[str, str], str] = {}
        for element in part.flatten().notes:
            if isinstance(element, m21.chord.Chord):
                continue
            note_id = object_to_note_id.get(id(element), "")
            if not note_id:
                continue
            key = (get_voice_id(element), element.pitch.nameWithOctave)
            tie = tie_type(element)
            if tie == "start":
                active[key] = note_id
            elif tie in {"continue", "stop"} and key in active:
                append_relation("tie", active[key], note_id)
                if tie == "continue":
                    active[key] = note_id
                else:
                    active.pop(key, None)

    return pd.DataFrame(rows)


def raw_legato_state(row: pd.Series) -> str:
    starts = row.get("slur_state") in {"start", "start_stop"} or row.get("tie_type") == "start"
    stops = row.get("slur_state") in {"stop", "start_stop"} or row.get("tie_type") == "stop"
    continues = row.get("tie_type") == "continue"

    if starts and stops:
        return "分离"
    if starts:
        return "起点"
    if stops:
        return "结束"
    if continues:
        return "中"
    return "无"


def apply_legato_state_machine(notes_df: pd.DataFrame) -> pd.DataFrame:
    df = notes_df.copy()
    raw_states = [raw_legato_state(row) for _, row in df.iterrows()]
    tags: list[str] = []
    in_legato = False

    for raw in raw_states:
        if raw == "分离":
            tags.append("连奏分离点")
            in_legato = True
        elif raw == "起点":
            tags.append("连奏起点")
            in_legato = True
        elif raw == "结束":
            tags.append("连奏结束" if in_legato else "无")
            in_legato = False
        elif raw == "中":
            tags.append("连奏中")
            in_legato = True
        else:
            tags.append("连奏中" if in_legato else "无")

    df["连奏信息"] = tags
    return df






def make_legacy_table(notes_df: pd.DataFrame, merge_same_pitch: bool) -> pd.DataFrame:
    df = apply_legato_state_machine(notes_df)

    if merge_same_pitch:
        raise ValueError("Only the frozen non-merging score representation is included.")

    legacy = pd.DataFrame(
        {
            "累计时间(s)": df["onset_sec"].round(3),
            "音高频率(Hz)": df["pitch_hz"].round(2),
            "持续时间(s)": df["duration_sec"].round(3),
            "音名": df["pitch_name"],
            "累计时长(s)": df["end_sec"].round(3),
            "连奏信息": df["连奏信息"],
            "BPM": df["bpm"].round(3),
            "装饰音判定": df["is_grace"].map({True: "装饰音", False: "主音"}),
            "score_note_id": df["score_note_id"],
            "参与起音对齐": df["参与起音对齐"],
            "对齐组ID": df["对齐组ID"],
            "对齐锚点ID": df["对齐锚点ID"],
            "逻辑持续时间(s)": df["逻辑持续时间(s)"].round(6),
            "逻辑持续时间(拍)": df["逻辑持续时间(拍)"].round(6),
            "逻辑结束时间(s)": df["逻辑结束时间(s)"].round(6),
            "小节号": df["measure_number"],
            "声部": df["voice"],
            "谱面起点(拍)": df["onset_quarter"],
            "谱面时值(拍)": df["duration_quarter"],
            "强弱记号": df["dynamic_at_onset"],
            "当前强弱": df["active_dynamic"],
            "强弱变化": df["dynamic_change"],
            "延音线状态": df["tie_type"].map({
                "start": "起点", "continue": "中", "stop": "结束"
            }).fillna(""),
            "连音线状态": df["slur_state"].map(span_state_cn),
            "连音线编号": df["slur_number"],
            "滑音线状态": df["glissando_state"].map(span_state_cn),
            "滑音线编号": df["glissando_number"],
            "谱面文字": df["direction_words_at_onset"],
            "拍号": df["time_signature"],
            "调号升降数": df["key_fifths"],
        }
    )
    return legacy


def validate_outputs(
    notes_df: pd.DataFrame,
    legacy_df: pd.DataFrame,
    relations_df: pd.DataFrame,
) -> None:
    if notes_df.empty:
        raise ValueError("No pitched notes were parsed from the score.")
    if notes_df["score_note_id"].duplicated().any():
        duplicates = notes_df.loc[
            notes_df["score_note_id"].duplicated(), "score_note_id"
        ].tolist()
        raise ValueError(f"Duplicate score_note_id values: {duplicates[:10]}")
    missing = [column for column in LEGACY_COLUMNS if column not in legacy_df.columns]
    if missing:
        raise ValueError(f"Legacy table missing required columns: {missing}")
    if not legacy_df["累计时间(s)"].is_monotonic_increasing:
        raise ValueError("Legacy onset times are not monotonic.")
    if not relations_df.empty:
        valid_ids = set(notes_df["score_note_id"])
        bad = relations_df[
            ~relations_df["from_score_note_id"].isin(valid_ids)
            | ~relations_df["to_score_note_id"].isin(valid_ids)
        ]
        if not bad.empty:
            raise ValueError("Relations contain note IDs absent from score_notes.csv.")



def add_relation_columns(
    table_df: pd.DataFrame,
    relations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fold the relation table back into each score row so the parser only needs
    to output one CSV.

    For merged legacy rows, every ID in merged_score_note_ids participates.
    """
    df = table_df.copy()

    if relations_df.empty:
        df["关系_内部"] = ""
        df["关系_指向"] = ""
        df["关系_来自"] = ""
        return df

    relation_records = relations_df.to_dict("records")
    internal_values: list[str] = []
    outgoing_values: list[str] = []
    incoming_values: list[str] = []

    for _, row in df.iterrows():
        raw_ids = str(row.get("merged_score_note_ids", row.get("score_note_id", "")))
        member_ids = {item for item in raw_ids.split("|") if item and item != "nan"}

        internal: list[str] = []
        outgoing: list[str] = []
        incoming: list[str] = []

        for relation in relation_records:
            relation_type = str(relation.get("relation_type", ""))
            from_id = str(relation.get("from_score_note_id", ""))
            to_id = str(relation.get("to_score_note_id", ""))
            relation_id = str(relation.get("relation_id", ""))

            from_inside = from_id in member_ids
            to_inside = to_id in member_ids

            if from_inside and to_inside:
                internal.append(
                    f"{relation_id}:{relation_type}:{from_id}->{to_id}"
                )
            elif from_inside:
                outgoing.append(
                    f"{relation_id}:{relation_type}->{to_id}"
                )
            elif to_inside:
                incoming.append(
                    f"{relation_id}:{relation_type}<-{from_id}"
                )

        internal_values.append("|".join(internal))
        outgoing_values.append("|".join(outgoing))
        incoming_values.append("|".join(incoming))

    df["关系_内部"] = internal_values
    df["关系_指向"] = outgoing_values
    df["关系_来自"] = incoming_values
    return df


def write_outputs(
    input_path: Path,
    output_dir: Path,
    table_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    directions_df: pd.DataFrame,
) -> dict[str, Path]:
    """
    Write files according to the True / False switches at the top of this file.

    Default behavior:
        only *_score_final.csv is created.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    outputs: dict[str, Path] = {}

    if OUTPUT_SCORE_FINAL:
        path = output_dir / f"{stem}_score_final.csv"
        table_df.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["score_final"] = path

    if OUTPUT_SCORE_NOTES:
        path = output_dir / f"{stem}_score_notes.csv"
        notes_df.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["score_notes"] = path

    if OUTPUT_SCORE_RELATIONS:
        path = output_dir / f"{stem}_score_relations.csv"
        relations_df.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["score_relations"] = path

    if OUTPUT_SCORE_DIRECTIONS:
        path = output_dir / f"{stem}_score_directions.csv"
        directions_df.to_csv(path, index=False, encoding="utf-8-sig")
        outputs["score_directions"] = path

    if OUTPUT_SCORE_META:
        path = output_dir / f"{stem}_score_meta.json"
        meta = {
            "input_file": input_path.name,
            "score_final_rows": int(len(table_df)),
            "original_note_rows": int(len(notes_df)),
            "relation_rows": int(len(relations_df)),
            "direction_rows": int(len(directions_df)),
            "enabled_outputs": {
                "score_final": OUTPUT_SCORE_FINAL,
                "score_notes": OUTPUT_SCORE_NOTES,
                "score_relations": OUTPUT_SCORE_RELATIONS,
                "score_directions": OUTPUT_SCORE_DIRECTIONS,
                "score_meta": OUTPUT_SCORE_META,
            },
        }
        path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs["score_meta"] = path

    if not outputs:
        raise RuntimeError(
            "所有输出开关都是 False，没有文件会被生成。"
            "请至少把 OUTPUT_SCORE_FINAL 改为 True。"
        )

    return outputs

def parse_musicxml(
    input_path: Path,
    output_dir: Path,
    merge_same_pitch: bool = LEGACY_MERGE_SAME_PITCH_LEGATO,
) -> Path:
    print(f"Reading MusicXML: {input_path}")
    score = m21.converter.parse(str(input_path))
    tempo_map = build_tempo_map(score)
    wedge_intervals_by_part = collect_wedge_intervals(input_path, score)

    notes_df, directions_df, object_to_note_id = make_note_rows(
        score, tempo_map, wedge_intervals_by_part
    )
    relations_df = build_relations(score, object_to_note_id)
    table_df = make_legacy_table(notes_df, merge_same_pitch=merge_same_pitch)

    validate_outputs(notes_df, table_df, relations_df)
    outputs = write_outputs(
        input_path=input_path,
        output_dir=output_dir,
        table_df=table_df,
        notes_df=notes_df,
        relations_df=relations_df,
        directions_df=directions_df,
    )

    print(f"Parsed original note rows: {len(notes_df)}")
    print(f"Final table rows: {len(table_df)}")
    print(f"Validated note relations: {len(relations_df)}")
    print("Generated files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")

    return outputs.get("score_final", next(iter(outputs.values())))

def read_musicxml_to_score_final_csv() -> None:
    """Compatibility entry point used by older runners."""
    base_dir = Path(
        os.environ.get("ERHU_WORK_DIR", str(Path(__file__).resolve().parent))
    ).resolve()
    input_path = choose_input_file(base_dir)
    parse_musicxml(input_path, base_dir)


def main() -> None:
    """
    Standalone mode:
    1. Place this script beside exactly one .musicxml/.xml file.
    2. Run: python mxml.py
    3. All outputs are written beside the input score.
    """
    read_musicxml_to_score_final_csv()


if __name__ == "__main__":
    main()

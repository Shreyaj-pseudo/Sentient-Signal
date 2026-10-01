"""Day 1 command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from sentient_signals.asr import WHISPER_TINY_EN_PARAMETERS, word_error_rate
from sentient_signals.audio import decode_audio, write_pcm16_wav
from sentient_signals.data import (
    load_manifest,
    read_split,
    representative_subset,
    write_manifest,
)
from sentient_signals.features import FeatureExtractor
from sentient_signals.model import InferencePipeline, fit_head


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare_main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare official MELD manifests and mono 16 kHz WAVs"
    )
    parser.add_argument("--raw-root", type=Path, default=Path("MELD.Raw"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "dev", "test"), default=("train", "dev", "test")
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report: dict[str, object] = {"splits": {}, "failures": []}
    for split in args.splits:
        items, missing = read_split(args.raw_root, split, args.output_root)
        selected = representative_subset(items, args.limit) if args.limit is not None else items
        prepared = []
        for item in selected:
            try:
                samples = decode_audio(Path(item.source_path))
                write_pcm16_wav(Path(item.audio_path), samples)
                prepared.append(replace(item, duration_seconds=len(samples) / 16_000))
            except Exception as error:  # report individual corrupt media without losing the run
                report["failures"].append({"clip": item.clip_id, "error": str(error)})
        write_manifest(args.output_root / f"{split}.jsonl", prepared)
        report["splits"][split] = {
            "csv_rows": len(items) + len(missing),
            "prepared": len(prepared),
            "missing": missing,
        }
    _write_json(Path("reports/data_preparation.json"), report)
    print(json.dumps(report, indent=2))


def cache_main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen text/audio representations")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-mode", choices=("encoders", "feasibility"), default="encoders")
    parser.add_argument(
        "--reuse-audio-from",
        type=Path,
        help="Reuse matching cached audio vectors while rebuilding text features",
    )
    args = parser.parse_args()
    items = load_manifest(args.manifest)
    if args.limit:
        items = representative_subset(items, args.limit, args.seed)
    extractor = FeatureExtractor(args.feature_mode)
    reusable_audio = {}
    if args.reuse_audio_from:
        with np.load(args.reuse_audio_from, allow_pickle=False) as cached:
            if str(cached["feature_mode"].item()) != args.feature_mode:
                raise ValueError("Reusable audio cache has a different feature mode")
            reusable_audio = {
                str(clip_id): feature.copy()
                for clip_id, feature in zip(cached["clip_ids"], cached["audio"], strict=True)
            }
    text_features, audio_features, labels, clip_ids = [], [], [], []
    for index, item in enumerate(items, 1):
        if item.clip_id in reusable_audio:
            text_feature = extractor.encode_text(item.text)
            audio_feature = reusable_audio[item.clip_id]
        else:
            text_feature, audio_feature, _ = extractor.encode(item.text, Path(item.audio_path))
        text_features.append(text_feature)
        audio_features.append(audio_feature)
        labels.append(item.label)
        clip_ids.append(item.clip_id)
        if index % 25 == 0:
            print(f"encoded {index}/{len(items)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        text=np.stack(text_features),
        audio=np.stack(audio_features),
        labels=np.asarray(labels),
        clip_ids=np.asarray(clip_ids),
        feature_mode=np.asarray(args.feature_mode),
    )


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train a small text, audio, or fusion head")
    parser.add_argument("--train-features", type=Path, required=True)
    parser.add_argument("--dev-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/day1_subset.pt"))
    parser.add_argument("--modality", choices=("text", "audio", "fusion"), default="fusion")
    parser.add_argument("--class-weighting", choices=("none", "inverse_sqrt"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Use all development examples; intended only for legacy feasibility reproduction",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            fit_head(
                args.train_features,
                args.dev_features,
                args.output,
                epochs=args.epochs,
                seed=args.seed,
                modality=args.modality,
                class_weighting=args.class_weighting,
                reserve_calibration=not args.include_calibration,
            ),
            indent=2,
        )
    )


def predict_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a prepared WAV through local ASR and emotion inference"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--text",
        help="Optional supplied transcript; omit it to transcribe the WAV locally",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prepared WAV files; models stay loaded between files",
    )
    parser.add_argument("--asr-model", default="tiny.en")
    parser.add_argument("--threads", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.text is not None and len(args.audio) != 1:
        parser.error("--text can only be used with one --audio file")
    pipeline = InferencePipeline(
        args.checkpoint,
        asr_model=args.asr_model,
        cpu_threads=args.threads,
        local_files_only=args.offline,
    )
    for audio_path in args.audio:
        print(pipeline.predict(audio_path, text=args.text).model_dump_json(indent=2))


def benchmark_main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark representative prepared clips")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--threads", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--feature-mode", choices=("encoders", "feasibility"), default="encoders")
    parser.add_argument("--output", type=Path, default=Path("reports/day1_benchmark.json"))
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import psutil
    import torch

    torch.set_num_threads(args.threads)
    items = representative_subset(load_manifest(args.manifest), args.limit)
    extractor = FeatureExtractor(args.feature_mode)
    process = psutil.Process()
    cold_started = time.perf_counter()
    extractor.encode(items[0].text, Path(items[0].audio_path))
    cold_start_ms = (time.perf_counter() - cold_started) * 1000
    peak_rss = process.memory_info().rss
    elapsed = []
    started_all = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        extractor.encode(item.text, Path(item.audio_path))
        elapsed.append((time.perf_counter() - started) * 1000)
        peak_rss = max(peak_rss, process.memory_info().rss)
    input_dimension = extractor.text_dimension + extractor.audio_dimension
    parameter_counts = {
        "fusion_head": input_dimension * 128 + 128 + 128 * 7 + 7,
    }
    if args.feature_mode == "encoders":
        parameter_counts["text_encoder"] = sum(
            parameter.numel() for parameter in extractor._text_model.parameters()
        )
        parameter_counts["audio_encoder"] = sum(
            parameter.numel() for parameter in extractor._audio_model.parameters()
        )
    report = {
        "artifact_kind": f"day1_{args.feature_mode}_benchmark",
        "feature_mode": args.feature_mode,
        "clips": len(elapsed),
        "threads": args.threads,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "wall_seconds": time.perf_counter() - started_all,
        "cold_start_ms": cold_start_ms,
        "peak_process_rss_bytes": peak_rss,
        "learned_parameters": parameter_counts,
        "latency_ms": {
            "mean": statistics.fmean(elapsed),
            "p50": float(np.percentile(elapsed, 50)),
            "p95": float(np.percentile(elapsed, 95)),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2))


def e2e_benchmark_main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark warm local ASR-to-response inference on real prepared clips"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--threads", type=int, choices=(2, 4, 8), default=8)
    parser.add_argument("--asr-model", default="tiny.en")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/day1_e2e_benchmark.json"))
    args = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    import psutil
    import torch

    torch.set_num_threads(args.threads)
    items = representative_subset(load_manifest(args.manifest), args.limit)
    if not items:
        raise ValueError("The benchmark manifest contains no examples")
    process = psutil.Process()
    pipeline = InferencePipeline(
        args.checkpoint,
        asr_model=args.asr_model,
        cpu_threads=args.threads,
        local_files_only=args.offline,
    )
    cold_started = time.perf_counter()
    pipeline.predict(Path(items[0].audio_path))
    cold_start_ms = (time.perf_counter() - cold_started) * 1000
    peak_rss = process.memory_info().rss
    total_times = []
    transcription_times = []
    feature_times = []
    error_rates = []
    examples = []
    for item in items:
        result = pipeline.predict(Path(item.audio_path))
        total_times.append(result.timings.total_ms)
        transcription_times.append(result.timings.transcription_ms)
        feature_times.append(result.timings.audio_ms + result.timings.text_ms)
        error_rates.append(word_error_rate(item.text, result.transcript))
        peak_rss = max(peak_rss, process.memory_info().rss)
        if len(examples) < 5:
            examples.append(
                {
                    "clip_id": item.clip_id,
                    "reference": item.text,
                    "transcript": result.transcript,
                    "emotion": result.emotion,
                    "uncertain": result.uncertain,
                }
            )
    parameter_counts = {
        "asr": WHISPER_TINY_EN_PARAMETERS,
        "text_encoder": sum(
            parameter.numel() for parameter in pipeline.extractor._text_model.parameters()
        ),
        "audio_encoder": sum(
            parameter.numel() for parameter in pipeline.extractor._audio_model.parameters()
        ),
        "fusion_head": sum(parameter.numel() for parameter in pipeline.model.parameters()),
    }
    report = {
        "artifact_kind": "day1_end_to_end_benchmark",
        "clips": len(items),
        "threads": args.threads,
        "asr_model": args.asr_model,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cold_start_and_first_result_ms": cold_start_ms,
        "peak_process_rss_bytes": peak_rss,
        "learned_parameters": parameter_counts,
        "total_learned_parameters": sum(parameter_counts.values()),
        "word_error_rate_mean": statistics.fmean(error_rates),
        "latency_ms": {
            "total_mean": statistics.fmean(total_times),
            "total_p50": float(np.percentile(total_times, 50)),
            "total_p95": float(np.percentile(total_times, 95)),
            "transcription_mean": statistics.fmean(transcription_times),
            "features_mean": statistics.fmean(feature_times),
        },
        "examples": examples,
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2))


def serve_main() -> None:
    parser = argparse.ArgumentParser(description="Serve the local microphone and replay interface")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/day1_encoder_subset.pt")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["SENTIENT_CHECKPOINT"] = str(args.checkpoint)
    import uvicorn

    from sentient_signals.service import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port)

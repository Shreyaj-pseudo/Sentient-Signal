"""Small fusion head, training, serialization, and runtime prediction."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from sentient_signals.calibration import dialogue_calibration_mask
from sentient_signals.constants import EMOTIONS
from sentient_signals.features import FeatureExtractor
from sentient_signals.policy import PolicyDecision, choose_response
from sentient_signals.schema import PredictionResult, Timings


class FusionHead(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dimension, len(EMOTIONS)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class AttentiveAudioHead(nn.Module):
    """Learn which temporal audio bins carry emotion before classification."""

    def __init__(self, frame_dimension: int) -> None:
        super().__init__()
        self.attention = nn.Linear(frame_dimension, 1)
        self.classifier = FusionHead(frame_dimension * 2)

    def forward(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.attention(frames).squeeze(-1).masked_fill(~mask, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        mean = torch.sum(frames * weights.unsqueeze(-1), dim=1)
        variance = torch.sum((frames - mean.unsqueeze(1)).square() * weights.unsqueeze(-1), dim=1)
        return self.classifier(torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=1))


class GatedFusionHead(nn.Module):
    """Let text lead while learning when audio should change the prediction."""

    def __init__(
        self, text_dimension: int, audio_dimension: int, hidden_dimension: int = 128
    ) -> None:
        super().__init__()
        self.text_hidden = nn.Sequential(
            nn.Linear(text_dimension, hidden_dimension), nn.ReLU(), nn.Dropout(0.2)
        )
        self.audio_hidden = nn.Sequential(
            nn.Linear(audio_dimension, hidden_dimension), nn.ReLU(), nn.Dropout(0.2)
        )
        self.text_output = nn.Linear(hidden_dimension, len(EMOTIONS))
        self.audio_output = nn.Linear(hidden_dimension, len(EMOTIONS))
        self.gate = nn.Linear(hidden_dimension * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, np.log(4))  # Start at 80% text, 20% audio.

    def forward(
        self, text: torch.Tensor, audio: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        text_hidden = self.text_hidden(text)
        audio_hidden = self.audio_hidden(audio)
        text_logits = self.text_output(text_hidden)
        audio_logits = self.audio_output(audio_hidden)
        text_weight = torch.sigmoid(self.gate(torch.cat([text_hidden, audio_hidden], dim=1)))
        probabilities = text_weight * torch.softmax(text_logits, dim=1) + (
            1 - text_weight
        ) * torch.softmax(audio_logits, dim=1)
        return probabilities.clamp_min(1e-12).log(), text_logits, audio_logits, text_weight


def fit_head(
    train_path: Path,
    dev_path: Path,
    output_path: Path,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
    modality: str = "fusion",
    class_weighting: str = "none",
    reserve_calibration: bool = True,
) -> dict[str, object]:
    if modality not in {"text", "audio", "fusion"}:
        raise ValueError("modality must be text, audio, or fusion")
    if class_weighting not in {"none", "inverse_sqrt"}:
        raise ValueError("class_weighting must be none or inverse_sqrt")
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive")
    torch.manual_seed(seed)
    train = np.load(train_path, allow_pickle=False)
    dev = np.load(dev_path, allow_pickle=False)
    mode = str(train["feature_mode"].item())
    if mode != str(dev["feature_mode"].item()):
        raise ValueError("Train and development feature modes differ")
    train_x = _feature_matrix(train, modality)
    dev_x = _feature_matrix(dev, modality)
    dev_y = dev["labels"].astype(np.int64)
    calibration_clip_ids = np.asarray([], dtype=str)
    if reserve_calibration:
        calibration_mask = dialogue_calibration_mask(dev["clip_ids"])
        calibration_clip_ids = dev["clip_ids"][calibration_mask].astype(str)
        dev_x = dev_x[~calibration_mask]
        dev_y = dev_y[~calibration_mask]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - mean) / scale
    dev_x = (dev_x - mean) / scale
    train_y = train["labels"].astype(np.int64)
    model = FusionHead(train_x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    weights = None
    if class_weighting == "inverse_sqrt":
        counts = np.bincount(train_y, minlength=len(EMOTIONS)).astype(np.float32)
        weights = torch.from_numpy(1 / np.sqrt(np.maximum(counts, 1)))
        weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weights)
    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(train_y)
    rng = np.random.default_rng(seed)
    best_score = -1.0
    best_state = None
    stale = 0
    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_y))
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_tensor[indices]), y_tensor[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            predicted = model(torch.from_numpy(dev_x)).argmax(dim=1).numpy()
        score = float(
            f1_score(
                dev_y,
                predicted,
                labels=list(range(len(EMOTIONS))),
                average="macro",
                zero_division=0,
            )
        )
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    artifact = {
        "artifact_kind": (
            f"day2_subset_{mode}_{modality}" if len(train_y) < 9989 else f"day2_full_{modality}"
        ),
        "feature_mode": mode,
        "modality": modality,
        "class_weighting": class_weighting,
        "text_dimension": int(train["text"].shape[1]),
        "audio_dimension": int(train["audio"].shape[1]),
        "normalization_mean": torch.from_numpy(mean),
        "normalization_scale": torch.from_numpy(scale),
        "model_state": best_state,
        "dev_macro_f1": best_score,
        "seed": seed,
        "train_examples": len(train_y),
        "dev_examples": len(dev_y),
        "calibration_clip_ids": calibration_clip_ids.tolist(),
    }
    if "feature_config" in train.files:
        artifact["feature_config"] = json.loads(str(train["feature_config"].item()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    return {
        key: value for key, value in artifact.items() if not isinstance(value, (dict, torch.Tensor))
    }


def _feature_matrix(features, modality: str) -> np.ndarray:
    if modality == "text":
        value = features["text"]
    elif modality == "audio":
        value = features["audio"]
    else:
        value = np.concatenate([features["text"], features["audio"]], axis=1)
    return value.astype(np.float32)


class InferencePipeline:
    """Keep all models resident so repeated turns avoid model reload time."""

    def __init__(
        self,
        checkpoint: Path,
        asr_model: str = "tiny.en",
        cpu_threads: int = 8,
        local_files_only: bool = False,
    ) -> None:
        self.artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.set_num_threads(cpu_threads)
        feature_config = self.artifact.get("feature_config", {})
        self.extractor = FeatureExtractor(
            str(self.artifact["feature_mode"]),
            local_files_only=local_files_only,
            audio_encoder=str(feature_config.get("audio_encoder", "speech")),
        )
        self.modality = str(self.artifact.get("modality", "fusion"))
        dimensions = {
            "text": int(self.artifact["text_dimension"]),
            "audio": int(self.artifact["audio_dimension"]),
            "fusion": int(self.artifact["text_dimension"]) + int(self.artifact["audio_dimension"]),
        }
        if self.modality == "probability":
            self.heads = {}
            for modality in ("text", "audio"):
                head = FusionHead(dimensions[modality])
                head.load_state_dict(self.artifact[f"{modality}_head"]["model_state"])
                self.heads[modality] = head.eval()
            self.model = nn.ModuleDict(self.heads)
        else:
            self.model = FusionHead(dimensions[self.modality])
            self.model.load_state_dict(self.artifact["model_state"])
            self.model.eval()
        self.asr_model = asr_model
        self.cpu_threads = cpu_threads
        self.local_files_only = local_files_only
        self._transcriber = None

    def _get_transcriber(self):
        if self._transcriber is None:
            from sentient_signals.asr import LocalTranscriber

            self._transcriber = LocalTranscriber(
                self.asr_model,
                cpu_threads=self.cpu_threads,
                local_files_only=self.local_files_only,
            )
        return self._transcriber

    def predict(self, audio_path: Path, text: str | None = None) -> PredictionResult:
        overall = time.perf_counter()
        transcription_ms = 0.0
        transcript_source = "supplied"
        if text is None:
            started = time.perf_counter()
            text = self._get_transcriber().transcribe(audio_path).text
            transcription_ms = (time.perf_counter() - started) * 1000
            transcript_source = "asr"
        text_feature, audio_feature, feature_times = self.extractor.encode(text, audio_path)
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = self.predict_features(text_feature, audio_feature)
        classifier_ms = (time.perf_counter() - started) * 1000
        order = np.argsort(probabilities)[::-1]
        uncertain = bool(
            probabilities[order[0]] < 0.55
            or probabilities[order[0]] - probabilities[order[1]] < 0.15
        )
        emotion = EMOTIONS[int(order[0])]
        if not text.strip():
            uncertain = True
            decision = PolicyDecision(
                "I couldn't hear any speech. Please try again.",
                "retry_empty_transcript_v1",
                "error",
            )
        else:
            decision = choose_response(emotion, uncertain, text)
        return PredictionResult(
            turn_id=str(uuid.uuid4()),
            transcript=text,
            transcript_source=transcript_source,
            emotion=emotion,
            probabilities={label: float(probabilities[i]) for i, label in enumerate(EMOTIONS)},
            uncertain=uncertain,
            response_text=decision.response_text,
            policy_id=decision.policy_id,
            gesture_id=decision.gesture_id,
            timings=Timings(
                transcription_ms=transcription_ms,
                audio_ms=feature_times.audio_ms,
                text_ms=feature_times.text_ms,
                classifier_ms=classifier_ms,
                total_ms=(time.perf_counter() - overall) * 1000,
            ),
        )

    def predict_features(self, text: np.ndarray, audio: np.ndarray) -> np.ndarray:
        """Apply the deployed head(s) and calibration to shared encoder features."""

        def head_logits(head, artifact, vector):
            features = torch.from_numpy(vector.astype(np.float32))
            features = (features - artifact["normalization_mean"]) / artifact["normalization_scale"]
            return head(features[None, :])

        with torch.inference_mode():
            if self.modality == "probability":
                text_probs = torch.softmax(
                    head_logits(self.heads["text"], self.artifact["text_head"], text), dim=1
                )
                audio_probs = torch.softmax(
                    head_logits(self.heads["audio"], self.artifact["audio_head"], audio), dim=1
                )
                alpha = self.artifact["alpha"]
                logits = (alpha * text_probs + (1 - alpha) * audio_probs).clamp_min(1e-12).log()
            else:
                vector = {"text": text, "audio": audio, "fusion": np.concatenate([text, audio])}[
                    self.modality
                ]
                logits = head_logits(self.model, self.artifact, vector)
            return torch.softmax(logits / self.artifact.get("temperature", 1.0), dim=1)[0].numpy()


def predict(
    checkpoint: Path,
    text: str | None,
    audio_path: Path,
    asr_model: str = "tiny.en",
    cpu_threads: int = 8,
    local_files_only: bool = False,
) -> PredictionResult:
    """One-shot compatibility wrapper; persistent callers should reuse InferencePipeline."""
    pipeline = InferencePipeline(
        checkpoint,
        asr_model=asr_model,
        cpu_threads=cpu_threads,
        local_files_only=local_files_only,
    )
    return pipeline.predict(audio_path, text=text)

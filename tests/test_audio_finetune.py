from pathlib import Path

from torch import nn

from scripts.finetune_audio import AudioDataset, unfreeze_last_layers
from sentient_signals.data import ManifestItem


class FakeAudioModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.wav2vec2 = nn.Module()
        self.wav2vec2.encoder = nn.Module()
        self.wav2vec2.encoder.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.projector = nn.Linear(2, 2)
        self.classifier = nn.Linear(2, 7)


def test_only_requested_tail_and_heads_are_trainable() -> None:
    model = FakeAudioModel()
    names = unfreeze_last_layers(model, 2)

    assert names
    assert all(
        not parameter.requires_grad for parameter in model.wav2vec2.encoder.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.wav2vec2.encoder.layers[-2].parameters()
    )
    assert all(parameter.requires_grad for parameter in model.projector.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_audio_dataset_uses_portable_processed_path(monkeypatch) -> None:
    item = ManifestItem("train", 1, 2, "speaker", "text", "joy", 3, "source", "C:/old.wav")
    root = Path("portable-root")
    expected = root / "train" / "dia1_utt2.wav"
    monkeypatch.setattr("scripts.finetune_audio.read_pcm16_wav", lambda path: [path])

    audio, label = AudioDataset([item], root)[0]

    assert audio == [expected]
    assert label == 3

import numpy as np, pandas as pd, pytest, soundfile as sf
from pathlib import Path
from audioshield.data.safe_audio import load_audio_strict, AudioReadError
from audioshield.data.aug_assets import fingerprint_asset_dir, resolve_aug_assets, AugAssetError
import sys; sys.path.insert(0, "scripts")
from extend_manifests import derive

def test_strict_loader_raises_with_row_identity(tmp_path):
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"\x00" * 64)
    with pytest.raises(AudioReadError) as e:
        load_audio_strict(bad, "corpusX/bad.wav", allowlist=set())
    assert "corpusX/bad.wav" in str(e.value)

def test_strict_loader_allowlist_returns_none(tmp_path):
    bad = tmp_path / "bad.wav"; bad.write_bytes(b"\x00" * 64)
    assert load_audio_strict(bad, "x/bad.wav", allowlist={"x/bad.wav"}) is None

def test_strict_loader_reads_good_audio(tmp_path):
    p = tmp_path / "ok.wav"; sf.write(p, np.zeros(1600, dtype="float32") + 0.01, 16000)
    x, sr = load_audio_strict(p, "x/ok.wav", allowlist=set())
    assert sr == 16000 and x.shape == (1600,)

def test_aug_fingerprint_stable_and_failfast(tmp_path):
    with pytest.raises(AugAssetError): fingerprint_asset_dir(tmp_path / "nope")
    d = tmp_path / "rirs"; d.mkdir()
    for i in range(12): sf.write(d / f"r{i}.wav", np.zeros(160, dtype="float32"), 16000)
    f1, f2 = fingerprint_asset_dir(d), fingerprint_asset_dir(d)
    assert f1["listing_sha256"] == f2["listing_sha256"] and f1["n_files"] == 12
    with pytest.raises(AugAssetError): resolve_aug_assets({"augmentation": {}})

CASES = [  # (row, expected subset) — pinned to the LIVE schema samples you pasted
    (dict(corpus="ai4t", utt_id="ai4t/real/-lPqD0Kj-gA_000.wav", path="datasets/05_AI4T/AI4T_dataset_seg/real/-lPqD0Kj-gA_000.wav",
          target="0", attack="bonafide"), dict(source_id="-lPqD0Kj-gA", generator_id="NA")),
    (dict(corpus="diffssd", utt_id="diffssd/real_speech/librispeech/dev-clean/1272/128104/x.flac",
          path="datasets/03_DiffSSD/real_speech/librispeech/dev-clean/1272/128104/x.flac",
          target="0", attack="bonafide"), dict(speaker_id="ls-1272", source_id="ls-1272-128104", language="en")),
    (dict(corpus="replaydf", utt_id="replaydf/wav/09252b6aeda2/spoof/bark/de/9d0b.wav",
          path="datasets/04_ReplayDF/wav/09252b6aeda2/spoof/bark/de/9d0b.wav",
          target="1", attack="openvoicev2"), dict(channel_id="09252b6aeda2", generator_id="bark", language="de")),
    (dict(corpus="vctk", utt_id="vctk/p225/p225_001.wav", path="datasets/09_VCTK/.../p225/p225_001.wav",
          target="0", attack="bonafide"), dict(speaker_id="p225", language="en")),
    (dict(corpus="mlaad", utt_id="mlaad/fake/de/MeloTTS/book_01_f000015.wav", path="10_MLAAD/fake/de/MeloTTS/book_01_f000015.wav",
          target="1", attack="na"), dict(language="de", generator_id="MeloTTS", source_id="book_01")),
]
@pytest.mark.parametrize("row,exp", CASES)
def test_derive_rules(row, exp):
    got = derive(row)
    for k, v in exp.items():
        assert got[k] == v, f"{row['corpus']}.{k}: got {got[k]!r}, want {v!r}"
    assert all(got[k] != "" for k in got)


def test_replaydf_generator_from_path_not_placeholder():
    from extend_manifests import derive
    got = derive(dict(corpus="replaydf", utt_id="replaydf/wav/09252b6aeda2/spoof/bark/de/9d0b.wav",
                      path="datasets/04_ReplayDF/wav/09252b6aeda2/spoof/bark/de/9d0b.wav",
                      target="1", attack="openvoicev2"))
    assert got["generator_id"] == "bark", got["generator_id"]   # NOT the placeholder
    assert got["language"] == "de"

def test_inthewild_placeholder_attack_yields_na_generator():
    from extend_manifests import derive
    got = derive(dict(corpus="inthewild", utt_id="inthewild/1.wav",
                      path="datasets/02_In-the-Wild/release_in_the_wild/1.wav",
                      target="1", attack="openvoicev2"))
    assert got["generator_id"] == "NA", got["generator_id"]     # honest missing, not fiction


def test_diffssd_flat_generator_from_path_fallback():
    """Bug fix: the diffssd path-fallback (used when attack doesn't already supply a
    generator) must index BY NAME (generated_speech + 1), not a hardcoded p[2] -- p[2]
    was literally the "generated_speech" directory component itself."""
    got = derive(dict(corpus="diffssd", utt_id="diffssd/generated_speech/gradtts/sentence_0.wav",
                      path="datasets/03_DiffSSD/generated_speech/gradtts/sentence_0.wav",
                      target="1", attack="na"))
    assert got["generator_id"] == "gradtts", got["generator_id"]


def test_diffssd_openvoicev2_speaker_and_accent_from_path():
    got = derive(dict(
        corpus="diffssd",
        utt_id="diffssd/generated_speech/openvoicev2/speaker_100/sentence_0_en-au.wav",
        path="datasets/03_DiffSSD/generated_speech/openvoicev2/speaker_100/sentence_0_en-au.wav",
        target="1", attack="openvoicev2",
    ))
    assert got["generator_id"] == "openvoicev2", got["generator_id"]
    assert got["speaker_id"] == "speaker_100", got["speaker_id"]
    assert got["language"] == "en-au", got["language"]


def test_diffssd_manifest_never_labels_generated_speech_as_generator():
    """Regenerated-manifest invariant, not just a synthetic-row check: the
    global (not per-corpus) PLACEHOLDER_ATTACK bug produced
    generator_id="generated_speech" for every diffssd openvoicev2 row (25,000 of
    them) in the pre-fix manifest."""
    df = pd.read_csv("manifests/v2/diffssd.csv", dtype=str, keep_default_na=False)
    assert not (df["generator_id"] == "generated_speech").any()

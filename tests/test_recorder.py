import pytest

from app.core import recorder


FAKE_DEVICES = [
    {"name": "Microsoft 音效對應表 - Input", "max_input_channels": 2, "hostapi": 1},
    {"name": "麥克風 (PD200X Podcast Microphone)", "max_input_channels": 2, "hostapi": 0},
    {"name": "喇叭 (Realtek Audio)", "max_input_channels": 0, "hostapi": 0},
    {"name": "麥克風 (PD200X Podcast Microphone)", "max_input_channels": 1, "hostapi": 1},
]
FAKE_HOSTAPIS = [{"name": "Windows WASAPI"}, {"name": "MME"}]


class _FakeDefault:
    device = (0, None)  # 預設輸入 = index 0（MME 對應表）


@pytest.fixture(autouse=True)
def fake_sounddevice(monkeypatch):
    monkeypatch.setattr(recorder.sd, "query_devices",
                        lambda idx=None: FAKE_DEVICES[idx]
                        if idx is not None else FAKE_DEVICES)
    monkeypatch.setattr(
        recorder.sd, "query_hostapis", lambda i: FAKE_HOSTAPIS[i])
    monkeypatch.setattr(recorder.sd, "default", _FakeDefault())


def test_list_input_devices_only_wasapi_inputs():
    assert recorder.list_input_devices() == [
        "麥克風 (PD200X Podcast Microphone)",
    ]


def test_resolve_default_returns_none():
    assert recorder.resolve_input_device("default") is None
    assert recorder.resolve_input_device(None) is None
    assert recorder.resolve_input_device("") is None


def test_resolve_named_device_returns_mme_index():
    assert recorder.resolve_input_device(
        "麥克風 (PD200X Podcast Microphone)") == 1


def test_resolve_missing_device_raises():
    with pytest.raises(LookupError):
        recorder.resolve_input_device("已拔掉的 USB 麥克風")


def test_names_match_handles_mme_truncation():
    from app.core.mic_isolation import _names_match
    assert _names_match("麥克風 (PD200X Podcast Microphone)",
                        "麥克風 (PD200X Podcast Microphone)")
    # MME 名稱被截斷成 31 字元的情況
    assert _names_match("麥克風 (PD200X Podcast Microphone)",
                        "麥克風 (PD200X Podcast Micro")
    assert not _names_match("麥克風 (ToDesk Virtual Audio)",
                            "麥克風 (PD200X Podcast Microphone)")
    assert not _names_match("", "x")

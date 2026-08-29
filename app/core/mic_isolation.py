from ctypes import POINTER, c_uint, c_void_p, c_wchar_p

import comtypes
from comtypes import CLSCTX_ALL, COMMETHOD, GUID, HRESULT, IUnknown

from pycaw.pycaw import IAudioEndpointVolume
from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
from pycaw.constants import CLSID_MMDeviceEnumerator, DEVICE_STATE, EDataFlow

from .recorder import DEFAULT_DEVICE

_ROLES = (0, 1, 2)  # eConsole, eMultimedia, eCommunications


class IPolicyConfig(IUnknown):
    """未公開但長年穩定的介面（AudioSwitcher/SoundVolumeView 都在用），
    只呼叫 SetDefaultEndpoint，其餘方法只是佔住 vtable 位置。"""
    _iid_ = GUID("{f8679f50-850a-41cf-9c72-430f290290c8}")
    _methods_ = (
        COMMETHOD([], HRESULT, "GetMixFormat",
                  (["in"], c_wchar_p), (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "GetDeviceFormat",
                  (["in"], c_wchar_p), (["in"], c_uint),
                  (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "ResetDeviceFormat", (["in"], c_wchar_p)),
        COMMETHOD([], HRESULT, "SetDeviceFormat",
                  (["in"], c_wchar_p), (["in"], c_void_p), (["in"], c_void_p)),
        COMMETHOD([], HRESULT, "GetProcessingPeriod",
                  (["in"], c_wchar_p), (["in"], c_uint),
                  (["out"], POINTER(c_void_p)), (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "SetProcessingPeriod",
                  (["in"], c_wchar_p), (["in"], c_void_p)),
        COMMETHOD([], HRESULT, "GetShareMode",
                  (["in"], c_wchar_p), (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "SetShareMode",
                  (["in"], c_wchar_p), (["in"], c_void_p)),
        COMMETHOD([], HRESULT, "GetPropertyValue",
                  (["in"], c_wchar_p), (["in"], c_uint),
                  (["in"], c_void_p), (["out"], POINTER(c_void_p))),
        COMMETHOD([], HRESULT, "SetPropertyValue",
                  (["in"], c_wchar_p), (["in"], c_uint),
                  (["in"], c_void_p), (["in"], c_void_p)),
        COMMETHOD([], HRESULT, "SetDefaultEndpoint",
                  (["in"], c_wchar_p, "wszDeviceId"),
                  (["in"], c_uint, "eRole")),
        COMMETHOD([], HRESULT, "SetEndpointVisibility",
                  (["in"], c_wchar_p), (["in"], c_uint)),
    )


CLSID_PolicyConfigClient = GUID("{870af99c-171d-4f9e-af0d-e63df40c2bc9}")


def _names_match(endpoint_name: str, mme_name: str) -> bool:
    """MME 的裝置名稱是端點 FriendlyName 截斷到 31 字元的版本。"""
    if not endpoint_name or not mme_name:
        return False
    return endpoint_name == mme_name or endpoint_name.startswith(mme_name)


class MicIsolation:
    """按住錄音期間，讓「只有本程式聽得到麥克風」：

    1. 把 Windows 預設錄音裝置（三種 role）切到一顆誘餌裝置——
       跟隨系統預設的軟體（Discord/Teams/瀏覽器等）會立刻跟過去。
    2. 把除了錄音中那顆以外的所有錄音裝置整顆靜音（誘餌也在內），
       所以跟過去的軟體聽到的是無聲。
    3. 放開後恢復靜音狀態與原本的預設裝置。

    限制：手動指定綁死在同一顆實體麥克風的軟體無法隔離——Windows 對
    錄音裝置的 per-app session 靜音等於整顆裝置靜音（實測），靜音了
    別人自己也錄不到；那類軟體請用它自己的靜音鍵。

    isolate/restore 需在同一個已 CoInitialize 的執行緒呼叫。
    """

    def __init__(self):
        self._muted = []
        self._original_defaults = None  # {role: device_id}

    @staticmethod
    def _friendly_name(device) -> str:
        from pycaw.utils import AudioUtilities
        try:
            return AudioUtilities.CreateDevice(device).FriendlyName or ""
        except Exception:
            return ""

    def isolate(self, recording_device_name: str) -> int:
        """回傳被靜音的其他裝置數。"""
        self._muted = []
        self._original_defaults = None
        enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER,
        )
        keep_id = None
        if not recording_device_name or recording_device_name == DEFAULT_DEVICE:
            keep_id = enumerator.GetDefaultAudioEndpoint(
                EDataFlow.eCapture.value, 0).GetId()

        devices = enumerator.EnumAudioEndpoints(
            EDataFlow.eCapture.value, DEVICE_STATE.ACTIVE.value)
        ours = None      # (id)
        others = []      # [(id, IMMDevice)]
        for i in range(devices.GetCount()):
            try:
                device = devices.Item(i)
                dev_id = device.GetId()
                if keep_id is not None:
                    is_ours = dev_id == keep_id
                else:
                    is_ours = _names_match(
                        self._friendly_name(device), recording_device_name)
                if is_ours and ours is None:
                    ours = dev_id
                else:
                    others.append((dev_id, device))
            except comtypes.COMError:
                continue

        # 1) 預設裝置切到誘餌（任何一顆非錄音裝置；等下會被靜音）
        if others:
            decoy_id = others[0][0]
            try:
                originals = {
                    role: enumerator.GetDefaultAudioEndpoint(
                        EDataFlow.eCapture.value, role).GetId()
                    for role in _ROLES
                }
                policy = comtypes.CoCreateInstance(
                    CLSID_PolicyConfigClient, IPolicyConfig,
                    comtypes.CLSCTX_ALL)
                for role in _ROLES:
                    policy.SetDefaultEndpoint(decoy_id, role)
                self._original_defaults = originals
            except comtypes.COMError:
                self._original_defaults = None

        # 2) 靜音其他所有裝置（只記錄原本沒靜音的，恢復時不會誤開）
        for _dev_id, device in others:
            try:
                volume = device.Activate(
                    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
                ).QueryInterface(IAudioEndpointVolume)
                if volume.GetMute():
                    continue
                volume.SetMute(1, None)
                self._muted.append(volume)
            except comtypes.COMError:
                continue
        return len(self._muted)

    # 舊名稱保留相容
    mute_other_devices = isolate

    def restore(self):
        for volume in self._muted:
            try:
                volume.SetMute(0, None)
            except comtypes.COMError:
                continue
        self._muted = []
        if self._original_defaults:
            try:
                policy = comtypes.CoCreateInstance(
                    CLSID_PolicyConfigClient, IPolicyConfig,
                    comtypes.CLSCTX_ALL)
                for role, dev_id in self._original_defaults.items():
                    policy.SetDefaultEndpoint(dev_id, role)
            except comtypes.COMError:
                pass
        self._original_defaults = None

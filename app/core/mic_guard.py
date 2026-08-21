import os

import comtypes
from comtypes import CLSCTX_ALL

from pycaw.pycaw import (
    IAudioSessionControl2,
    IAudioSessionManager2,
    ISimpleAudioVolume,
)
from pycaw.api.mmdeviceapi import IMMDeviceEnumerator
from pycaw.constants import CLSID_MMDeviceEnumerator, DEVICE_STATE, EDataFlow


class MicGuard:
    """錄音期間把其他 app 在所有作用中錄音裝置上的 session 靜音，事後恢復。

    只把「原本沒被靜音」的 session 靜音並記錄，恢復時只解除這些，
    不會誤開使用者原本就自行靜音的 app。
    mute_others() 與 unmute_others() 必須在同一個執行緒呼叫（COM apartment），
    且該執行緒需先 CoInitialize——建議用 single-thread executor 包住。
    """

    def __init__(self):
        self._muted_sessions = []

    @staticmethod
    def co_initialize():
        comtypes.CoInitialize()

    def mute_others(self):
        self._muted_sessions = []
        my_pid = os.getpid()
        enumerator = comtypes.CoCreateInstance(
            CLSID_MMDeviceEnumerator,
            IMMDeviceEnumerator,
            comtypes.CLSCTX_INPROC_SERVER,
        )
        devices = enumerator.EnumAudioEndpoints(
            EDataFlow.eCapture.value, DEVICE_STATE.ACTIVE.value)
        for i in range(devices.GetCount()):
            try:
                device = devices.Item(i)
                interface = device.Activate(
                    IAudioSessionManager2._iid_, CLSCTX_ALL, None)
                manager = interface.QueryInterface(IAudioSessionManager2)
                session_enum = manager.GetSessionEnumerator()
            except comtypes.COMError:
                continue
            for j in range(session_enum.GetCount()):
                try:
                    ctl = session_enum.GetSession(j)
                    ctl2 = ctl.QueryInterface(IAudioSessionControl2)
                    pid = ctl2.GetProcessId()
                    if pid in (0, my_pid):
                        continue
                    volume = ctl.QueryInterface(ISimpleAudioVolume)
                    if volume.GetMute():
                        continue
                    volume.SetMute(1, None)
                    self._muted_sessions.append(volume)
                except comtypes.COMError:
                    continue

    def unmute_others(self):
        for volume in self._muted_sessions:
            try:
                volume.SetMute(0, None)
            except comtypes.COMError:
                continue
        self._muted_sessions = []

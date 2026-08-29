# -*- coding: utf-8 -*-
"""開機自動啟動：在使用者的「啟動」資料夾建立/移除捷徑。

程式啟動時就會在背景載入語音模型，之後隨時按熱鍵都是零等待
（縮小成懸浮球即可，不用開著視窗）。
"""
import os
import sys
from pathlib import Path

from ..config import BASE_DIR

PROJECT = BASE_DIR
# 打包後 exe 就是自己；開發環境用專案根目錄的啟動器
EXE = (Path(sys.executable) if getattr(sys, "frozen", False)
       else PROJECT / "AI語音中翻英.exe")
ICON = PROJECT / "app.ico"


def _shortcut_path() -> Path:
    return (Path(os.environ["APPDATA"]) /
            "Microsoft/Windows/Start Menu/Programs/Startup/AI語音中翻英.lnk")


def is_enabled() -> bool:
    return _shortcut_path().exists()


def set_enabled(enabled: bool):
    path = _shortcut_path()
    if not enabled:
        if path.exists():
            path.unlink()
        return
    import pythoncom
    from win32com.propsys import propsys, pscon
    from win32com.shell import shell

    from ..config import APP_ID

    pythoncom.CoInitialize()
    link = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
        shell.IID_IShellLink)
    link.SetPath(str(EXE))
    link.SetWorkingDirectory(str(PROJECT))
    link.SetIconLocation(str(ICON), 0)
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(APP_ID))
    store.Commit()
    persist = link.QueryInterface(pythoncom.IID_IPersistFile)
    persist.Save(str(path), 0)

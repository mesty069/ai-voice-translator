# -*- coding: utf-8 -*-
"""在開始功能表建立「AI語音中翻英」捷徑（含 AppUserModelID）。

執行後可在開始功能表找到程式並釘選到工作列；因為捷徑與程式視窗
使用同一個 AppUserModelID，執行中的視窗會跟釘選圖示併成同一顆。
"""
import os
import sys
from pathlib import Path

import pythoncom
from win32com.propsys import propsys, pscon
from win32com.shell import shell

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.main import APP_ID  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
EXE = PROJECT / "AI語音中翻英.exe"
ICON = PROJECT / "app.ico"
SHORTCUT = (Path(os.environ["APPDATA"]) /
            "Microsoft/Windows/Start Menu/Programs/AI語音中翻英.lnk")


def main():
    pythoncom.CoInitialize()
    link = pythoncom.CoCreateInstance(
        shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
        shell.IID_IShellLink)
    link.SetPath(str(EXE))
    link.SetWorkingDirectory(str(PROJECT))
    link.SetIconLocation(str(ICON), 0)
    link.SetDescription("按住熱鍵說中文，AI 梳理並翻譯成英文")

    # 設定 AppUserModelID（與 app/main.py 一致），釘選後視窗才會合併
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID,
                   propsys.PROPVARIANTType(APP_ID))
    store.Commit()

    persist = link.QueryInterface(pythoncom.IID_IPersistFile)
    persist.Save(str(SHORTCUT), 0)
    print(f"shortcut created: {SHORTCUT}")

    # 驗證 AUMID 有寫進去
    store2 = propsys.SHGetPropertyStoreFromParsingName(str(SHORTCUT))
    value = store2.GetValue(pscon.PKEY_AppUserModel_ID).GetValue()
    print(f"AppUserModelID: {value}")
    assert value == APP_ID


if __name__ == "__main__":
    main()

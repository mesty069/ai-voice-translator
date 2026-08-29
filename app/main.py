import ctypes
import os
import sys
from pathlib import Path


def _ensure_std_streams():
    """pythonw / 打包 exe 沒有 console，sys.stdout/stderr 是 None；
    任何第三方庫（tqdm、transformers 警告…）一寫就崩。給它們一個黑洞。
    必須在所有第三方 import 之前執行——qfluentwidgets 等套件在 import 時
    就會往 stdout 寫東西。"""
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))


_ensure_std_streams()

import truststore  # noqa: E402

from app.config import APP_ID, BASE_DIR  # noqa: E402  工作列釘選/群組識別

ICON_PATH = BASE_DIR / "app.ico"

# 改用 Windows 憑證存放區驗證 TLS：防毒/VPN（如 Norton Web Shield）會攔截
# HTTPS 並用自己的根憑證重簽，Python 內建的 certifi 清單不認得它們。
# 必須在建立任何 SSL 連線前呼叫。
truststore.inject_into_ssl()

from PySide6.QtGui import QColor, QIcon
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

from qfluentwidgets import Theme, setTheme, setThemeColor

from app.config import Config
from app.controller import AppController
from app.ui.main_window import MainWindow

SINGLE_INSTANCE_KEY = "ai-voice-zh2en-single-instance"


def acquire_single_instance():
    """確保同時只有一個實例（多實例會搶熱鍵與麥克風）。

    已有實例時通知它顯示視窗，本次直接退出。
    回傳 QLocalServer（需保持存活）；已有實例時回傳 None。
    """
    probe = QLocalSocket()
    probe.connectToServer(SINGLE_INSTANCE_KEY)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return None
    # 清掉前次異常結束殘留的 socket 檔，再建立伺服器
    QLocalServer.removeServer(SINGLE_INSTANCE_KEY)
    server = QLocalServer()
    if not server.listen(SINGLE_INSTANCE_KEY):
        return None
    return server


def main():
    # 必須在建立任何視窗前設定，執行中的視窗才會跟釘選的捷徑併成同一顆
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass

    app = QApplication(sys.argv)
    if ICON_PATH.exists():
        app.setWindowIcon(QIcon(str(ICON_PATH)))
    # 視窗收成懸浮球時主視窗會 hide，不能因此結束程式
    app.setQuitOnLastWindowClosed(False)

    instance_server = acquire_single_instance()
    if instance_server is None:
        sys.exit(0)

    config = Config()
    theme = config.get("ui", "theme", default="auto")
    setTheme({"auto": Theme.AUTO, "light": Theme.LIGHT,
              "dark": Theme.DARK}.get(theme, Theme.AUTO))
    setThemeColor(QColor(config.get("ui", "theme_color", default="#0078d4")))

    controller = AppController(config)
    window = MainWindow(config, controller)

    def _on_second_launch():
        conn = instance_server.nextPendingConnection()
        if conn is not None:
            conn.disconnectFromServer()
        window._restore_from_bubble()
        window.raise_()
        window.activateWindow()

    instance_server.newConnection.connect(_on_second_launch)

    window.show()
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

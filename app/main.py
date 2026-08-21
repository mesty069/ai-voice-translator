import sys

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from qfluentwidgets import Theme, setTheme, setThemeColor

from app.config import Config
from app.controller import AppController
from app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    # 視窗收成懸浮球時主視窗會 hide，不能因此結束程式
    app.setQuitOnLastWindowClosed(False)

    config = Config()
    theme = config.get("ui", "theme", default="auto")
    setTheme({"auto": Theme.AUTO, "light": Theme.LIGHT,
              "dark": Theme.DARK}.get(theme, Theme.AUTO))
    setThemeColor(QColor(config.get("ui", "theme_color", default="#0078d4")))

    controller = AppController(config)
    window = MainWindow(config, controller)
    window.show()
    controller.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

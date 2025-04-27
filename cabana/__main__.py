from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import sys
from pathlib import Path
from .cabana_gui import MainWindow

def main():
    # Enable High DPI display before creating QApplication
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    window = MainWindow()
    icon_path = Path(__file__).resolve().parent.parent / "cabana-logo.ico"
    app.setWindowIcon(QIcon(str(icon_path)))
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
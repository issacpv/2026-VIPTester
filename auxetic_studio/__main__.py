"""Entry point: ``python -m auxetic_studio``."""

import sys

from PyQt6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv=None):
    if argv is None:
        argv = sys.argv
    app = QApplication.instance() or QApplication(argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

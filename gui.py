import queue
import sys
import threading

from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile

import mlx.core as mx
import mlx_whisper
from mlx_whisper.transcribe import ModelHolder
import numpy as np

from record import LiveRecorder


def load_ui_widget(path: str, parent: QWidget = None) -> QWidget:
    loader = QUiLoader()
    ui_file = QFile(path)
    if not ui_file.open(QFile.ReadOnly):
        raise FileNotFoundError(f"Cannot open UI file: {path}")
    widget = loader.load(ui_file, parent)
    ui_file.close()
    if widget is None:
        raise RuntimeError(f"Failed to load UI from: {path}")
    return widget


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = load_ui_widget('ui/mainwindow.ui', self)

        self.recorder = LiveRecorder()
        self.audio_thread = None
        self.transcribe_thread = None
        self.ui.recordButton.clicked.connect(self.start_recording)

    def start_recording(self):
        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.audio_thread = threading.Thread(target=self.audio_thread_func)
            self.transcribe_thread = threading.Thread(
                target=self.transcribe_thread_func)
            self.audio_thread.start()
            self.transcribe_thread.start()

    def audio_thread_func(self):
        print("[Info] Audio thread started.")
        self.recorder.start()
        print("[Info] Audio thread stopped.")

    def transcribe_thread_func(self):
        print("[Info] Loading model...")
        holder = ModelHolder.get_model('mlx-community/whisper-small-mlx', mx.float16)
        try:
            while True:
                try:
                    chunk = self.recorder.queue.get(timeout=1.0)
                    print('get chunk')
                except queue.Empty:
                    if not self.recorder.running:
                        break
                    continue

                chunk = np.squeeze(chunk)

                result = mlx_whisper.transcribe(
                    chunk, path_or_hf_repo='mlx-community/whisper-small-mlx', word_timestamps=True
                )

                if result["segments"]:
                    text = result["segments"][-1]["text"]
                    print(text, flush=True)
                    self.ui.chatContent.addItem(text)
        except Exception as e:
            print(e)
        print("[Info] Transcription thread stopped.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.ui.show()
    sys.exit(app.exec())

import sys
import threading
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QListWidgetItem
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import Qt, QFile, QObject, QThread, QObject, QThread, Signal, Slot, Qt
from RealtimeSTT import AudioToTextRecorder


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


class TranscriptionWorker(QObject):
    stabilized = Signal(str)
    finished = Signal()

    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self._running = True

    @Slot()
    def run(self):
        while self._running and not self.recorder.is_shut_down:
            s = self.recorder.text()
            if s:
                self.stabilized.emit(s)
        self.finished.emit()

    def stop(self):
        self._running = False


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = load_ui_widget('ui/mainwindow.ui', self)

        self.recorder = AudioToTextRecorder(
            model='base',
            language='en',
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.on_realtime_transcription_update,
            spinner=False
        )

        self.chat_lock = threading.Lock()

        self.transcribe_thread = None
        self.transcribe_worker = None

        self.ui.recordButton.clicked.connect(self.start_recording)
        # self.ui.stopButton.clicked.connect(self.stop_recording)

    def on_realtime_transcription_update(self, s):
        with self.chat_lock:
            print('Update:', s)
            if self.ui.chatContent.count() > 0 and \
                    self.ui.chatContent.item(self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp':
                item = self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1)
                item.setText(s)
            else:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, 'temp')
                self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

    @Slot(str)
    def on_realtime_transcription_stabilized(self, s):
        with self.chat_lock:
            print('Stable:', s)
            if self.ui.chatContent.count() > 0 and \
                    self.ui.chatContent.item(self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp':
                self.ui.chatContent.takeItem(self.ui.chatContent.count() - 1)
            item = QListWidgetItem(s)
            self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

    def start_recording(self):
        self.recorder.start()

        self.transcribe_thread = QThread()
        self.transcribe_worker = TranscriptionWorker(self.recorder)

        self.transcribe_worker.moveToThread(self.transcribe_thread)
        self.transcribe_worker.stabilized.connect(
            self.on_realtime_transcription_stabilized)
        self.transcribe_thread.started.connect(self.transcribe_worker.run)
        self.transcribe_worker.finished.connect(self.transcribe_thread.quit)
        self.transcribe_worker.finished.connect(
            self.transcribe_worker.deleteLater)
        self.transcribe_thread.finished.connect(
            self.transcribe_thread.deleteLater)

        self.transcribe_thread.start()

    def stop_recording(self):
        self.recorder.stop()
        if self.transcribe_worker:
            self.transcribe_worker.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.ui.show()
    sys.exit(app.exec())

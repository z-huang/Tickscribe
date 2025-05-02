import sys
import threading
from transcribe import transcribe_file
import re
import os

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QFileDialog, QListWidgetItem, QInputDialog
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import (
    Qt, QFile, QObject, QThread,
    Signal, Slot, QTimer, QMetaObject
)
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


class FileTranscriptionWorker(QObject):
    transcription_completed = Signal(str)
    finished = Signal()

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    @Slot()
    def run(self):
        text = transcribe_file(
            self.file_path,
            model_path='mlx-community/whisper-small-mlx'
        )
        self.transcription_completed.emit(text)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = load_ui_widget('ui/mainwindow.ui', self)

        # 目前使用的 session 檔案 (完整路徑)
        self.current_session_file = None

        # Transcriber
        self.recorder = AudioToTextRecorder(
            model='base',
            language='en',
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.on_realtime_transcription_update,
            realtime_model_type='base',
            spinner=False,
            compute_type="float32",
            no_log_file=True,
        )

        self.chat_lock = threading.Lock()
        self.transcribe_thread = None
        self.transcribe_worker = None
        self.is_recording = False

        self.update_buffer = None
        self.update_timer = QTimer()
        self.update_timer.setInterval(300)
        self.update_timer.timeout.connect(self.flush_update_buffer)
        self.update_timer.setSingleShot(True)

        # 介面按鈕連接
        self.ui.newChatButton.clicked.connect(self.new_chat)
        self.ui.recordButton.clicked.connect(self.toggle_recording)
        self.ui.recordButton.setText("Start Recording")
        self.ui.uploadButton.clicked.connect(self.open_and_transcribe)

        self.ui.chatContent.setWordWrap(True)

        # 確保 transcript 資料夾存在
        self.transcript_dir = 'transcripts'
        os.makedirs(self.transcript_dir, exist_ok=True)

        # 左側清單：點選時載入對應的 transcript
        self.ui.chatList.itemClicked.connect(self.load_transcript)
        self.load_session_list()

    def new_chat(self):
        # 1) 取得新的 session 名稱
        name, ok = QInputDialog.getText(self, "New Chat", "Enter chat name:")
        if not ok or not name.strip():
            return
        base = name.strip()
        filename = f"{base}.txt"
        filepath = os.path.join(self.transcript_dir, filename)
        counter = 1
        # 處理重名
        while os.path.exists(filepath):
            filename = f"{base}({counter}).txt"
            filepath = os.path.join(self.transcript_dir, filename)
            counter += 1
        # 建立空檔案
        with open(filepath, 'w', encoding='utf-8'):
            pass
        # 重新載入清單並切換到新 session
        self.load_session_list()
        items = self.ui.chatList.findItems(
            os.path.splitext(filename)[0], Qt.MatchExactly)
        if items:
            self.ui.chatList.setCurrentItem(items[0])
            self.load_transcript(items[0])

    def load_session_list(self):
        """掃描 transcript_dir，把檔名（去掉 .txt）加入清單"""
        self.ui.chatList.clear()
        for fn in sorted(os.listdir(self.transcript_dir)):
            if fn.lower().endswith('.txt'):
                name = fn[:-4]
                self.ui.chatList.addItem(name)

    def load_transcript(self, item):
        """載入指定 session，同時設定 current_session_file"""
        session_name = item.text()
        self.current_session_file = os.path.join(
            self.transcript_dir, f"{session_name}.txt")
        self.ui.chatContent.clear()
        with open(self.current_session_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.ui.chatContent.addItem(line)

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def on_realtime_transcription_update(self, s):
        self.update_buffer = s
        QMetaObject.invokeMethod(
            self.update_timer, "start", Qt.QueuedConnection)

    @Slot()
    def flush_update_buffer(self):
        if self.update_buffer is None:
            self.update_timer.stop()
            return

        with self.chat_lock:
            s = self.update_buffer
            self.update_buffer = None

            if (
                self.ui.chatContent.count() > 0 and
                self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp'
            ):
                item = self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1)
                item.setText(s)
            else:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, 'temp')
                self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

        self.update_timer.stop()

    @Slot(str)
    def on_realtime_transcription_stabilized(self, s):
        with self.chat_lock:
            # 移除暫存
            if (
                self.ui.chatContent.count() > 0 and
                self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp'
            ):
                self.ui.chatContent.takeItem(self.ui.chatContent.count() - 1)
            # 加入最終文字
            item = QListWidgetItem(s)
            self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

            # 寫入目前 session 檔案
            if self.current_session_file:
                with open(self.current_session_file, 'a', encoding='utf-8') as f:
                    f.write(s + '\n')

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

        self.is_recording = True
        self.ui.recordButton.setText("Stop Recording")

    def stop_recording(self):
        self.recorder.stop()
        if self.transcribe_worker:
            self.transcribe_worker.stop()

        self.is_recording = False
        self.ui.recordButton.setText("Start Recording")

    def open_and_transcribe(self):
        # 打開檔案對話框，只顯示 .wav
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 WAV 檔",
            "",
            "WAV Files (*.wav)"
        )
        if not file_path:
            return

        self.ui.statusbar.showMessage("Transcription in progress...")

        self.transcribe_thread = QThread()
        self.transcribe_worker = FileTranscriptionWorker(file_path)

        self.transcribe_worker.moveToThread(self.transcribe_thread)
        self.transcribe_worker.transcription_completed.connect(
            self.on_file_transcription_completed)
        self.transcribe_thread.started.connect(self.transcribe_worker.run)
        self.transcribe_worker.finished.connect(self.transcribe_thread.quit)
        self.transcribe_worker.finished.connect(
            self.transcribe_worker.deleteLater)
        self.transcribe_thread.finished.connect(
            self.transcribe_thread.deleteLater)

        self.transcribe_thread.start()

    @Slot(str)
    def on_file_transcription_completed(self, text):
        sentences = re.split(r'(?<=[。！？\.\!?])\s*', text)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                self.ui.chatContent.addItem(sent)

        self.ui.statusbar.showMessage("Transcription completed.", 2000)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.ui.show()
    sys.exit(app.exec())

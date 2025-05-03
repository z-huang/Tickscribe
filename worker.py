from PySide6.QtCore import QObject, Signal, Slot
from RealtimeSTT import AudioToTextRecorder

from utils import transcribe_file


class TranscriptionWorker(QObject):
    stabilized = Signal(str)
    finished = Signal()

    def __init__(self, recorder: AudioToTextRecorder):
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
            model_path="mlx-community/whisper-small-mlx"
        )
        self.transcription_completed.emit(text)
        self.finished.emit()

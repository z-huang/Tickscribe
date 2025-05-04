import mlx_whisper
from PySide6.QtCore import QObject, Signal, Slot
from RealtimeSTT import AudioToTextRecorder


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
        result = mlx_whisper.transcribe(
            self.file_path,
            path_or_hf_repo='mlx-community/whisper-small-mlx'
        )
        text = result['text']
        self.transcription_completed.emit(text)
        self.finished.emit()

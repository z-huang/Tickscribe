import mlx_whisper
from mlx_lm import load, stream_generate
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


model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")


class LLMWorker(QObject):
    token_received = Signal(str)
    finished = Signal()

    def __init__(self, messages):
        super().__init__()
        self.messages = messages
        self._abort = False

    @Slot()
    def run(self):
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True
            )

        for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
            self.token_received.emit(response.text)

        self.finished.emit()

    def stop(self):
        self._abort = True

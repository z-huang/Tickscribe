import mlx_whisper
from mlx_lm import load, stream_generate
from PySide6.QtCore import QObject, Signal, Slot
from RealtimeSTT import AudioToTextRecorder

from utils import clean_str

# Worker for real-time transcription from audio input


class TranscriptionWorker(QObject):
    # Signal emitted when stabilized text is available
    stabilized = Signal(str)
    finished = Signal()       # Signal emitted when the worker finishes

    def __init__(self, recorder: AudioToTextRecorder):
        super().__init__()
        self.recorder = recorder
        self._running = True

    @Slot()
    def run(self):
        # Continuously check for new transcribed text while running
        while self._running and not self.recorder.is_shut_down:
            s = self.recorder.text()
            s = clean_str(s)
            if s:
                # Emit new stabilized text
                self.stabilized.emit(s)
        self.finished.emit()  # Emit finished signal when done

    def stop(self):
        self._running = False  # Stop the worker loop

# Worker for transcribing audio files


class FileTranscriptionWorker(QObject):
    # Signal emitted with the transcription result
    transcription_completed = Signal(str)
    finished = Signal()                    # Signal emitted when the worker finishes

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    @Slot()
    def run(self):
        # Transcribe the audio file using mlx_whisper
        result = mlx_whisper.transcribe(
            self.file_path,
            path_or_hf_repo='mlx-community/whisper-small-mlx'
        )
        text = result['text']
        self.transcription_completed.emit(text)  # Emit the transcribed text
        self.finished.emit()                     # Emit finished signal


# Load the LLM model and tokenizer
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

# Worker for generating responses from a language model


class LLMWorker(QObject):
    token_received = Signal(str)  # Signal emitted for each generated token
    finished = Signal()           # Signal emitted when generation is finished

    def __init__(self, messages):
        super().__init__()
        self.messages = messages
        self._abort = False

    @Slot()
    def run(self):
        # Prepare the prompt using the chat template if available
        if tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(
                self.messages,
                add_generation_prompt=True
            )

        # Stream generated tokens and emit them one by one
        for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
            self.token_received.emit(response.text)

        self.finished.emit()  # Emit finished signal when done

    def stop(self):
        self._abort = True  # Set abort flag (not used in run loop)

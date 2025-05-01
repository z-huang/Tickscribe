import io
import threading
import queue
import time
import mlx.core as mx
import numpy as np
from record import LiveRecorder
from mlx_whisper.transcribe import ModelHolder
import mlx_whisper

def audio_thread_func(recorder: LiveRecorder):
    print("[Info] Audio thread started.")
    recorder.start()
    print("[Info] Audio thread stopped.")

def transcribe_thread_func(recorder: LiveRecorder, model_name="mlx-community/whisper-large-v3-turbo"):
    print("[Info] Loading model...")
    holder = ModelHolder.get_model(model_name, mx.float16)
    print("[Info] Model loaded. Transcription thread started.")

    try:
        while True:
            try:
                chunk = recorder.queue.get(timeout=1.0)
            except queue.Empty:
                if not recorder.running:
                    break
                continue

            chunk = np.squeeze(chunk)

            # === Start timing ===
            start = time.perf_counter()

            result = mlx_whisper.transcribe(
                chunk, path_or_hf_repo=model_name, word_timestamps=True
            )

            end = time.perf_counter()
            latency_ms = (end - start) * 1000

            if result["segments"]:
                print(f"[{latency_ms:.2f} ms] {result['segments'][-1]['text']}", flush=True)

    except KeyboardInterrupt:
        print("Transcription interrupted.")
    print("[Info] Transcription thread stopped.")



if __name__ == "__main__":
    recorder = LiveRecorder()
    audio_thread = threading.Thread(target=audio_thread_func, args=(recorder,))
    transcribe_thread = threading.Thread(target=transcribe_thread_func, args=(recorder,"mlx-community/whisper-small-mlx",))

    audio_thread.start()
    transcribe_thread.start()

    try:
        input("Press Enter to stop...\n")
    finally:
        recorder.stop()
        audio_thread.join()
        transcribe_thread.join()
        print("All threads terminated.")
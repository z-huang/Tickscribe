import io
import threading
import queue
import time
import soundfile as sf
import mlx.core as mx
import numpy as np
from record import LiveRecorder
from mlx_whisper.transcribe import ModelHolder
import mlx_whisper

def audio_thread_func(recorder: LiveRecorder):
    print("[Info] Audio thread started.")
    recorder.start()
    while recorder.running:
        time.sleep(0.1)
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
            result = mlx_whisper.transcribe(chunk, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", word_timestamps=True)
            if result["segments"]:
                print(result["segments"][-1]["text"], flush=True)
                
    except KeyboardInterrupt:
        print("Transcription interrupted.")
    print("[Info] Transcription thread stopped.")



if __name__ == "__main__":
    recorder = LiveRecorder()
    audio_thread = threading.Thread(target=audio_thread_func, args=(recorder,))
    transcribe_thread = threading.Thread(target=transcribe_thread_func, args=(recorder,))

    audio_thread.start()
    transcribe_thread.start()

    try:
        input("Press Enter to stop...\n")
    finally:
        recorder.stop()
        audio_thread.join()
        transcribe_thread.join()
        print("All threads terminated.")
import io
import threading
import queue
import time
import mlx.core as mx
import numpy as np
from record import LiveRecorder
from mlx_whisper.transcribe import ModelHolder
from mlx_lm import load, generate
import mlx_whisper

confirmed_text = ''
unconfirmed_audio = None

def longest_overlap(a, b):
    max_len = min(len(a), len(b))
    for i in range(max_len, 0, -1):
        if a[-i:] == b[:i]:
            return a[-i:]
    return []

def after_second_last_sentence(text: str, punct: str = ".!?") -> str:
    """
    Return the substring that follows the SECOND-to-last punctuation mark
    in `punct`.  If there are fewer than two such marks, return the whole
    string after the last (or the whole string if none).
    """
    # Find last punctuation
    last = max(text.rfind(p) for p in punct)
    if last == -1:                          # no punctuation at all
        return text.strip()
    
    # Find second-to-last, searching only *before* the last occurrence
    second_last = max(text.rfind(p, 0, last) for p in punct)
    start_idx = second_last + 1 if second_last != -1 else last + 1
    return text[start_idx:].lstrip()

def audio_thread_func(recorder: LiveRecorder):
    print("[Info] Audio thread started.")
    recorder.start()

def transcribe_thread_func(recorder: LiveRecorder, model_name="mlx-community/whisper-large-v3-turbo"):
    global unconfirmed_audio, confirmed_text
    print("[Info] Loading model...")
    holder = ModelHolder.get_model(model_name, mx.float16)
    print("[Info] Model loaded. Transcription thread started.")
    prev_text = None
    try:
        while True:
            try:
                chunk = recorder.queue.get(timeout=1.0)
                chunk = np.squeeze(chunk)
                unconfirmed_audio = np.concatenate((unconfirmed_audio, chunk)) if unconfirmed_audio is not None else chunk
            except queue.Empty:
                if not recorder.running:
                    break
                continue

            prompt = confirmed_text
            if confirmed_text:
                prompt = after_second_last_sentence(confirmed_text)

            # === Start timing ===
            start = time.perf_counter()
            result = mlx_whisper.transcribe(
                unconfirmed_audio, path_or_hf_repo=model_name, word_timestamps=True, initial_prompt=prompt,
            )
            text = []
            if not result["segments"]:
                continue
            for w in result["segments"][0]["words"]:
                text.append((w["word"].strip(), w["start"], w["end"]))
            
            common_words = None
            if prev_text is not None:
                prev_words = [w[0] for w in prev_text]
                curr_words = [w[0] for w in text]
                common_words = longest_overlap(prev_words, curr_words)
                index = len(common_words) - 1
                timestamp = text[index][2]

            if common_words is not None and len(common_words) >= 2:
                confirmed_text += ' ' + ' '.join(common_words)
                print(f"[Confirmed] {confirmed_text}", flush=True)
                cutoff = round((timestamp - 0.15) * 16000)
                unconfirmed_audio = unconfirmed_audio[cutoff:]

            prev_text = text


            end = time.perf_counter()
            latency_ms = (end - start) * 1000

            if result["segments"]:
                print(f"[{latency_ms:.2f} ms] {result['text']}", flush=True)

    except KeyboardInterrupt:
        print("Transcription interrupted.")
    print("[Info] Transcription thread stopped.")


if __name__ == "__main__":
    recorder = LiveRecorder()
    transcribe_thread = threading.Thread(target=transcribe_thread_func, args=(recorder,"mlx-community/whisper-small-mlx",))
    audio_thread = threading.Thread(target=audio_thread_func, args=(recorder,))
    transcribe_thread.start()
    audio_thread.start()
    try:
        input("Press Enter to stop...\n")
    finally:
        recorder.stop()
        print("[Info] Audio thread stopped.")
        audio_thread.join()
        transcribe_thread.join()
        print("All threads terminated.")
        print(f"[Final text] {confirmed_text}")
        # model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        # prompt = f"""
        # Fix my paragraph. 
        # Here is the paragraph: {confirmed_text}
        # Please only return the fixed paragraph without any additional information or explanation.
        # """

        # if tokenizer.chat_template is not None:
        #     messages = [{"role": "user", "content": prompt}]
        #     prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # response = generate(model, tokenizer, prompt=prompt)
        # print(f"[LLM] {response}")
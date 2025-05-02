import mlx_whisper
import time

def transcribe_file(file_path, model_path="mlx-community/whisper-small-mlx", verbose=False):
    start = time.perf_counter()
    result = mlx_whisper.transcribe(file_path, path_or_hf_repo=model_path)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    if verbose:
        return f"[{latency_ms:.2f} ms] {result['text']}"
    return result['text']

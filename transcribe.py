import mlx_whisper
import time

def transcribe_file(file_path, model_path="mlx-community/whisper-small-mlx", verbose=False):
    """
    Transcribe the audio file using MLX Whisper.
    :param file_path: Path to the audio file.
    :return: Transcription result.
    """
    start = time.perf_counter()

    result = mlx_whisper.transcribe("test.wav", path_or_hf_repo=model_path)

    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    if verbose:
        print(f"[{latency_ms:.2f} ms] {result['text']}")
    else:
        print(result['text'])
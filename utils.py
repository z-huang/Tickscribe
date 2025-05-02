import mlx_whisper
import time
from mlx_lm import load, stream_generate

def transcribe_file(file_path, model_path="mlx-community/whisper-small-mlx", verbose=False):
    start = time.perf_counter()
    result = mlx_whisper.transcribe(file_path, path_or_hf_repo=model_path)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    if verbose:
        return f"[{latency_ms:.2f} ms] {result['text']}"
    return result['text']


def summarize_text(text, model_path="mlx-community/Llama-3.2-1B-Instruct-4bit"):
    model, tokenizer = load(model_path)
    prompt = f"""
    Please summarize the following text in bullet lists.
    Here is the text: {text}
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
        print(response.text, end="", flush=True)
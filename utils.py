
from mlx_lm import load, stream_generate
from PySide6.QtCore import QFile
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QWidget


def load_ui_widget(path: str, parent: QWidget = None) -> QWidget:
    loader = QUiLoader()
    ui_file = QFile(path)
    if not ui_file.open(QFile.ReadOnly):
        raise FileNotFoundError(f"Cannot open UI file: {path}")
    widget = loader.load(ui_file, parent)
    ui_file.close()
    if widget is None:
        raise RuntimeError(f"Failed to load UI from: {path}")
    return widget


def summarize_text(text, model_path="mlx-community/Llama-3.2-1B-Instruct-4bit"):
    model, tokenizer = load(model_path)
    prompt = f"""
    Please summarize the following text in bullet lists.
    Here is the text: {text}
    """
    if tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)

    for response in stream_generate(model, tokenizer, prompt, max_tokens=2048):
        print(response.text, end="", flush=True)

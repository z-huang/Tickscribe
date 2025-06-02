
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


BAD_WORDS = {
    'shit',
    'fuck',
}


def clean_str(s):
    processed_words = []
    for word in s.split():
        for bad_word in BAD_WORDS:
            if bad_word in word.lower():
                processed_words.append('*' * len(word))
                break
        else:
            processed_words.append(word)
    return ' '.join(processed_words)

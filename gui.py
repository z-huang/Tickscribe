import sys
import threading
from utils import transcribe_file
import re
import os
import sqlite3
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QFileDialog, QListWidgetItem, QInputDialog,
    QMessageBox
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import (
    Qt, QFile, QObject, QThread,
    Signal, Slot, QTimer, QMetaObject
)
from RealtimeSTT import AudioToTextRecorder


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


class Database:
    def __init__(self, db_path='transcripts.db'):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        conn = self.get_connection()
        cursor = conn.cursor()

        # Create sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create transcripts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
        ''')

        conn.commit()
        conn.close()

    def get_all_sessions(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, name, created_at FROM sessions ORDER BY created_at DESC')
        sessions = cursor.fetchall()
        conn.close()
        return sessions

    def create_session(self, name):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO sessions (name) VALUES (?)', (name,))
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return session_id
        except sqlite3.IntegrityError:
            # Handle duplicate session name
            return None

    def get_session_id_by_name(self, name):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM sessions WHERE name = ?', (name,))
        result = cursor.fetchone()
        conn.close()
        return result['id'] if result else None

    def get_session_name_by_id(self, session_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT name FROM sessions WHERE id = ?', (session_id,))
        result = cursor.fetchone()
        conn.close()
        return result['name'] if result else None

    def add_transcript(self, session_id, text):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO transcripts (session_id, text) VALUES (?, ?)',
            (session_id, text)
        )
        conn.commit()
        conn.close()

    def get_transcripts_by_session_id(self, session_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT id, text, timestamp FROM transcripts WHERE session_id = ? ORDER BY timestamp',
            (session_id,)
        )
        transcripts = cursor.fetchall()
        conn.close()
        return transcripts

    def delete_session(self, session_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        conn.commit()
        conn.close()

    def rename_session(self, session_id, new_name):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE sessions SET name = ? WHERE id = ?', (new_name, session_id))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            # Handle duplicate session name
            return False


class TranscriptionWorker(QObject):
    stabilized = Signal(str)
    finished = Signal()

    def __init__(self, recorder):
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
        text = transcribe_file(
            self.file_path,
            model_path='mlx-community/whisper-small-mlx'
        )
        self.transcription_completed.emit(text)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = load_ui_widget('ui/mainwindow.ui', self)
        self.setCentralWidget(self.ui)

        # Initialize SQLite database
        self.db = Database()

        # Current session ID
        self.current_session_id = None

        # Transcriber
        self.recorder = AudioToTextRecorder(
            model='base',
            language='en',
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.on_realtime_transcription_update,
            realtime_model_type='base',
            spinner=False,
            compute_type="float32",
            no_log_file=True,
        )

        self.chat_lock = threading.Lock()
        self.transcribe_thread = None
        self.transcribe_worker = None
        self.upload_thread = None
        self.upload_worker = None
        self.is_recording = False

        self.update_buffer = None
        self.update_timer = QTimer()
        self.update_timer.setInterval(300)
        self.update_timer.timeout.connect(self.flush_update_buffer)
        self.update_timer.setSingleShot(True)

        # 介面按鈕連接
        self.ui.newChatButton.clicked.connect(self.new_chat)
        self.ui.recordButton.clicked.connect(self.toggle_recording)
        self.ui.recordButton.setText("Start Recording")
        self.ui.uploadButton.clicked.connect(self.open_and_transcribe)

        # Add context menu for the chat list
        self.ui.chatList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.chatList.customContextMenuRequested.connect(
            self.show_chat_context_menu)

        self.ui.chatContent.setWordWrap(True)

        # 左側清單：點選時載入對應的 transcript
        self.ui.chatList.itemClicked.connect(self.load_transcript)
        self.load_session_list()

    def show_chat_context_menu(self, position):
        from PySide6.QtWidgets import QMenu, QAction

        if not self.ui.chatList.currentItem():
            return

        menu = QMenu()
        rename_action = QAction("Rename", self)
        delete_action = QAction("Delete", self)

        rename_action.triggered.connect(self.rename_current_session)
        delete_action.triggered.connect(self.delete_current_session)

        menu.addAction(rename_action)
        menu.addAction(delete_action)

        menu.exec_(self.ui.chatList.mapToGlobal(position))

    def rename_current_session(self):
        if not self.ui.chatList.currentItem():
            return

        current_name = self.ui.chatList.currentItem().text()
        session_id = self.db.get_session_id_by_name(current_name)

        if not session_id:
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Chat", "Enter new name:", text=current_name)

        if ok and new_name.strip() and new_name != current_name:
            success = self.db.rename_session(session_id, new_name)
            if success:
                self.load_session_list()
                # Select the renamed session
                items = self.ui.chatList.findItems(new_name, Qt.MatchExactly)
                if items:
                    self.ui.chatList.setCurrentItem(items[0])
            else:
                QMessageBox.warning(self, "Rename Failed",
                                    "A session with that name already exists.")

    def delete_current_session(self):
        if not self.ui.chatList.currentItem():
            return

        current_name = self.ui.chatList.currentItem().text()
        session_id = self.db.get_session_id_by_name(current_name)

        if not session_id:
            return

        confirm = QMessageBox.question(
            self, "Delete Chat",
            f"Are you sure you want to delete '{current_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if confirm == QMessageBox.Yes:
            self.db.delete_session(session_id)
            self.ui.chatContent.clear()
            self.current_session_id = None
            self.load_session_list()

    def new_chat(self):
        # Get a new session name
        name, ok = QInputDialog.getText(self, "New Chat", "Enter chat name:")
        if not ok or not name.strip():
            return

        # Create new session in database
        session_id = self.db.create_session(name.strip())
        if not session_id:
            # Handle duplicate name
            QMessageBox.warning(
                self, "Error", "A chat with that name already exists.")
            return

        # Reload the session list and switch to the new session
        self.load_session_list()
        items = self.ui.chatList.findItems(name.strip(), Qt.MatchExactly)
        if items:
            self.ui.chatList.setCurrentItem(items[0])
            self.load_transcript(items[0])

    def load_session_list(self):
        """Load all sessions from the database into the list widget"""
        self.ui.chatList.clear()
        sessions = self.db.get_all_sessions()
        for session in sessions:
            self.ui.chatList.addItem(session['name'])

    def load_transcript(self, item):
        """Load the transcripts for the selected session"""
        session_name = item.text()
        session_id = self.db.get_session_id_by_name(session_name)

        if not session_id:
            return

        self.current_session_id = session_id
        self.ui.chatContent.clear()

        transcripts = self.db.get_transcripts_by_session_id(session_id)
        for transcript in transcripts:
            self.ui.chatContent.addItem(transcript['text'])

    def toggle_recording(self):
        if not self.is_recording:
            if not self.current_session_id:
                QMessageBox.warning(self, "No Chat Selected",
                                    "Please create or select a chat first.")
                return
            self.start_recording()
        else:
            self.stop_recording()

    def on_realtime_transcription_update(self, s):
        self.update_buffer = s
        QMetaObject.invokeMethod(
            self.update_timer, "start", Qt.QueuedConnection)

    @Slot()
    def flush_update_buffer(self):
        if self.update_buffer is None:
            self.update_timer.stop()
            return

        with self.chat_lock:
            s = self.update_buffer
            self.update_buffer = None

            if (
                self.ui.chatContent.count() > 0 and
                self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp'
            ):
                item = self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1)
                item.setText(s)
            else:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, 'temp')
                self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

        self.update_timer.stop()

    @Slot(str)
    def on_realtime_transcription_stabilized(self, s):
        if not self.current_session_id:
            return

        with self.chat_lock:
            # Remove temporary item
            if (
                self.ui.chatContent.count() > 0 and
                self.ui.chatContent.item(
                    self.ui.chatContent.count() - 1).data(Qt.UserRole) == 'temp'
            ):
                self.ui.chatContent.takeItem(self.ui.chatContent.count() - 1)

            # Add final text to UI
            item = QListWidgetItem(s)
            self.ui.chatContent.addItem(item)
            self.ui.chatContent.scrollToBottom()

            # Save transcript to database
            self.db.add_transcript(self.current_session_id, s)

    def start_recording(self):
        self.recorder.start()

        self.transcribe_thread = QThread()
        self.transcribe_worker = TranscriptionWorker(self.recorder)

        self.transcribe_worker.moveToThread(self.transcribe_thread)
        self.transcribe_worker.stabilized.connect(
            self.on_realtime_transcription_stabilized)
        self.transcribe_thread.started.connect(self.transcribe_worker.run)
        self.transcribe_worker.finished.connect(self.transcribe_thread.quit)
        self.transcribe_worker.finished.connect(
            self.transcribe_worker.deleteLater)
        self.transcribe_thread.finished.connect(
            self.transcribe_thread.deleteLater)

        self.transcribe_thread.start()

        self.is_recording = True
        self.ui.recordButton.setText("Stop Recording")

    def stop_recording(self):
        self.recorder.stop()
        if self.transcribe_worker:
            self.transcribe_worker.stop()

        self.is_recording = False
        self.ui.recordButton.setText("Start Recording")

    def open_and_transcribe(self):
        if not self.current_session_id:
            QMessageBox.warning(self, "No Chat Selected",
                                "Please create or select a chat first.")
            return

        # Open file dialog to select a .wav file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇音訊或影片檔",
            "",
            "Media Files (*.wav *.mp3 *.mp4);;Audio (*.wav *.mp3);;Video (*.mp4)"
        )
        if not file_path:
            return

        self.ui.statusbar.showMessage("Transcription in progress...")

        self.upload_thread = QThread()
        self.upload_worker = FileTranscriptionWorker(file_path)

        self.upload_worker.moveToThread(self.upload_thread)
        self.upload_worker.transcription_completed.connect(
            self.on_file_transcription_completed)
        self.upload_thread.started.connect(self.upload_worker.run)
        self.upload_worker.finished.connect(self.upload_thread.quit)
        self.upload_worker.finished.connect(self.upload_worker.deleteLater)
        self.upload_thread.finished.connect(self.upload_thread.deleteLater)

        self.upload_thread.start()

    @Slot(str)
    def on_file_transcription_completed(self, text):
        if not self.current_session_id:
            return

        sentences = re.split(r'(?<=[。！？\.\!?])\s*', text)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                self.ui.chatContent.addItem(sent)
                # Save each sentence to database
                self.db.add_transcript(self.current_session_id, sent)

        self.ui.statusbar.showMessage("Transcription completed.", 2000)

    def closeEvent(self, event):
        print('close')
        self.stop_recording()
        self.recorder.shutdown()

        if self.transcribe_thread and self.transcribe_thread.isRunning():
            self.transcribe_worker.stop()
            self.transcribe_thread.quit()
            self.transcribe_thread.wait()

        if self.upload_thread and self.upload_thread.isRunning():
            self.upload_worker.stop()
            self.upload_thread.quit()
            self.upload_thread.wait()

        self.update_timer.stop()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.aboutToQuit.connect(lambda: print("App is quitting"))
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())

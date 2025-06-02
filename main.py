import re
import sys
import threading

from PySide6.QtCore import QMetaObject, Qt, QThread, QTimer, Slot
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QFileDialog, QInputDialog,
                               QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox)
from RealtimeSTT import AudioToTextRecorder

from database import Database
from utils import clean_str, load_ui_widget
from workers import FileTranscriptionWorker, LLMWorker, TranscriptionWorker


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tickscribe")
        self.resize(800, 600)
        # Load UI from .ui file
        self.ui = load_ui_widget("ui/mainwindow.ui", self)
        self.setCentralWidget(self.ui)

        # Initialize SQLite database
        self.db = Database()

        # Current session ID
        self.current_session_id = None

        # Initialize real-time audio-to-text recorder
        self.recorder = AudioToTextRecorder(
            model="base",
            language="en",
            enable_realtime_transcription=True,
            on_realtime_transcription_update=self.on_realtime_transcription_update,
            realtime_model_type="base",
            spinner=False,
            compute_type="float32",
            no_log_file=True,
        )

        # Threading and state variables
        self.chat_lock = threading.Lock()
        self.transcribe_thread = None
        self.transcribe_worker = None
        self.upload_thread = None
        self.upload_worker = None
        self.is_recording = False

        # Buffer and timer for UI updates
        self.update_buffer = None
        self.update_timer = QTimer()
        self.update_timer.setInterval(300)
        self.update_timer.timeout.connect(self.flush_update_buffer)
        self.update_timer.setSingleShot(True)

        # Connect UI buttons to their handlers
        self.ui.newChatButton.clicked.connect(self.new_chat)
        self.ui.recordButton.clicked.connect(self.toggle_recording)
        self.ui.recordButton.setText("Start Recording")
        self.ui.uploadButton.clicked.connect(self.open_and_transcribe)

        # Add context menu for the chat list
        self.ui.chatList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ui.chatList.customContextMenuRequested.connect(
            self.show_chat_context_menu)

        # Enable word wrap for text widgets
        self.ui.transcribeContent.setWordWrap(True)
        self.ui.llmChatList.setWordWrap(True)

        # Load transcript when a chat is selected
        self.ui.chatList.itemClicked.connect(self.load_transcript)

        # LLM (Large Language Model) chat variables and connections
        self.llm_messages = []
        self.partial_response = ''
        self.ui.summaryButton.clicked.connect(self.summarize)
        self.ui.clearChatButton.clicked.connect(self.clear_chat)
        self.ui.sendButton.clicked.connect(self.send_message)
        self.ui.chatLineEdit.returnPressed.connect(self.send_message)
        self.llm_worker_thread = None

        # Load all chat sessions on startup
        self.load_session_list()

    def load_session_list(self):
        """Load all sessions from the database into the list widget"""
        self.ui.chatList.clear()
        sessions = self.db.get_all_sessions()
        for session in sessions:
            self.ui.chatList.addItem(session["name"])

    def new_chat(self):
        """Create a new chat session"""
        # Get a new session name from user input
        name, ok = QInputDialog.getText(self, "New Chat", "Enter chat name:")
        if not ok or not name.strip():
            return
        try:
            # Create new session in database
            session_id = self.db.create_session(name.strip())
        except Exception as e:
            QMessageBox.critical(self, "Database Error",
                                 f"Failed to create chat:\n{e}")
            return

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

    def show_chat_context_menu(self, position):
        """Show context menu for renaming or deleting a chat session"""
        if not self.ui.chatList.currentItem():
            return

        menu = QMenu()
        rename_action = QAction("Rename", self)
        delete_action = QAction("Delete", self)

        rename_action.triggered.connect(self.rename_current_session)
        delete_action.triggered.connect(self.delete_current_session)

        menu.addAction(rename_action)
        menu.addAction(delete_action)

        menu.exec(self.ui.chatList.mapToGlobal(position))

    def rename_current_session(self):
        """Rename the currently selected chat session"""
        if not self.ui.chatList.currentItem():
            return

        current_name = self.ui.chatList.currentItem().text()
        session_id = self.db.get_session_id_by_name(current_name)

        if not session_id:
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Chat", "Enter new name:", text=current_name
        )

        if ok and new_name.strip() and new_name != current_name:
            success = self.db.rename_session(session_id, new_name)
            if success:
                self.load_session_list()
                # Select the renamed session
                items = self.ui.chatList.findItems(new_name, Qt.MatchExactly)
                if items:
                    self.ui.chatList.setCurrentItem(items[0])
            else:
                QMessageBox.warning(
                    self, "Rename Failed", "A session with that name already exists."
                )

    def delete_current_session(self):
        """Delete the currently selected chat session"""
        if not self.ui.chatList.currentItem():
            return

        current_name = self.ui.chatList.currentItem().text()
        session_id = self.db.get_session_id_by_name(current_name)

        if not session_id:
            return

        confirm = QMessageBox.question(
            self,
            "Delete Chat",
            f"Are you sure you want to delete '{current_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if confirm == QMessageBox.Yes:
            self.db.delete_session(session_id)
            self.ui.transcribeContent.clear()
            self.current_session_id = None
            self.load_session_list()

    def load_transcript(self, item):
        """Load the transcripts for the selected session"""
        session_name = item.text()
        session_id = self.db.get_session_id_by_name(session_name)

        if not session_id:
            return

        self.current_session_id = session_id
        self.ui.transcribeContent.clear()

        transcripts = self.db.get_transcripts_by_session_id(session_id)
        for transcript in transcripts:
            self.ui.transcribeContent.addItem(transcript["text"])

    def toggle_recording(self):
        """Start or stop audio recording and transcription"""
        if not self.is_recording:
            if not self.current_session_id:
                QMessageBox.warning(
                    self, "No Chat Selected", "Please create or select a chat first."
                )
                return
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start real-time audio recording and transcription"""
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
        """Stop audio recording and transcription"""
        self.recorder.stop()
        if self.transcribe_worker:
            self.transcribe_worker.stop()

        self.is_recording = False
        self.ui.recordButton.setText("Start Recording")

    def on_realtime_transcription_update(self, s):
        """Handle real-time transcription updates (partial results)"""
        s = clean_str(s)
        self.update_buffer = s
        QMetaObject.invokeMethod(
            self.update_timer, "start", Qt.QueuedConnection)

    @Slot(str)
    def on_realtime_transcription_stabilized(self, s):
        """Handle stabilized (finalized) transcription results"""
        if not self.current_session_id:
            return

        with self.chat_lock:
            # Remove temporary item if present
            if (
                self.ui.transcribeContent.count() > 0
                and self.ui.transcribeContent.item(self.ui.transcribeContent.count() - 1).data(
                    Qt.UserRole
                )
                == "temp"
            ):
                self.ui.transcribeContent.takeItem(
                    self.ui.transcribeContent.count() - 1)

            # Add final text to UI
            item = QListWidgetItem(s)
            self.ui.transcribeContent.addItem(item)
            self.ui.transcribeContent.scrollToBottom()

            # Save transcript to database
            self.db.add_transcript(self.current_session_id, s)

    @Slot()
    def flush_update_buffer(self):
        """Flush the update buffer to the UI (for partial transcription)"""
        if self.update_buffer is None:
            self.update_timer.stop()
            return

        with self.chat_lock:
            s = self.update_buffer
            self.update_buffer = None

            # Update or add temporary item for partial transcription
            if (
                self.ui.transcribeContent.count() > 0
                and self.ui.transcribeContent.item(self.ui.transcribeContent.count() - 1).data(
                    Qt.UserRole
                )
                == "temp"
            ):
                item = self.ui.transcribeContent.item(
                    self.ui.transcribeContent.count() - 1)
                item.setText(s)
            else:
                item = QListWidgetItem(s)
                item.setData(Qt.UserRole, "temp")
                self.ui.transcribeContent.addItem(item)
            self.ui.transcribeContent.scrollToBottom()

        self.update_timer.stop()

    def open_and_transcribe(self):
        """Open a file dialog and transcribe the selected audio/video file"""
        if not self.current_session_id:
            QMessageBox.warning(
                self, "No Chat Selected", "Please create or select a chat first."
            )
            return

        # Open file dialog to select a .wav, .mp3, or .mp4 file
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Audio or Video File",
            "",
            "Media Files (*.wav *.mp3 *.mp4);;Audio (*.wav *.mp3);;Video (*.mp4)",
        )
        if not file_path:
            return

        self.statusBar().showMessage("Transcription in progress...")

        self.upload_thread = QThread()
        self.upload_worker = FileTranscriptionWorker(file_path)

        self.upload_worker.moveToThread(self.upload_thread)
        self.upload_worker.transcription_completed.connect(
            self.on_file_transcription_completed
        )
        self.upload_thread.started.connect(self.upload_worker.run)
        self.upload_worker.finished.connect(self.upload_thread.quit)
        self.upload_worker.finished.connect(self.upload_worker.deleteLater)
        self.upload_thread.finished.connect(self.upload_thread.deleteLater)

        self.upload_thread.start()

    @Slot(str)
    def on_file_transcription_completed(self, text):
        """Handle completion of file transcription"""
        if not self.current_session_id:
            return

        # Split text into sentences for display and storage
        sentences = re.split(r"(?<=[。！？\.\!?])\s*", text)

        for sent in sentences:
            sent = sent.strip()
            if sent:
                self.ui.transcribeContent.addItem(sent)
                # Save each sentence to database
                self.db.add_transcript(self.current_session_id, sent)

        self.statusBar().showMessage("Transcription completed.", 2000)

    @Slot()
    def summarize(self):
        """Send a summarize command to the LLM chat"""
        self.ui.rightPane.setCurrentWidget(self.ui.tabChat)
        self.ui.chatLineEdit.setText("Summarize")
        self.send_message()

    def clear_chat(self):
        """Clear the LLM chat history"""
        self.llm_messages.clear()
        self.ui.llmChatList.clear()

    def send_message(self):
        """Send a message to the LLM and handle the response"""
        if not self.ui.sendButton.isEnabled():
            return
        user_text = self.ui.chatLineEdit.text().strip()
        if user_text == '':
            return

        self.llm_messages.append({
            'role': 'user',
            'content': user_text
        })
        # Gather all current transcriptions for context
        current_transcriptions = [
            self.ui.transcribeContent.item(i).text()
            for i in range(self.ui.transcribeContent.count())
        ]
        transcription_text = "\n".join(current_transcriptions)
        self.ui.chatLineEdit.clear()
        self.ui.sendButton.setEnabled(False)
        self.ui.llmChatList.addItem(f"[User] {user_text}")
        self.ui.llmChatList.scrollToBottom()

        # Start LLM worker thread for response generation
        self.llm_worker_thread = QThread()
        self.llm_worker = LLMWorker(
            messages=[
                {
                    'role': 'system',
                    'content': 'You are an AI assistant helping users with transcriptions. Answer questions, summarize transcripts, and provide helpful suggestions related to audio or text transcriptions.'
                },
                {
                    'role': 'user',
                    'content': f'''\
# Transcription
{transcription_text}
'''
                }
            ] + self.llm_messages)
        self.llm_worker.moveToThread(self.llm_worker_thread)
        self.llm_worker.token_received.connect(self.append_token)
        self.llm_worker.finished.connect(self.query_finished)
        self.llm_worker_thread.started.connect(self.llm_worker.run)
        self.llm_worker.finished.connect(self.llm_worker_thread.quit)
        self.llm_worker.finished.connect(self.llm_worker.deleteLater)
        self.llm_worker_thread.finished.connect(
            self.llm_worker_thread.deleteLater)
        self.llm_worker_thread.start()

    @Slot(str)
    def append_token(self, token):
        """Append a token from the LLM to the chat UI (streaming output)"""
        self.partial_response += token

        scrollbar = self.ui.llmChatList.verticalScrollBar()
        at_bottom = scrollbar.value() == scrollbar.maximum()

        # If assistant message not yet shown, add a new item
        if (self.ui.llmChatList.count() == 0 or
                not self.ui.llmChatList.item(self.ui.llmChatList.count() - 1).data(0).startswith("[Assistant]")):
            self.ui.llmChatList.addItem(f"[Assistant] {self.partial_response}")
        else:
            # Update the last item with the current accumulated tokens
            last_item = self.ui.llmChatList.item(
                self.ui.llmChatList.count() - 1)
            last_item.setText(f"[Assistant] {self.partial_response}")

        if at_bottom:
            self.ui.llmChatList.scrollToBottom()

    @Slot()
    def query_finished(self):
        """Handle completion of LLM response"""
        self.llm_messages.append({
            'role': 'assistant',
            'content': self.partial_response
        })
        self.partial_response = ''
        self.ui.sendButton.setEnabled(True)

    def closeEvent(self, event):
        """Handle application close event (cleanup resources)"""
        self.stop_recording()
        self.recorder.shutdown()
        self.update_timer.stop()
        if self.llm_worker_thread and hasattr(self.llm_worker_thread, 'stop'):
            self.llm_worker_thread.stop()
        event.accept()


def main():
    """Main entry point for the application"""
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

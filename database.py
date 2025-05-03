from PySide6.QtSql import QSqlDatabase, QSqlQuery


class Database:
    def __init__(self, db_path='transcripts.db'):
        self.db_path = db_path
        self.db = QSqlDatabase.addDatabase("QSQLITE")
        self.db.setDatabaseName(self.db_path)
        if not self.db.open():
            raise Exception("Failed to open database")
        self.init_db()

    def init_db(self):
        query = QSqlQuery()

        query.exec(
            '''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )

        query.exec(
            '''
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
            '''
        )

    def get_all_sessions(self):
        sessions = []
        query = QSqlQuery(
            "SELECT id, name, created_at FROM sessions ORDER BY created_at DESC")
        while query.next():
            sessions.append({
                'id': query.value(0),
                'name': query.value(1),
                'created_at': query.value(2)
            })
        return sessions

    def create_session(self, name):
        query = QSqlQuery()
        query.prepare("INSERT INTO sessions (name) VALUES (:name)")
        query.bindValue(":name", name)
        if not query.exec():
            return None
        return query.lastInsertId()

    def get_session_id_by_name(self, name):
        query = QSqlQuery()
        query.prepare("SELECT id FROM sessions WHERE name = :name")
        query.bindValue(":name", name)
        query.exec()
        if query.next():
            return query.value(0)
        return None

    def get_session_name_by_id(self, session_id):
        query = QSqlQuery()
        query.prepare("SELECT name FROM sessions WHERE id = :id")
        query.bindValue(":id", session_id)
        query.exec()
        if query.next():
            return query.value(0)
        return None

    def delete_session(self, session_id):
        query = QSqlQuery()
        query.prepare("DELETE FROM sessions WHERE id = :id")
        query.bindValue(":id", session_id)
        query.exec()

    def rename_session(self, session_id, new_name):
        query = QSqlQuery()
        query.prepare("UPDATE sessions SET name = :name WHERE id = :id")
        query.bindValue(":name", new_name)
        query.bindValue(":id", session_id)
        return query.exec()

    def add_transcript(self, session_id, text):
        query = QSqlQuery()
        query.prepare('''
            INSERT INTO transcripts (session_id, text)
            VALUES (:session_id, :text)
        ''')
        query.bindValue(":session_id", session_id)
        query.bindValue(":text", text)
        query.exec()

    def get_transcripts_by_session_id(self, session_id):
        transcripts = []
        query = QSqlQuery()
        query.prepare('''
            SELECT id, text, timestamp FROM transcripts
            WHERE session_id = :session_id
            ORDER BY timestamp
        ''')
        query.bindValue(":session_id", session_id)
        query.exec()
        while query.next():
            transcripts.append({
                'id': query.value(0),
                'text': query.value(1),
                'timestamp': query.value(2)
            })
        return transcripts

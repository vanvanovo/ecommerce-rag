"""MySQL 客户端：FAQ 高频问答表 + 会话历史表。"""
import pymysql
from base.config import config
from base.logger import logger


class MySQLClient:
    def __init__(self):
        self.connection = pymysql.connect(
            host=config.MYSQL_HOST, port=config.MYSQL_PORT,
            user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
            db=config.MYSQL_DATABASE, charset='utf8mb4',
        )
        self.cursor = self.connection.cursor()
        logger.info("MySQL 连接成功")
        self.init_tables()

    def init_tables(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS faq (
                id INT AUTO_INCREMENT PRIMARY KEY,
                source VARCHAR(50),           -- 来源：手机/售后政策…
                question VARCHAR(1000),
                answer VARCHAR(3000),
                tag VARCHAR(100),             -- 价格/保修/退换货…
                updated_at DATETIME DEFAULT NOW(),
                INDEX idx_source (source)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id VARCHAR(36) NOT NULL,
                question TEXT, answer TEXT,
                timestamp DATETIME NOT NULL,
                INDEX idx_session (session_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """)
        self.connection.commit()

    def fetch_all_questions(self):
        """BM25 索引构建：全量问题列表。"""
        self.cursor.execute("SELECT question FROM faq")
        return [row[0] for row in self.cursor.fetchall()]

    def fetch_answer(self, question):
        self.cursor.execute("SELECT answer FROM faq WHERE question=%s", (question,))
        row = self.cursor.fetchone()
        return row[0] if row else None

    def insert_faq(self, source, question, answer, tag=""):
        self.cursor.execute(
            "INSERT INTO faq (source, question, answer, tag) VALUES (%s,%s,%s,%s)",
            (source, question, answer, tag))
        self.connection.commit()

    def insert_history(self, session_id, question, answer):
        self.cursor.execute(
            "INSERT INTO conversations (session_id, question, answer, timestamp) VALUES (%s,%s,%s,NOW())",
            (session_id, question, answer))

    def fetch_recent_history(self, session_id, limit=5):
        self.cursor.execute("""
            SELECT question, answer FROM conversations
            WHERE session_id=%s ORDER BY timestamp DESC LIMIT %s
        """, (session_id, limit))
        return [{"question": r[0], "answer": r[1]} for r in self.cursor.fetchall()][::-1]

    def keep_recent_only(self, session_id, keep=5):
        self.cursor.execute("""
            DELETE FROM conversations WHERE session_id=%s AND id NOT IN (
                SELECT id FROM (SELECT id FROM conversations WHERE session_id=%s
                                ORDER BY timestamp DESC LIMIT %s) t)
        """, (session_id, session_id, keep))

    def commit(self):
        self.connection.commit()

    def close(self):
        try:
            self.connection.close()
        except Exception:
            pass
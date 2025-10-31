"""
Document tracking service for managing document lifecycle.
"""

from typing import Dict, Any, List, Optional
import sqlite3
from pathlib import Path
from utils.logger import get_logger
import datetime

logger = get_logger("document_tracking_service")


class DocumentTrackingService:
    """Service for tracking document ingestion and status"""

    def __init__(self, db_path: str = "document_tracking.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE NOT NULL,
                    file_path TEXT,
                    file_hash TEXT,
                    ingestion_time TEXT,
                    chunk_count INTEGER,
                    status TEXT,
                    metadata TEXT
                )
            """)

            conn.commit()
            conn.close()

            logger.info("Document tracking database initialized")

        except Exception as e:
            logger.error(f"Failed to initialize tracking database: {str(e)}")

    def track_document(
        self,
        filename: str,
        file_path: Optional[str] = None,
        file_hash: Optional[str] = None,
        chunk_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Track a document ingestion"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (filename, file_path, file_hash, ingestion_time, chunk_count, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                filename,
                file_path,
                file_hash,
                datetime.datetime.utcnow().isoformat(),
                chunk_count,
                "ingested",
                str(metadata) if metadata else None,
            ))

            conn.commit()
            conn.close()

            logger.info(f"Document tracked: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to track document: {str(e)}")
            return False

    def get_document_status(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get status of a tracked document"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT * FROM documents WHERE filename = ?
            """, (filename,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "id": row[0],
                    "filename": row[1],
                    "file_path": row[2],
                    "file_hash": row[3],
                    "ingestion_time": row[4],
                    "chunk_count": row[5],
                    "status": row[6],
                    "metadata": row[7],
                }

            return None

        except Exception as e:
            logger.error(f"Failed to get document status: {str(e)}")
            return None

    def list_tracked_documents(self) -> List[Dict[str, Any]]:
        """List all tracked documents"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM documents ORDER BY ingestion_time DESC")

            rows = cursor.fetchall()
            conn.close()

            documents = []
            for row in rows:
                documents.append({
                    "id": row[0],
                    "filename": row[1],
                    "file_path": row[2],
                    "file_hash": row[3],
                    "ingestion_time": row[4],
                    "chunk_count": row[5],
                    "status": row[6],
                    "metadata": row[7],
                })

            return documents

        except Exception as e:
            logger.error(f"Failed to list tracked documents: {str(e)}")
            return []

    def is_document_ingested(self, filename: str, file_hash: Optional[str] = None) -> bool:
        """Check if document is already ingested"""
        status = self.get_document_status(filename)

        if not status:
            return False

        if file_hash and status.get("file_hash") != file_hash:
            return False

        return status.get("status") == "ingested"


# Global instance
document_tracking_service = DocumentTrackingService()

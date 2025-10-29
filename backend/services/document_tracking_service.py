import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
from utils.logger import get_logger

# Use enhanced logger
logger = get_logger("document_tracking_service")
import threading


class DocumentTrackingService:
    """Service for tracking uploaded documents per user to prevent duplicate indexing"""

    def __init__(self, db_path: str = "document_tracking.db"):
        """Initialize the document tracking service with SQLite database"""
        # Store database in backend directory
        backend_dir = Path(__file__).parent.parent
        self.db_path = str(backend_dir / db_path)
        self._lock = threading.Lock()
        self._initialize_database()

    def _initialize_database(self):
        """Create the documents table if it doesn't exist"""
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Create documents table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_hash TEXT NOT NULL,
                        file_size INTEGER,
                        upload_date TEXT NOT NULL,
                        index_status TEXT DEFAULT 'indexed',
                        chunk_count INTEGER DEFAULT 0,
                        UNIQUE(user_id, file_hash)
                    )
                """)

                # Create indexes for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_id 
                    ON documents(user_id)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_file_hash 
                    ON documents(file_hash)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_hash 
                    ON documents(user_id, file_hash)
                """)

                conn.commit()
                conn.close()

                logger.info(f"Document tracking database initialized: {self.db_path}")

        except Exception as e:
            logger.error(f"Failed to initialize document tracking database: {str(e)}")
            raise

    def check_document_exists(self, user_id: str, file_hash: str) -> Optional[Dict]:
        """
        Check if a document with the same hash already exists for this user

        Args:
            user_id: User identifier (token)
            file_hash: SHA256 hash of the file

        Returns:
            Document info dict if exists, None otherwise
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, filename, file_hash, file_size, upload_date, 
                           index_status, chunk_count
                    FROM documents 
                    WHERE user_id = ? AND file_hash = ?
                """,
                    (user_id, file_hash),
                )

                row = cursor.fetchone()
                conn.close()

                if row:
                    return {
                        "id": row[0],
                        "filename": row[1],
                        "file_hash": row[2],
                        "file_size": row[3],
                        "upload_date": row[4],
                        "index_status": row[5],
                        "chunk_count": row[6],
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to check document existence: {str(e)}")
            return None

    def add_document(
        self,
        user_id: str,
        filename: str,
        file_hash: str,
        file_size: int,
        chunk_count: int = 0,
    ) -> bool:
        """
        Add a new document to tracking

        Args:
            user_id: User identifier (token)
            filename: Original filename
            file_hash: SHA256 hash of the file
            file_size: File size in bytes
            chunk_count: Number of chunks indexed

        Returns:
            True if successful, False otherwise
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                upload_date = datetime.now().isoformat()

                # Use INSERT OR REPLACE to handle duplicates
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO documents 
                    (user_id, filename, file_hash, file_size, upload_date, 
                     index_status, chunk_count)
                    VALUES (?, ?, ?, ?, ?, 'indexed', ?)
                """,
                    (user_id, filename, file_hash, file_size, upload_date, chunk_count),
                )

                conn.commit()
                conn.close()

                logger.info("Document added to tracking")

                logger.debug(
                    f"Document details: user_id={user_id}, filename={filename}, hash={file_hash[:16]}, size={file_size}, chunks={chunk_count}"
                )
                return True

        except Exception as e:
            logger.error(f"Failed to add document to tracking: {str(e)}")
            return False

    def get_user_documents(self, user_id: str) -> List[Dict]:
        """
        Get all documents for a user

        Args:
            user_id: User identifier (token)

        Returns:
            List of document info dicts
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, filename, file_hash, file_size, upload_date, 
                           index_status, chunk_count
                    FROM documents 
                    WHERE user_id = ?
                    ORDER BY upload_date DESC
                """,
                    (user_id,),
                )

                rows = cursor.fetchall()
                conn.close()

                documents = []
                for row in rows:
                    documents.append(
                        {
                            "id": row[0],
                            "filename": row[1],
                            "file_hash": row[2],
                            "file_size": row[3],
                            "upload_date": row[4],
                            "index_status": row[5],
                            "chunk_count": row[6],
                        }
                    )

                return documents

        except Exception as e:
            logger.error(f"Failed to get user documents: {str(e)}")
            return []

    def update_chunk_count(
        self, user_id: str, file_hash: str, chunk_count: int
    ) -> bool:
        """
        Update the chunk count for a document

        Args:
            user_id: User identifier
            file_hash: File hash
            chunk_count: Number of chunks

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE documents 
                    SET chunk_count = ?
                    WHERE user_id = ? AND file_hash = ?
                """,
                    (chunk_count, user_id, file_hash),
                )

                conn.commit()
                conn.close()
                return True

        except Exception as e:
            logger.error(f"Failed to update chunk count: {str(e)}")
            return False

    def delete_document(self, user_id: str, file_hash: str) -> bool:
        """
        Delete a document from tracking

        Args:
            user_id: User identifier
            file_hash: File hash

        Returns:
            True if successful
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    DELETE FROM documents 
                    WHERE user_id = ? AND file_hash = ?
                """,
                    (user_id, file_hash),
                )

                conn.commit()
                conn.close()

                logger.info("Document deleted from tracking")
                logger.debug(
                    f"Deleted document details: user_id={user_id}, hash={file_hash[:16]}"
                )

                return True

        except Exception as e:
            logger.error(f"Failed to delete document: {str(e)}")
            return False

    def get_document_by_filename(self, user_id: str, filename: str) -> Optional[Dict]:
        """
        Get document info by filename for a specific user

        Args:
            user_id: User identifier
            filename: Filename to search for

        Returns:
            Document info dict if found, None otherwise
        """
        try:
            with self._lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute(
                    """
                    SELECT id, filename, file_hash, file_size, upload_date, 
                           index_status, chunk_count
                    FROM documents 
                    WHERE user_id = ? AND filename = ?
                    ORDER BY upload_date DESC
                    LIMIT 1
                """,
                    (user_id, filename),
                )

                row = cursor.fetchone()
                conn.close()

                if row:
                    return {
                        "id": row[0],
                        "filename": row[1],
                        "file_hash": row[2],
                        "file_size": row[3],
                        "upload_date": row[4],
                        "index_status": row[5],
                        "chunk_count": row[6],
                    }
                return None

        except Exception as e:
            logger.error(f"Failed to get document by filename: {str(e)}")
            return None


# Global instance
document_tracking_service = DocumentTrackingService()

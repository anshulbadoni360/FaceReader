"""Database setup and models."""

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

import sqlite3

DB_PATH = Path("data/jobs.db")
DB_PATH.parent.mkdir(exist_ok=True)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job:
    """Job model for tracking image analysis."""
    
    def __init__(
        self,
        job_id: str,
        filename: str,
        status: JobStatus = JobStatus.PENDING,
        created_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        self.job_id = job_id
        self.filename = filename
        self.status = status
        self.created_at = created_at or datetime.utcnow()
        self.completed_at = completed_at
        self.result = result
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }


class Database:
    """SQLite database for storing jobs."""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT
                )
            """)
            conn.commit()
    
    def create_job(self, job_id: str, filename: str) -> Job:
        """Create a new job."""
        now = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO jobs (job_id, filename, status, created_at)
                VALUES (?, ?, ?, ?)
            """, (job_id, filename, JobStatus.PENDING.value, now.isoformat()))
            conn.commit()
        
        return Job(job_id, filename, JobStatus.PENDING, now)
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
        
        if not row:
            return None
        
        return self._row_to_job(row)
    
    def update_job_status(self, job_id: str, status: JobStatus):
        """Update job status."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ? WHERE job_id = ?",
                (status.value, job_id)
            )
            conn.commit()
    
    def update_job_result(self, job_id: str, result: Dict[str, Any]):
        """Update job with result."""
        completed_at = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE jobs 
                SET status = ?, result = ?, completed_at = ?
                WHERE job_id = ?
            """, (
                JobStatus.COMPLETED.value,
                json.dumps(result),
                completed_at.isoformat(),
                job_id
            ))
            conn.commit()
    
    def update_job_error(self, job_id: str, error: str):
        """Update job with error."""
        completed_at = datetime.utcnow()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE jobs 
                SET status = ?, error = ?, completed_at = ?
                WHERE job_id = ?
            """, (
                JobStatus.FAILED.value,
                error,
                completed_at.isoformat(),
                job_id
            ))
            conn.commit()
    
    def _row_to_job(self, row) -> Job:
        """Convert database row to Job object."""
        (job_id, filename, status, created_at, completed_at, result, error) = row
        return Job(
            job_id=job_id,
            filename=filename,
            status=JobStatus(status),
            created_at=datetime.fromisoformat(created_at),
            completed_at=datetime.fromisoformat(completed_at) if completed_at else None,
            result=json.loads(result) if result else None,
            error=error,
        )


# Global database instance
db = Database()

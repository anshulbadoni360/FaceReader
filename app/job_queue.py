"""Job queue management."""

import queue
import threading
import uuid
from pathlib import Path
from typing import Optional

from app.database import db, JobStatus


class JobQueue:
    """Thread-safe job queue for processing images."""
    
    def __init__(self, max_size: int = 100):
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._lock = threading.Lock()
    
    def submit(self, filename: str, file_path: str) -> str:
        """Submit image file to queue, return job ID."""
        job_id = str(uuid.uuid4())
        
        # Create job in database
        db.create_job(job_id, filename)
        
        # Add to queue
        try:
            self._queue.put((job_id, file_path), timeout=5)
            return job_id
        except queue.Full:
            db.update_job_error(job_id, "Queue is full, try again later")
            raise Exception("Queue is full, try again later")
    
    def get(self, timeout: float = 1.0) -> Optional[tuple]:
        """Get next job from queue."""
        try:
            job_id, file_path = self._queue.get(timeout=timeout)
            return job_id, file_path
        except queue.Empty:
            return None
    
    def task_done(self):
        """Mark task as done."""
        self._queue.task_done()
    
    def size(self) -> int:
        """Get queue size."""
        return self._queue.qsize()


# Global job queue instance
job_queue = JobQueue()

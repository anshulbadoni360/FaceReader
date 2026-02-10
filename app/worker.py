"""Worker for processing jobs from queue."""

import logging
import threading
import time
from pathlib import Path

from app.database import db, JobStatus
from app.job_queue import job_queue
from app.services.analyzer import get_analyzer

logger = logging.getLogger(__name__)


class ImageAnalysisWorker:
    """Worker that processes images from the job queue."""
    
    def __init__(self, num_workers: int = 1):
        self.num_workers = num_workers
        self.threads = []
        self.running = False
        self.analyzer = None
    
    def start(self):
        """Start worker threads."""
        self.running = True
        self.analyzer = get_analyzer()
        
        for i in range(self.num_workers):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"ImageWorker-{i}",
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
        
        logger.info(f"Started {self.num_workers} worker threads")
    
    def stop(self):
        """Stop all worker threads."""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=5)
        logger.info("Stopped all worker threads")
    
    def _worker_loop(self):
        """Main worker loop that processes jobs."""
        while self.running:
            try:
                # Get next job from queue
                item = job_queue.get(timeout=1.0)
                if item is None:
                    continue
                
                job_id, file_path = item
                
                try:
                    # Update job status
                    db.update_job_status(job_id, JobStatus.PROCESSING)
                    
                    # Process image
                    result = self._process_image(file_path)
                    
                    # Save result to database
                    db.update_job_result(job_id, result)
                    logger.info(f"Job {job_id} completed successfully")
                
                except Exception as e:
                    # Save error to database
                    error_msg = str(e)
                    db.update_job_error(job_id, error_msg)
                    logger.error(f"Job {job_id} failed: {error_msg}")
                
                finally:
                    # Cleanup uploaded file after processing attempt
                    try:
                        path = Path(file_path)
                        if path.exists():
                            path.unlink()
                    except Exception as cleanup_error:
                        logger.warning(f"Failed to delete file {file_path}: {cleanup_error}")
                    job_queue.task_done()
            
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(0.1)
    
    def _process_image(self, file_path: str) -> dict:
        """Process single image and return result."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Analyze image using the analyzer
        result = self.analyzer.analyze(
            file_path,
            try_rotations=True
        )
        
        return result


# Global worker instance
_worker_instance = None


def get_worker() -> ImageAnalysisWorker:
    """Get or create worker instance."""
    global _worker_instance
    if _worker_instance is None:
        _worker_instance = ImageAnalysisWorker(num_workers=2)
    return _worker_instance

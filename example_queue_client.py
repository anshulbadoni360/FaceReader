#!/usr/bin/env python3
"""
Example: Using the Queue-Based Image Processing API

This script demonstrates how to:
1. Submit an image for processing
2. Poll for job completion
3. Retrieve analysis results
4. Handle errors
"""

import requests
import json
import time
import sys
from pathlib import Path


API_URL = "http://localhost:8000"
POLL_INTERVAL = 1  # seconds
MAX_WAIT = 300  # 5 minutes


class EmotionAPIClient:
    """Client for queue-based emotion analysis API."""
    
    def __init__(self, base_url: str = API_URL):
        self.base_url = base_url.rstrip("/")
    
    def submit_image(self, image_path: str) -> str:
        """
        Submit image for analysis.
        
        Args:
            image_path: Path to image file
        
        Returns:
            job_id: Unique job identifier
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/submit",
                files={"file": f}
            )
        
        if response.status_code != 200:
            raise Exception(f"Failed to submit image: {response.text}")
        
        data = response.json()
        return data["job_id"]
    
    def get_job_status(self, job_id: str) -> dict:
        """Get job status."""
        response = requests.get(f"{self.base_url}/job/{job_id}")
        
        if response.status_code == 404:
            raise Exception(f"Job not found: {job_id}")
        
        return response.json()
    
    def get_result(self, job_id: str) -> dict:
        """Get analysis result."""
        response = requests.get(f"{self.base_url}/result/{job_id}")
        
        if response.status_code == 404:
            raise Exception(f"Job not found: {job_id}")
        elif response.status_code == 202:
            raise Exception("Job still processing")
        elif response.status_code == 500:
            return {"error": response.json()}
        
        return response.json()
    
    def wait_for_result(
        self,
        job_id: str,
        timeout: int = MAX_WAIT,
        poll_interval: int = POLL_INTERVAL
    ) -> dict:
        """
        Wait for job completion and return result.
        
        Args:
            job_id: Job identifier
            timeout: Max wait time in seconds
            poll_interval: Polling interval in seconds
        
        Returns:
            Analysis result
        """
        start_time = time.time()
        
        while True:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            try:
                status = self.get_job_status(job_id)
                status_value = status["status"]
                
                print(f"[{elapsed:.1f}s] Status: {status_value}")
                
                if status_value == "completed":
                    return self.get_result(job_id)
                elif status_value == "failed":
                    raise Exception(f"Job failed: {status.get('error')}")
                
            except Exception as e:
                if "still processing" in str(e) or "pending" in str(e):
                    pass  # Continue polling
                else:
                    raise
            
            time.sleep(poll_interval)
    
    def analyze(self, image_path: str) -> dict:
        """
        Analyze image synchronously (submit + wait).
        
        Args:
            image_path: Path to image file
        
        Returns:
            Analysis result
        """
        print(f"Submitting {image_path}...")
        job_id = self.submit_image(image_path)
        print(f"Job ID: {job_id}")
        
        print("Waiting for analysis...")
        result = self.wait_for_result(job_id)
        return result


def main():
    """Main example."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze image emotions using queue-based API"
    )
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--url", default=API_URL, help="API base URL")
    parser.add_argument("--timeout", type=int, default=MAX_WAIT, 
                       help="Max wait time (seconds)")
    
    args = parser.parse_args()
    
    try:
        client = EmotionAPIClient(args.url)
        result = client.analyze(args.image)
        
        print("\n" + "="*60)
        print("EMOTION ANALYSIS RESULT")
        print("="*60)
        print(json.dumps(result, indent=2))
        
        # Summary
        if result.get("FaceAnalyzed"):
            emotions = result.get("FacialExpressions", {})
            if isinstance(emotions, dict):
                dominant = emotions.get("DominantBasicEmotion")
                print(f"\n✓ Dominant Emotion: {dominant}")
        else:
            print("\n✗ No face detected in image")
        
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

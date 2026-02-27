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


API_URL = "http://localhost:9090"
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
        
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
            
        with open(image_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/submit",
                files={"file": (image_path.name, f, mime_type)}
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
    import itertools
    
    parser = argparse.ArgumentParser(
        description="Analyze image emotions using queue-based API"
    )
    parser.add_argument("path", help="Path to image file or folder containing images")
    parser.add_argument("--url", default=API_URL, help="API base URL")
    parser.add_argument("--timeout", type=int, default=MAX_WAIT, 
                       help="Max wait time (seconds)")
    parser.add_argument("--max", type=int, default=100, 
                       help="Max number of images to process if a folder is provided")
    
    args = parser.parse_args()
    
    try:
        client = EmotionAPIClient(args.url)
        input_path = Path(args.path)
        
        if input_path.is_file():
            # Process a single image
            result = client.analyze(str(input_path))
            print("\n" + "="*60)
            print("EMOTION ANALYSIS RESULT")
            print("="*60)
            print(json.dumps(result, indent=2))
            
            if result.get("FaceAnalyzed"):
                emotions = result.get("FacialExpressions", {})
                if isinstance(emotions, dict):
                    dominant = emotions.get("DominantBasicEmotion")
                    print(f"\n✓ Dominant Emotion: {dominant}")
            else:
                print("\n✗ No face detected in image")
                
        elif input_path.is_dir():
            # Process up to N images in a directory
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
            all_files = [f for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
            selected_files = all_files[:args.max]
            
            if not selected_files:
                print(f"No valid images found in {input_path}")
                return 0
                
            print(f"Found {len(all_files)} images. Submitting {len(selected_files)} to the queue...")
            
            # 1. Submit them all as fast as possible
            job_dict = {}  # Map job_id to filename
            for img_path in selected_files:
                try:
                    job_id = client.submit_image(str(img_path))
                    job_dict[job_id] = {"file": img_path.name, "status": "pending"}
                    print(f"Submitted {img_path.name} -> ID: {job_id}")
                except Exception as e:
                    print(f"Failed to submit {img_path.name}: {e}")
            
            print(f"\nSuccessfully submitted {len(job_dict)} images.")
            print("Now polling for results (this may take a while depending on queue load)...\n")
            
            # 2. Loop until all are done
            start_time = time.time()
            completed = 0
            
            while completed < len(job_dict):
                elapsed = time.time() - start_time
                if elapsed > args.timeout:
                    print(f"\nTimeout reached ({args.timeout}s). Exiting.")
                    break
                    
                completed = 0
                active_statuses = []
                
                for job_id, info in list(job_dict.items()):
                    if info["status"] in ["completed", "failed"]:
                        completed += 1
                        continue
                        
                    try:
                        status_data = client.get_job_status(job_id)
                        current_status = status_data["status"]
                        job_dict[job_id]["status"] = current_status
                        
                        if current_status == "completed":
                            result = client.get_result(job_id)
                            dom = result.get("FacialExpressions", {}).get("DominantBasicEmotion", "None") if result.get("FaceAnalyzed") else "No Face"
                            print(f"[{elapsed:.1f}s] DONE: {info['file']} -> {dom}")
                            completed += 1
                        elif current_status == "failed":
                            print(f"[{elapsed:.1f}s] FAILED: {info['file']} -> {status_data.get('error')}")
                            completed += 1
                        else:
                            active_statuses.append(current_status)
                    except Exception as e:
                        if "still processing" not in str(e) and "pending" not in str(e):
                            pass
                
                # Print a progress update if not done
                if completed < len(job_dict):
                    pending_ct = active_statuses.count("pending")
                    processing_ct = active_statuses.count("processing")
                    print(f"   ... Progress: {completed}/{len(job_dict)} done (Pending: {pending_ct}, Processing: {processing_ct})")
                    time.sleep(args.timeout if POLL_INTERVAL < 1 else POLL_INTERVAL)

            print("\n" + "="*60)
            print("ALL BATCH JOBS FINISHED")
            print("="*60)
            
        else:
            print(f"Error: {input_path} is neither a file nor a directory.")
            return 1
            
        return 0
    
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

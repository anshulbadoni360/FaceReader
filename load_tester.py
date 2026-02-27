import requests
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def submit_image(url: str, image_path: str) -> bool:
    """Submit a single image. Returns True if successful."""
    try:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
            
        with open(image_path, "rb") as f:
            response = requests.post(
                url,
                files={"file": (Path(image_path).name, f, mime_type)}
            )
        return response.status_code == 200
    except Exception as e:
        return False

def test_batch(url: str, images: list, count: int) -> tuple[int, int, float]:
    """Test a batch of size 'count' concurrently. Returns (successes, failures, time_taken)"""
    import random
    start_time = time.time()
    successes = 0
    failures = 0
    
    # Randomly select 'count' images from the list (with replacement if we need more images than we have)
    batch_images = random.choices(images, k=count)
    
    with ThreadPoolExecutor(max_workers=count) as executor:
        futures = [executor.submit(submit_image, url, img_path) for img_path in batch_images]
        for future in as_completed(futures):
            if future.result():
                successes += 1
            else:
                failures += 1
                
    time_taken = time.time() - start_time
    return successes, failures, time_taken

def main():
    parser = argparse.ArgumentParser(description="Find the breaking point of the FaceReader API")
    parser.add_argument("path", help="Path to a test image file OR a folder containing images")
    parser.add_argument("--url", default="https://mind.monetanalytics.com/facereader/analyze", help="API URL (e.g. http://localhost:9090/submit or /analyze)")
    args = parser.parse_args()
    
    input_path = Path(args.path)
    if not input_path.exists():
        print(f"Path not found: {args.path}")
        return

    images = []
    if input_path.is_file():
        images.append(str(input_path))
    elif input_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [str(f) for f in input_path.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
        if not images:
            print(f"No valid images found in folder: {input_path}")
            return
    
    print("==================================================")
    print("🚀 FaceReader API Breaking Point Test")
    print("==================================================")
    print(f"Target URL: {args.url}")
    print(f"Loaded {len(images)} images to test with.")
    print("We will hit the endpoint concurrently (at the exact same time).")
    print("We will double the number of requests each round.")
    
    batch_size = 10
    round_num = 1
    
    while True:
        print(f"\n--- Round {round_num}: Sending {batch_size} concurrent requests ---")
        successes, failures, time_taken = test_batch(args.url, images, batch_size)
        
        print(f"Result: {successes} Succeeded | {failures} Failed | Time: {time_taken:.2f} seconds")
        
        failure_rate = failures / batch_size
        if failure_rate >= 0.40:

            print("\n🚨 BREAKING POINT REACHED! 🚨")
            print(f"The server started failing at {batch_size} simultaneous connections.")
            print("This means the Uvicorn web server dropped requests before they could even enter the queue.")
            break
            
        print("Success! The server accepted all requests into the queue.")
        
        # Increase for next round
        batch_size += batch_size
        round_num += 1
        time.sleep(1) # tiny pause between rounds

if __name__ == "__main__":
    main()

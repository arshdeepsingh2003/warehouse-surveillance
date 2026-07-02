import base64
import json
import os
import sys
import uuid
from datetime import datetime, timezone
import urllib.request
import urllib.error

# Red mock image data: 1x1 red pixel base64 encoded jpeg
MOCK_B64_JPEG = (
    "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAIBAQEBAQIBAQECAgICAgQDAgICAgUEBAMEBgUGBg"
    "YFBgYGBwkIBgcJBwYGCAsICQoKCgoKBggLDAsKDAkKCgr/2wBDAQICAgICAgUDAwUKBwYHCgoKCgoKCgoKCgoKCgoKCgoKCg"
    "oKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgr/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAA"
    "ECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFh"
    "cYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6"
    "ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAA"
    "ECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNO"
    "El8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpa"
    "anqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4vooor+Uz/f"
    "w//9k="
)

def run_test():
    print("="*60)
    print("Warehouse Surveillance - Snapshot Ingestion Integration Test")
    print("="*60)

    url = "http://localhost:8000/api/v1/alerts/ingest"
    backend_root = os.path.dirname(os.path.abspath(__file__))
    uploads_dir = os.path.join(backend_root, "uploads", "alerts")
    os.makedirs(uploads_dir, exist_ok=True)

    # --- Scenario 1: Theft Alert (Should store the image) ---
    print("\n>>> Scenario 1: Theft Alert (Should store snapshot) <<<")
    theft_alert_id = str(uuid.uuid4())
    theft_payload = {
        "id":           theft_alert_id,
        "camera_id":    "cam-05",
        "zone":         "restricted_area",
        "alert_type":   "theft_attempt",
        "severity":     "high",
        "description":  "Test theft alert with red pixel snapshot",
        "person_id":    "99-P9999",
        "confidence":   0.99,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_b64": MOCK_B64_JPEG,
        "source":       "manual_test"
    }
    
    data = json.dumps(theft_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"1. Posting theft alert to: {url}")
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            response_body = response.read().decode("utf-8")
            print(f"   [SUCCESS] Received status: {status}")
            print(f"   [SUCCESS] Response: {response_body}")
    except urllib.error.URLError as e:
        print(f"   [FAIL] Could not connect to backend: {e}")
        print("   Make sure the FastAPI backend is running before executing this test.")
        sys.exit(1)

    # Verify file creation
    print("2. Checking local uploads folder for saved file...")
    files = [f for f in os.listdir(uploads_dir) if theft_alert_id in f]
    if not files:
        print(f"   [FAIL] No file found in '{uploads_dir}' matching alert ID: {theft_alert_id}")
        sys.exit(1)
        
    filename = files[0]
    filepath = os.path.join(uploads_dir, filename)
    print(f"   [SUCCESS] Found saved image file: {filename}")
    print(f"   [SUCCESS] File size: {os.path.getsize(filepath)} bytes")

    # Fetch the image via static routing
    static_url = f"http://localhost:8000/static/alerts/{filename}"
    print(f"3. Querying static URL: {static_url}")
    try:
        with urllib.request.urlopen(static_url) as response:
            status = response.getcode()
            content_type = response.headers.get("Content-Type")
            print(f"   [SUCCESS] Received status: {status}")
            print(f"   [SUCCESS] Content-Type: {content_type}")
            if "image" in content_type:
                print("   [SUCCESS] Theft snapshot successfully saved and served!")
            else:
                print("   [FAIL] Content-Type is not an image.")
                sys.exit(1)
    except urllib.error.URLError as e:
        print(f"   [FAIL] Fetching static file failed: {e}")
        sys.exit(1)

    # --- Scenario 2: Non-Theft Alert (Should NOT store the image) ---
    print("\n>>> Scenario 2: Non-Theft Alert (Should NOT store snapshot) <<<")
    nontheft_alert_id = str(uuid.uuid4())
    nontheft_payload = {
        "id":           nontheft_alert_id,
        "camera_id":    "cam-05",
        "zone":         "restricted_area",
        "alert_type":   "ppe_violation",  # Not a theft alert
        "severity":     "low",
        "description":  "Test PPE violation alert with red pixel snapshot",
        "person_id":    "99-P9999",
        "confidence":   0.95,
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_b64": MOCK_B64_JPEG,
        "source":       "manual_test"
    }

    data = json.dumps(nontheft_payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print(f"1. Posting non-theft alert to: {url}")
    try:
        with urllib.request.urlopen(req) as response:
            status = response.getcode()
            response_body = response.read().decode("utf-8")
            print(f"   [SUCCESS] Received status: {status}")
            print(f"   [SUCCESS] Response: {response_body}")
    except urllib.error.URLError as e:
        print(f"   [FAIL] Posting non-theft alert failed: {e}")
        sys.exit(1)

    # Verify NO file is created
    print("2. Verifying NO file is saved to local uploads folder...")
    files = [f for f in os.listdir(uploads_dir) if nontheft_alert_id in f]
    if files:
        print(f"   [FAIL] Found image file '{files[0]}' on disk for a non-theft alert!")
        sys.exit(1)
    else:
        print("   [SUCCESS] Confirmed: No image file saved for non-theft alert.")

    # Cleanup the theft file we created
    try:
        os.remove(filepath)
        print(f"\n4. Cleaned up test image: {filename}")
    except Exception as e:
        print(f"   [WARNING] Could not clean up test image: {e}")

    print("\nAll integration tests passed successfully!")

if __name__ == "__main__":
    run_test()

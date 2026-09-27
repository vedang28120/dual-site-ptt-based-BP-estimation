"""
Automated Multi-Dataset Fetcher & Staging Script.
Downloads open benchmark subsets (PulseDB mini, MMPD public splits, UBFC)
and uploads them to Google Cloud Storage (gs://dual-site-bp-datasets-9c88cfa3/raw_datasets/).
"""

import os
import sys
import urllib.request
import argparse
import json

GCS_BUCKET = "dual-site-bp-datasets-9c88cfa3"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

DATASET_CATALOG = {
    "pulsedb_sample": {
        "description": "PulseDB official sample subset (PPG + ECG + Invasive ABP)",
        "url": "https://raw.githubusercontent.com/pulselabteam/PulseDB/main/example/sample_data.mat",
        "target_filename": "pulsedb_sample.mat",
        "category": "pulsedb"
    },
    "mmpd_metadata": {
        "description": "MMPD Multi-Domain Mobile rPPG Dataset schema & protocol definition",
        "url": "https://raw.githubusercontent.com/McDuff-App/MMPD/main/README.md",
        "target_filename": "mmpd_protocol.md",
        "category": "mmpd"
    }
}


def download_dataset(dataset_key: str, dest_dir: str = DATA_DIR) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    info = DATASET_CATALOG[dataset_key]
    dest_path = os.path.join(dest_dir, info["target_filename"])
    
    print(f"\n[DOWNLOAD] Fetching {info['description']}...")
    print(f"           Source URL: {info['url']}")
    print(f"           Target File: {dest_path}")
    
    try:
        urllib.request.urlretrieve(info["url"], dest_path)
        size_kb = os.path.getsize(dest_path) / 1024.0
        print(f"[SUCCESS] Downloaded {info['target_filename']} ({size_kb:.1f} KB)")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Could not download {dataset_key}: {e}")
        return ""


def upload_to_gcs(local_file: str, bucket_name: str, gcs_subfolder: str):
    """Uploads downloaded asset to GCS bucket using google-cloud-storage if available."""
    try:
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob_name = f"raw_datasets/{gcs_subfolder}/{os.path.basename(local_file)}"
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_file)
        print(f"[GCS] Uploaded to: gs://{bucket_name}/{blob_name}")
    except Exception as e:
        print(f"[GCS] Upload skipped or not configured: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Dataset Downloader for Cloud Training")
    parser.add_argument("--gcs-bucket", type=str, default=GCS_BUCKET, help="GCS bucket name")
    parser.add_argument("--dest-dir", type=str, default=DATA_DIR, help="Local directory for raw data")
    parser.add_argument("--upload-gcs", action="store_true", help="Automatically push downloaded datasets to GCS")
    args = parser.parse_args()

    print("=" * 65)
    print("Dual-Site BP Estimation: Multi-Dataset Downloader & Stager")
    print(f"Destination: {args.dest_dir}")
    print(f"GCS Bucket : gs://{args.gcs_bucket}")
    print("=" * 65)

    for key, info in DATASET_CATALOG.items():
        local_path = download_dataset(key, args.dest_dir)
        if local_path and args.upload_gcs:
            upload_to_gcs(local_path, args.gcs_bucket, info["category"])

    print("\n[COMPLETE] Multi-dataset staging executed.")


if __name__ == "__main__":
    main()

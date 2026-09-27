"""
Google Cloud Storage and Model Training Setup Script for Dual-Site BP Estimation.

Automates:
1. Pre-flight authentication verification (google-cloud-auth-verification)
2. GCS Bucket creation with AI/ML security defaults (google-cloud-storage-bucket-architect)
   - Standard storage class, us-central1 (optimal GPU quota for free trial)
   - Uniform Bucket-Level Access (UBLA) enforced
   - Public Access Prevention enforced
   - Incomplete multipart upload abort lifecycle
3. Directory layout establishment (raw_datasets, checkpoints, exports_tflite)
4. Compute Engine VM / Vertex AI command generator
"""

import sys
import os
from typing import Optional

PROJECT_ID = "project-9c88cfa3-a55e-4c90-b6e"
PROJECT_NUMBER = "861860015978"
REGION = "us-central1"
BUCKET_NAME = "dual-site-bp-datasets-9c88cfa3"


def verify_authentication() -> bool:
    """
    Step 0: Pre-flight auth verification conforming to google-cloud-auth-verification.
    """
    print("=" * 65)
    print("1. Pre-Flight Authentication Verification")
    print("=" * 65)
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError

        credentials, project = google.auth.default()
        print(f"[SUCCESS] Google Cloud credentials detected.")
        print(f"          Active Project : {project or PROJECT_ID}")
        print(f"          Service Account/User: {getattr(credentials, 'service_account_email', 'User ADC')}")
        return True
    except ImportError:
        print("[WARNING] 'google-auth' or 'google-cloud-storage' not installed in local environment.")
        print("          Install via: pip install google-cloud-storage google-auth")
        return False
    except Exception as e:
        print(f"[AUTH ERROR] Could not determine Google Cloud credentials: {e}")
        print("\nCorrective Action (google-cloud-auth-verification):")
        print("Run the following commands in your terminal or Cloud Shell:")
        print("  1. gcloud auth login")
        print("  2. gcloud auth application-default login")
        return False


def setup_storage_bucket(project_id: str = PROJECT_ID, bucket_name: str = BUCKET_NAME, location: str = REGION):
    """
    Provisions a secure GCS bucket configured for AI/ML datasets and model checkpoints.
    """
    from google.cloud import storage

    client = storage.Client(project=project_id)
    print(f"\n2. Storage Bucket Architecture ({bucket_name})")
    print("-" * 65)

    bucket = client.bucket(bucket_name)

    if bucket.exists():
        print(f"[INFO] Bucket 'gs://{bucket_name}' already exists.")
    else:
        print(f"[INFO] Creating bucket 'gs://{bucket_name}' in '{location}'...")
        bucket.storage_class = "STANDARD"

        # Security Architecture (Uniform Bucket-Level Access & Public Access Prevention)
        bucket.iam_configuration.uniform_bucket_level_access_enabled = True
        bucket.iam_configuration.public_access_prevention = "enforced"

        # Lifecycle Rule: Abort incomplete multipart uploads after 7 days
        bucket.lifecycle_rules = [
            {
                "action": {"type": "AbortIncompleteMultipartUpload"},
                "condition": {"age": 7}
            }
        ]

        new_bucket = client.create_bucket(bucket, location=location)
        print(f"[SUCCESS] Bucket 'gs://{new_bucket.name}' created in {location}.")

    # Establish folder layout
    folders = [
        "raw_datasets/pulsedb/",
        "raw_datasets/vitaldb/",
        "processed_arrays/",
        "checkpoints/",
        "exports_tflite/"
    ]
    for folder in folders:
        blob = bucket.blob(f"{folder}.keep")
        if not blob.exists():
            blob.upload_from_string(b"", content_type="application/octet-stream")
            print(f"       + Established folder: gs://{bucket_name}/{folder}")

    print(f"\n[COMPLETE] Cloud Storage Bucket is ready for training datasets and weights!")
    print(f"Console Link: https://console.cloud.google.com/storage/browser/{bucket_name}?project={project_id}")


def print_compute_instructions():
    """
    Displays the exact commands to spin up the GPU training instance.
    """
    print("\n" + "=" * 65)
    print("3. Cloud Model Training Deployment Guide")
    print("=" * 65)
    print("To train using your free trial credits (₹28,664 available):")
    print("\nOption 1: Launch Compute Engine GPU VM (1x NVIDIA Tesla T4)")
    print(f"""
gcloud compute instances create bp-training-vm \\
  --project={PROJECT_ID} \\
  --zone={REGION}-a \\
  --machine-type=n1-standard-4 \\
  --accelerator=type=nvidia-tesla-t4,count=1 \\
  --maintenance-policy=TERMINATE \\
  --image-family=common-cu121-debian-11-py310 \\
  --image-project=deeplearning-platform-release \\
  --boot-disk-size=100GB \\
  --boot-disk-type=pd-ssd \\
  --metadata="install-nvidia-driver=True" \\
  --scopes=https://www.googleapis.com/auth/cloud-platform
""")
    print("Option 2: Run directly in Google Cloud Shell (CPU Prototype & Pre-processing)")
    print("Simply upload or clone your repository into Cloud Shell and run:")
    print("  bash scripts/gcp_pipeline_setup.sh")


if __name__ == "__main__":
    authed = verify_authentication()
    if authed:
        try:
            setup_storage_bucket()
        except Exception as e:
            print(f"[ERROR] Failed to configure bucket: {e}")
            print("Note: Ensure your account has the 'Storage Admin' role (roles/storage.admin).")
    print_compute_instructions()

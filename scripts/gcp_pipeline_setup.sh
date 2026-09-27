#!/usr/bin/env bash
# ==============================================================================
# Dual-Site BP Estimation: Google Cloud Platform Setup & Training Orchestration
# Project ID: project-9c88cfa3-a55e-4c90-b6e
# Target Region: us-central1 (Highest GPU quota availability for free trial credits)
# ==============================================================================

set -euo pipefail

# 1. Configuration Variables
PROJECT_ID="project-9c88cfa3-a55e-4c90-b6e"
REGION="us-central1"
ZONE="us-central1-a"
BUCKET_NAME="dual-site-bp-datasets-9c88cfa3"
VM_NAME="bp-training-vm"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"

echo "=== Dual-Site BP Estimation: GCP Pipeline Initializer ==="
echo "Project ID  : ${PROJECT_ID}"
echo "Region      : ${REGION}"
echo "GCS Bucket  : gs://${BUCKET_NAME}"
echo "=========================================================="

# 2. Mandatory Pre-Flight Auth Verification (google-cloud-auth-verification)
echo "[Step 1/5] Verifying Google Cloud Authentication..."
CURRENT_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")

if [[ -z "${CURRENT_ACCOUNT}" ]]; then
  echo "[ERROR] No active credentialed Google Cloud account found!"
  echo "Please authenticate by running:"
  echo "  gcloud auth login"
  echo "  gcloud auth application-default login"
  exit 1
fi
echo "[SUCCESS] Authenticated as: ${CURRENT_ACCOUNT}"

# Set default project
gcloud config set project "${PROJECT_ID}" --quiet

# 3. Enable Required Google Cloud APIs
echo "[Step 2/5] Enabling Compute, Storage, and Vertex AI APIs..."
gcloud services enable \
  storage.googleapis.com \
  compute.googleapis.com \
  aiplatform.googleapis.com \
  iam.googleapis.com --quiet

# 4. GCS Bucket Provisioning (google-cloud-storage-bucket-architect)
echo "[Step 3/5] Provisioning Cloud Storage Bucket..."
if gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
  echo "[INFO] Bucket gs://${BUCKET_NAME} already exists."
else
  echo "[INFO] Creating secure-by-default AI/ML dataset bucket: gs://${BUCKET_NAME} in ${REGION}..."
  CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-bucket-architect)" \
  gcloud storage buckets create "gs://${BUCKET_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" \
    --default-storage-class=STANDARD \
    --uniform-bucket-level-access \
    --public-access-prevention

  # Apply incomplete multipart upload cleanup policy (7 days)
  cat << 'EOF' > /tmp/gcs_lifecycle.json
{
  "rule": [
    {
      "action": {"type": "AbortIncompleteMultipartUpload"},
      "condition": {"age": 7}
    }
  ]
}
EOF

  CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-bucket-architect)" \
  gcloud storage buckets update "gs://${BUCKET_NAME}" --lifecycle-file=/tmp/gcs_lifecycle.json
  rm -f /tmp/gcs_lifecycle.json
  echo "[SUCCESS] Cloud Storage Bucket initialized."
fi

# Create directory structure in bucket
echo "[Step 4/5] Establishing GCS dataset folder layout..."
touch /tmp/.keep
CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-basics)" \
gcloud storage cp /tmp/.keep "gs://${BUCKET_NAME}/raw_datasets/pulsedb/.keep" --quiet
CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-basics)" \
gcloud storage cp /tmp/.keep "gs://${BUCKET_NAME}/checkpoints/.keep" --quiet
CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }gcs-skills gcs-skills/1.0 (skill:google-cloud-storage-basics)" \
gcloud storage cp /tmp/.keep "gs://${BUCKET_NAME}/exports_tflite/.keep" --quiet
rm -f /tmp/.keep

# 5. Compute Engine GPU VM Provisioning Helper
echo "[Step 5/5] Checking GPU VM Deployment..."
if gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" &>/dev/null; then
  echo "[INFO] Compute VM ${VM_NAME} already exists in ${ZONE}."
else
  echo "[READY] To spin up your training GPU VM using your free trial credits, execute:"
  echo ""
  echo "  gcloud compute instances create ${VM_NAME} \\"
  echo "    --project=${PROJECT_ID} \\"
  echo "    --zone=${ZONE} \\"
  echo "    --machine-type=${MACHINE_TYPE} \\"
  echo "    --accelerator=type=${GPU_TYPE},count=1 \\"
  echo "    --maintenance-policy=TERMINATE \\"
  echo "    --image-family=common-cu121-debian-11-py310 \\"
  echo "    --image-project=deeplearning-platform-release \\"
  echo "    --boot-disk-size=100GB \\"
  echo "    --boot-disk-type=pd-ssd \\"
  echo "    --metadata=\"install-nvidia-driver=True\" \\"
  echo "    --scopes=https://www.googleapis.com/auth/cloud-platform"
  echo ""
fi

echo "=========================================================="
echo "GCP Setup & Storage Provisioning Complete!"
echo "Bucket URL: https://console.cloud.google.com/storage/browser/${BUCKET_NAME}?project=${PROJECT_ID}"
echo "=========================================================="

export CLUSTER_NAME="mlperf-v5p"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export PROJECT_ID="cloud-tpu-multipod-dev"
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1"
export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
# TODO change it to the TPU type you want to run the workload on.
# For example, v5p-128 or v5p-256.
export TPU_TYPE="v5p-256"
export NUM_SLICES=1
export GCS_BASE_PATH="gs://v5p-microbenchmarks/report_data_${TPU_TYPE}_${RUN_ID}"

# TODO change it
# This is the name of the workload that will be created in the xpk command.
export WORKLOAD_NAME="prisha-tpu-mb-${TPU_TYPE}" # Changed name slightly

# Your GitHub username and branch with the changes
export GITHUB_USER="prishajain1"
export BRANCH_NAME="report-generator-feature" # *** Use the branch you pushed to ***

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region ${REGION} --project ${PROJECT_ID}


GCS_PATH="${GCS_BASE_PATH}/${TPU_TYPE}/metrics_report.jsonl"

# Command to be executed on the TPU VM
COMMAND="git clone -b ${BRANCH_NAME} https://github.com/${GITHUB_USER}/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
pip install -r requirements.txt && \
python src/run_benchmark.py --config=configs/xlml_v5p_256_utksharma.yaml --generate_report && \
gsutil -m cp /tmp/microbenchmarks/outputs/metrics_report.jsonl ${GCS_PATH}"


xpk workload create \
  --cluster=${CLUSTER_NAME} \
  --device-type=${TPU_TYPE} \
  --command=${COMMAND} \
  --num-slices=${NUM_SLICES} \
  --docker-image=${DOCKER_IMAGE} \
  --workload=${WORKLOAD_NAME}

# xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_NAME}

export CLUSTER_NAME="mlperf-v5p"
export REGION="europe-west4"
export ZONE="europe-west4-b"
export PROJECT_ID="cloud-tpu-multipod-dev"
# TODO change it to the TPU type you want to run the workload on.
# For example, v5p-128 or v5p-256.
export TPU_TYPE="v5p-256"
export NUM_SLICES=1

# TODO change it
# This is the name of the workload that will be created in the xpk command.
export WORKLOAD_NAME="prisha-tpu-mb-${TPU_TYPE}-report" # Changed name slightly

# Your GitHub username and branch with the changes
export GITHUB_USER="prishajain1"
export BRANCH_NAME="report-generator-feature" # *** Use the branch you pushed to ***

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} \
  --region ${REGION} --project ${PROJECT_ID}

RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
GCS_BASE_BUCKET="gs://v5p-microbenchmarks"
GCS_RUN_DIR="${GCS_BASE_BUCKET}/report_data_${TPU_TYPE}_${RUN_ID}"
TEMP_DIR="/tmp/microbenchmarks"
XLML_DIR="${TEMP_DIR}/xlml" # Defined for clarity

# Command to be executed on the TPU VM
COMMAND="rm -rf ${TEMP_DIR} && mkdir -p ${XLML_DIR} && \
git clone -b ${BRANCH_NAME} https://github.com/${GITHUB_USER}/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
pip install -r requirements.txt && \
python src/run_benchmark.py --config=configs/xlml_v5p_256_utksharma.yaml --generate_report && \
gsutil -m cp -r ${TEMP_DIR} ${GCS_RUN_DIR}/ && \
python src/report_generator.py --gcs_run_dir=${GCS_RUN_DIR} --tpu_type=${TPU_TYPE}"

xpk workload create \
  --cluster=${CLUSTER_NAME} \
  --device-type=${TPU_TYPE} \
  --command="${COMMAND}" \
  --num-slices=${NUM_SLICES} \
  --docker-image=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1 \
  --workload=${WORKLOAD_NAME}

echo "Monitor workload: xpk workload list --cluster=${CLUSTER_NAME}"
echo "To see logs: xpk workload tail --cluster=${CLUSTER_NAME} ${WORKLOAD_NAME}"
echo "Output will be in: ${GCS_RUN_DIR}"
# xpk workload delete --cluster=${CLUSTER_NAME} --workload=${WORKLOAD_NAME}

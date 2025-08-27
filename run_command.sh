export CLUSTER_NAME="bodaborg-v6e-256-lcscld-c"
export REGION="southamerica-west1"
export ZONE="southamerica-west1-a"
export PROJECT_ID="tpu-prod-env-one-vm"
export DOCKER_IMAGE="us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1"
export TPU_TYPE="v6e-256"

# Config file path within the git repository
export REPO_CONFIG_FILE="configs/xlml_v6e_256.yaml"

export GCS_BUCKET="v5p-microbenchmarks"
export RUN_ID=$(date +%Y-%m-%d-%H-%M-%S)
export GCS_BASE_PATH="gs://${GCS_BUCKET}"
# Base path for this specific run's reports
export GCS_REPORT_PATH="${GCS_BASE_PATH}/report_data_${RUN_ID}"
# Updated HLO dump path to be inside the run's report_data folder
export GCS_HLO_DUMP_PATH="${GCS_REPORT_PATH}/hlo_dumps"

export WORKLOAD_NAME="prishajain-mb-${TPU_TYPE}"

gcloud config set project ${PROJECT_ID}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials ${CLUSTER_NAME} --region ${REGION} --project ${PROJECT_ID}

GCS_JSONL_PATH="${GCS_REPORT_PATH}/metrics_report.jsonl"

XPK_COMMAND="set -e && \
git clone -b v6e https://github.com/prishajain1/accelerator-microbenchmarks.git && \
cd accelerator-microbenchmarks && \
pip install -r requirements.txt && \
export XLA_FLAGS='--xla_dump_to=${GCS_HLO_DUMP_PATH} --xla_dump_hlo_as_text' && \
python src/run_benchmark.py \
  --config ${REPO_CONFIG_FILE} \
  --gcs_jsonl_path='${GCS_JSONL_PATH}' \
  --tpu_type='${TPU_TYPE}'"

xpk workload create --cluster=${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} \
  --device-type=${TPU_TYPE} \
  --command="${XPK_COMMAND}" \
  --num-slices=1 \
  --docker-image=${DOCKER_IMAGE} \
  --workload=${WORKLOAD_NAME}

# xpk workload delete --cluster=${CLUSTER_NAME} --zone=${ZONE} --project=${PROJECT_ID} --workload=${WORKLOAD_NAME}

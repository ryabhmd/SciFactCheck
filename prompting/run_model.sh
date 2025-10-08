#!/usr/bin/env bash
#############################################
# Environment config
# Load env vars with `. sbin/config.sh`
#############################################

# === Project config ===

export PROJECT="factuality-eval/SciFactCheck"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export DEFAULT_PARTITION="RTX6000,V100-32GB,V100-16GB,H100,A100-40GB,A100-80GB,A100-PCI,RTXA6000-SLT"

export DEV_JOB_NAME=devjob
export DEV_PARTITION="RTX6000,V100-32GB,V100-16GB" #A100-80GB

export IMAGE=$ENROOT_IMAGE_DIR/factuality_eval_2025_10_08.sqsh

# === Model + Dataset parameters ===
MODEL_NAME=${MODEL_NAME:-"meta-llama/Meta-Llama-3.1-8B-Instruct"}
MODEL_TYPE=${MODEL_TYPE:-"hf"}
DATASET_NAME=${DATASET_NAME:-"rabuahmad/scifactcheck"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"${BASE_DIR}/prompt_template.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/results"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
LIMIT=${LIMIT:-0}  # 0 means no limit

# === API Keys ===
HF_API_KEY=${HF_API_KEY:-$HF_TOKEN}
LITELLM_API_KEY=${LITELLM_API_KEY:-$OPENAI_API_KEY}

# === Job name & log setup ===
JOB_NAME="factuality_${MODEL_TYPE}_$(basename $MODEL_NAME)"
LOG_DIR=${BASE_DIR}/logs
mkdir -p ${LOG_DIR}

# === Default SLURM container settings ===
export SLURM_MEM=${SLURM_MEM:-128G}
export SLURM_CPUS=${SLURM_CPUS:-4}
export ANY_PARTITION=${ANY_PARTITION:-A100-PCI,V100-32GB,RTX6000}

export SRUN="srun -K \
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME \
  --container-workdir=${BASE_DIR} \
  --container-image=$IMAGE \
  --ntasks=1 --nodes=1 \
  -p ${ANY_PARTITION} --gpus=1 \
  --cpus-per-gpu=${SLURM_CPUS} \
  --mem=${SLURM_MEM} \
  --export ALL"

# === Command to run ===
CMD="python run_model.py \
  --model_name \"$MODEL_NAME\" \
  --model_type \"$MODEL_TYPE\" \
  --dataset_name \"$DATASET_NAME\" \
  --prompt_template \"$PROMPT_TEMPLATE\" \
  --output_dir \"$OUTPUT_DIR\" \
  --max_new_tokens $MAX_NEW_TOKENS"

# Optional limit argument
if [[ $LIMIT -gt 0 ]]; then
  CMD=\"$CMD --limit $LIMIT\"
fi

# === Launch job ===
echo "-------------"
echo "Launching factuality evaluation job"
echo "Model: $MODEL_NAME ($MODEL_TYPE)"
echo "Dataset: $DATASET_NAME"
echo "Prompt: $PROMPT_TEMPLATE"
echo "Output: $OUTPUT_DIR"
echo "-------------"

${SRUN} ${CMD} \
  > ${LOG_DIR}/${JOB_NAME}.out 2> ${LOG_DIR}/${JOB_NAME}.err

echo "Job complete. Logs in ${LOG_DIR}/${JOB_NAME}.out"
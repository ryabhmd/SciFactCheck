#!/usr/bin/env bash
#############################################
# Environment config
# Load env vars with `. sbin/config.sh`
#############################################

# === Project config ===

export BASE_DIR="/netscratch/$USER/SciFactCheck/SciFactCheck/prompting"

export PROJECT="factuality-eval/SciFactCheck"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

export DEFAULT_PARTITION="V100-32GB,H100,A100-40GB,A100-80GB,A100-PCI"

export ENROOT_IMAGE_DIR=/netscratch/abu/images
export IMAGE=$ENROOT_IMAGE_DIR/scifactcheck_prompting_litellm.sqsh
# Weights & Biases
export WANDB_PROJECT=$PROJECT

# === Model + Dataset parameters ===
MODEL_NAME=${MODEL_NAME:-"gemini/gemini-2.5-pro"}
MODEL_TYPE=${MODEL_TYPE:-litellm}
DATASET_NAME=${DATASET_NAME:-"rabuahmad/scifactcheck"}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-"${BASE_DIR}/prompt_template.txt"}
OUTPUT_DIR=${OUTPUT_DIR:-"${BASE_DIR}/results"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
LIMIT=${LIMIT:-0}  # 0 means no limit

# === API Keys ===
HF_API_KEY=${HF_API_KEY:-$HF_TOKEN}
LITELLM_API_KEY=${LITELLM_API_KEY:-$OPENAI_API_KEY}

# === Job name & log setup ===
export JOB_NAME="factuality_${MODEL_TYPE}_$(basename $MODEL_NAME)"
LOG_DIR=${BASE_DIR}/logs
mkdir -p ${LOG_DIR}

# === Default SLURM container settings ===
export SLURM_MEM=${SLURM_MEM:-128G}
export SLURM_CPUS=${SLURM_CPUS:-4}
export ANY_PARTITION=${ANY_PARTITION:-RTXA6000,V100-32GB,H100,A100-40GB,A100-80GB,A100-PCI,RTXA6000-SLT}

export SRUN="srun -K \
  --job-name=${JOB_NAME}\
  --container-mounts=/netscratch:/netscratch,$HOME:$HOME \
  --container-workdir=${BASE_DIR} \
  --container-image=$IMAGE \
  --ntasks=1 --nodes=1 \
  -p ${ANY_PARTITION} --gpus=0 \
  --mem=${SLURM_MEM} \
  --export ALL"

# === Command to run ===
CMD=(python run_model.py
	--model_name "$MODEL_NAME"
	--model_type "$MODEL_TYPE"
	--dataset_name "$DATASET_NAME"
	--prompt_template "$PROMPT_TEMPLATE"
	--output_dir "$OUTPUT_DIR"
		  )

# Optional limit argument
if [[ $LIMIT -gt 0 ]]; then
  CMD=\"$CMD --limit $LIMIT\"
fi

# === Set up logging ===
OUT_FILE=${LOG_DIR}/${JOB_NAME}.out
ERR_FILE=${LOG_DIR}/${JOB_NAME}.err

# === Launch job ===
echo "-------------"
echo "Launching factuality evaluation job"
echo "Model: $MODEL_NAME ($MODEL_TYPE)"
echo "Dataset: $DATASET_NAME"
echo "Prompt: $PROMPT_TEMPLATE"
echo "Output: $OUTPUT_DIR"
echo "-------------"
echo "Start time: $(date)"
echo "Logs: ${OUT_FILE}"

# --- Heartbeat thread: shows time + GPU status every 2 minutes ---
(
  while true; do
     echo "[HEARTBEAT] $(date): job still running..."
     nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
     sleep 120
  done
) &
HEARTBEAT_PID=$!

# --- Run job and stream logs ---
${SRUN} "${CMD[@]}" \
	  2> >(tee "${ERR_FILE}" >&2) | tee "${OUT_FILE}"

# --- Cleanup ---
kill $HEARTBEAT_PID >/dev/null 2>&1 || true
echo "Job finished at $(date)"
echo "Logs written to:"
echo "  ${OUT_FILE}"
echo "  ${ERR_FILE}"

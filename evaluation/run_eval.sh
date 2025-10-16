#!/bin/bash

# --- Configuration defaults ---
export JOB_NAME=${JOB_NAME:-eval_llm}
export ANY_PARTITION=${ANY_PARTITION:-RTXA6000,V100-32GB,H100,A100-40GB,A100-80GB,A100-PCI,RTXA6000-SLT}
export SLURM_CPUS=${SLURM_CPUS:-4}
export SLURM_MEM=${SLURM_MEM:-32G}
export ENROOT_IMAGE_DIR=/netscratch/abu/images
export IMAGE=$ENROOT_IMAGE_DIR/scifactcheck_eval.sqsh

# -------------------- CONFIGURATION --------------------

# Paths
export BASE_DIR=/netscratch/abu/SciFactCheck/SciFactCheck
export CODE_DIR=$BASE_DIR/evaluation
export INPUT_DIR=$BASE_DIR/prompting/results
export RESULTS_DIR=$CODE_DIR/results
export LOGS_DIR=$BASE_DIR/logs/evals

# Create log directory if it doesn't exist
mkdir -p $LOGS_DIR
mkdir -p $RESULTS_DIR

# Model + evaluation setup
export MODEL_NAME="Qwen/Qwen3-4B-Instruct-2507"
export INPUT_FILE="$INPUT_DIR/meta-llama_Llama-3.1-8B-Instruct_results.jsonl"
export OUTPUT_FILE="$RESULTS_DIR/meta-llama_Llama-3.1-8B-Instruct_results_evaluated.jsonl"
export EVAL_TYPES="ovr-logprobs ovr-uncertainty attr"

# Optional: API keys (read from env vars or .env file)
export HF_TOKEN="your_huggingface_token_here"
export OPENAI_API_KEY="your_openai_key_here"

export SRUN="srun -K \
	--job-name=${JOB_NAME} \
	--container-mounts=/netscratch:/netscratch,$HOME:$HOME \
	--container-workdir=${CODE_DIR} \
	--container-image=${IMAGE} \
	--ntasks=1 --nodes=1 \
	-p ${ANY_PARTITION} --gpus=0 \
	--mem=${SLURM_MEM} \
	--export ALL"

# --- Command to run ---
CMD=(python run_eval.py
	--model_name "$MODEL_NAME"
	--file "$INPUT_FILE"
	--save "$OUTPUT_FILE"
	--eval $EVAL_TYPES)

# --- Logging setup ---
OUT_FILE=${LOG_DIR}/${JOB_NAME}_$(date +"%Y%m%d_%H%M%S").out
ERR_FILE=${LOG_DIR}/${JOB_NAME}_$(date +"%Y%m%d_%H%M%S").err

# -------------------- RUN --------------------

echo "============================================================="
echo " Launching LLM Evaluation Job"
echo "-------------------------------------------------------------"
echo "Model:        $MODEL_NAME"
echo "Input file:   $INPUT_FILE"
echo "Output file:  $OUTPUT_FILE"
echo "Evaluations:  $EVAL_TYPES"
echo "Base dir:     $BASE_DIR"
echo "Image:        $IMAGE"
echo "Partition(s): $ANY_PARTITION"
echo "Job name:     $JOB_NAME"
echo "-------------------------------------------------------------"
echo "Start time:   $(date)"
echo "Logs:         $OUT_FILE"
echo "============================================================="

# --- Heartbeat (GPU + time) every 2 minutes ---
(
  while true; do
	  echo "[HEARTBEAT] $(date): job running..."
	  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
	  sleep 120
  done
) &
HEARTBEAT_PID=$!

# --- Run the job with SRUN, stream logs live ---
${SRUN} "${CMD[@]}" 2> >(tee "${ERR_FILE}" >&2) | tee "${OUT_FILE}"

# --- Cleanup ---
kill $HEARTBEAT_PID >/dev/null 2>&1 || true
echo "Job finished at $(date)"
echo "Logs written to:"
echo "  ${OUT_FILE}"
echo "  ${ERR_FILE}"

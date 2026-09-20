# Shared configuration used by the ModelArts eight-GPU checkpoint evaluator.
RUN64="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_b224_mixedsft21906_8node_thinkact_20260918"
QWEN3_BASE="$BASE/DeepEyesV2/models/Qwen3-VL-8B-Instruct"
LABELS=(mixedsft_rl_best_step40 mixedsft_rl_final_step96 qwen3_base)
STEPS=(40 96 0)
MODELS=("$RUN64/best_huggingface"
        "$RUN64/global_step_96/actor/huggingface"
        "$QWEN3_BASE")
DATASETS=(
  "VStarBench HRBench4K HRBench8K"
  "VStarBench HRBench4K HRBench8K"
  "OCRBench ChartQA_TEST MME-RealWorld-Lite MME-RealWorld-CN"
)
export LMUData="$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_OFFLINE=1
export TOOL_CUDA_VISIBLE_DEVICES=0 MODEL_CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export GROUNDING_DINO_REPLICAS=2
export VISUAL_AGENT_MAX_TURNS=6 VISUAL_AGENT_MAX_TOKENS=6144 MAX_MODEL_LEN=65536
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt"
export VLMEVAL_API_NPROC=28 VLMEVAL_FAILED_SAMPLE_RETRIES=5
export EVAL_MODEL_SERVER_BACKEND=vllm EVAL_TIMEOUT_SECONDS=0
export CHECKPOINT_EVAL_OUTPUT_ROOT="$REPO_ROOT/outputs/vlmeval/mixedsft_rl_best40_step96"

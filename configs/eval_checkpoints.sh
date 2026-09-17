# Shared configuration used by the ModelArts unified entry.
RUN16="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_2node"
RUN64="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/zwz_deepeyesv2_3k_nocount_v1_n8_b224_8node_scratch"
LABELS=(16gpu_step120 16gpu_best80 64gpu_step50 64gpu_best40)
STEPS=(120 80 50 40)
MODELS=("$RUN16/global_step_120/actor/huggingface" "$RUN16/best_huggingface"
        "$RUN64/global_step_50/actor/huggingface" "$RUN64/best_huggingface")
export EVAL_DATASETS="VStarBench HRBench4K HRBench8K"
export TOOL_CUDA_VISIBLE_DEVICES=0 MODEL_CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export GROUNDING_DINO_REPLICAS=2
export VISUAL_AGENT_MAX_TURNS=6 VISUAL_AGENT_MAX_TOKENS=6144 MAX_MODEL_LEN=65536
export VLMEVAL_API_NPROC=28 VLMEVAL_FAILED_SAMPLE_RETRIES=5
export EVAL_MODEL_SERVER_BACKEND=vllm EVAL_TIMEOUT_SECONDS=0
export CHECKPOINT_EVAL_OUTPUT_ROOT="$REPO_ROOT/outputs/vlmeval/mixed_checkpoints"

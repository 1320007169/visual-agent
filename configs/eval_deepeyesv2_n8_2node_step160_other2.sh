# Shared configuration for the copied two-node DeepEyesV2 RL step160 model.
STEP160_COPY="$REPO_ROOT/saves/visual_agent_zwz_rl/qwen3/eval_snapshots/zwz_deepeyesv2_3k_nocount_v1_n8_mixedsft21906_2node_thinkact_step160_copy"
LABELS=(deepeyesv2_n8_2node_step160)
STEPS=(160)
MODELS=("$STEP160_COPY")
DATASETS=("VStarBench HRBench8K")
export LMUData="$BASE/DeepEyesV2/evaluation/VLMEvalKit/evaluation/VLMEvalKit/LMUData"
export HF_HOME="$BASE/hf_cache"
export HF_HUB_OFFLINE=1
export TOOL_CUDA_VISIBLE_DEVICES=0 MODEL_CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export GROUNDING_DINO_REPLICAS=2
export VISUAL_AGENT_MAX_TURNS=6 VISUAL_AGENT_MAX_TOKENS=6144 MAX_MODEL_LEN=65536
export VISUAL_AGENT_SYSTEM_PROMPT_FILE="$REPO_ROOT/prompts/visual_agent_rl_system_groundingdino.txt"
export VLMEVAL_API_NPROC=28 VLMEVAL_FAILED_SAMPLE_RETRIES=5
export EVAL_MODEL_SERVER_BACKEND=vllm EVAL_TIMEOUT_SECONDS=0
export CHECKPOINT_EVAL_OUTPUT_ROOT="$REPO_ROOT/outputs/vlmeval/deepeyesv2_n8_2node_step160_other2"

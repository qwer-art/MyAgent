#!/bin/bash
# Run pi0.5 LIBERO evaluation via HuggingFace LeRobot
# No arguments needed - just: bash script/run_pi05.sh
# 输出在 outputs/pi05_eval/ 目录下

set -euo pipefail

# ── 固定配置 ──────────────────────────────────────────────
PYTHON="/home/jerett/anaconda3/envs/lerobot_latest/bin/python"
LEROBOT_EVAL="/home/jerett/anaconda3/envs/lerobot_latest/bin/lerobot-eval"
MODEL_ID="lerobot/pi05_libero_finetuned_v044"
ENV_TASK="libero_spatial"       # 可改为: libero_object, libero_goal, libero_10, libero_90
TASK_IDS="[0]"                 # 只跑第0个task, 避免OOM; 跑全部去掉此参数
N_EPISODES=5
BATCH_SIZE=1
N_ACTION_STEPS=10              # 官方推荐: n_action_steps=10 (匹配 OpenPI)
DEVICE="cuda"
OUTPUT_DIR="./outputs/pi05_eval"

# MuJoCo 渲染 (无头环境用 egl, 有显示器可去掉)
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# 清除 SOCKS 代理 (httpx 不兼容 socksio 缺失), 保留 HTTP 代理
unset all_proxy ALL_PROXY
# HuggingFace 认证
if [ -f ~/.cache/huggingface/token ]; then
    export HF_TOKEN="$(cat ~/.cache/huggingface/token | tr -d '[:space:]')"
fi

echo "=========================================="
echo " pi0.5 LIBERO Evaluation (HuggingFace)"
echo "=========================================="
echo " Model:    ${MODEL_ID}"
echo " Task:     ${ENV_TASK} (task_id=${TASK_IDS})"
echo " Episodes: ${N_EPISODES}"
echo " Device:   ${DEVICE}"
echo " Output:   ${OUTPUT_DIR}"
echo "=========================================="

# ── 运行评测 ──────────────────────────────────────────────
"${LEROBOT_EVAL}" \
    --policy.path="${MODEL_ID}" \
    --env.type=libero \
    --env.task="${ENV_TASK}" \
    --env.task_ids="${TASK_IDS}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.batch_size="${BATCH_SIZE}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.device="${DEVICE}" \
    --policy.compile_model=false \
    --output_dir="${OUTPUT_DIR}"

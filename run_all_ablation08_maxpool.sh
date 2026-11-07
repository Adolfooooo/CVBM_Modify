#!/bin/bash
# ablation_08 CPCL 平均池化 vs 最大池化：串行驱动
# 顺序：LA → BraTS19；每个数据集内部 4 进程并发（2 进程/GPU）
trap '' HUP
cd /home/xuminghao/Projects/CVBM/CVBM-ABD || exit 1
export PATH="$HOME/anaconda3/envs/cvbm/bin:$PATH"
export PYTHONPATH="/home/xuminghao/Projects/CVBM/CVBM-ABD"
mkdir -p logs/ablation_08

ts() { date '+%F %T'; }
LOG=logs/ablation_08/driver.log
MARK=logs/ablation_08/DONE_ablation08_maxpool
rm -f "$MARK"

echo "[$(ts)] ===== DRIVER START (pid $$) =====" | tee -a "$LOG"

run_one() {
    local name="$1" script="$2" rc
    echo "[$(ts)] ===== START ${name} =====" | tee -a "$LOG"
    bash "$script" >> "$LOG" 2>&1
    rc=$?
    echo "[$(ts)] ===== END ${name} (exit=${rc}) =====" | tee -a "$LOG"
    return $rc
}

run_one LA experiments/cvbm_15_1/t2/a/a1/ablation/ablation_08/run_la_max_pooling_p4.sh
LA_RC=$?
run_one BRATS19 experiments/cvbm_15_1/t2/a/a1/ablation/ablation_08/run_brats19_max_pooling_p4.sh
BR_RC=$?

echo "[$(ts)] ===== ALL DONE (la=${LA_RC} brats19=${BR_RC}) =====" | tee -a "$LOG"
echo "finished $(ts) la=${LA_RC} brats19=${BR_RC}" > "$MARK"

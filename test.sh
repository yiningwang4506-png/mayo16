#!/bin/bash

CHECKPOINT_DIR="/root/autodl-tmp/CoreDiff-main/output/corediff_dose25_mayo2016_FCB_and_drl0.50_and_medclip2times/save_models"
MODEL_TYPE="ema_model-"
RESULTS_DIR="/root/autodl-tmp/CoreDiff-main/test_results"
PROJECT_DIR="/root/autodl-tmp/CoreDiff-main"
GPU=0

cd $PROJECT_DIR
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/all_results.txt"
echo "Checkpoint Testing Results" > $SUMMARY_FILE
echo "==========================" >> $SUMMARY_FILE

total=0
success=0

for model in $(ls -v $CHECKPOINT_DIR/${MODEL_TYPE}* 2>/dev/null); do
    total=$((total + 1))
    iter=$(basename "$model" | grep -oP '\d+')
    
    echo ""
    echo "=========================================="
    echo "[$total/60] 测试 iter $iter"
    echo "=========================================="
    
    output_dir="$RESULTS_DIR/iter_$iter"
    mkdir -p "$output_dir"
    
    # 尝试：设置 save_freq=1 让它在 resume 后立即执行一次测试
    CUDA_VISIBLE_DEVICES=$GPU python main.py \
      --model_name corediff \
      --run_name test_batch_iter_${iter} \
      --resume_iter $iter \
      --batch_size 4 \
      --max_iter $((iter + 1)) \
      --train_dataset mayo_2016 \
      --test_dataset mayo_2016 \
      --test_id 9 \
      --context \
      --only_adjust_two_step \
      --dose 25 \
      --save_freq 1 \
      --use_dfl_loss \
      --dfl_weight 0.2 2>&1 | tee "$output_dir/log.txt"
    
    if grep -q "psnr" "$output_dir/log.txt"; then
        echo "iter_$iter:" >> $SUMMARY_FILE
        grep "psnr" "$output_dir/log.txt" | tail -1 >> $SUMMARY_FILE
        echo "" >> $SUMMARY_FILE
        echo "✓ 成功"
        success=$((success + 1))
    else
        echo "✗ 失败或无指标"
    fi
    
    # 清理生成的文件避免累积
    rm -rf output/test_batch_iter_${iter} 2>/dev/null
done

echo ""
echo "=========================================="
echo "全部完成!"
echo "总计: $total 个"
echo "成功: $success 个"
echo "=========================================="
echo "查看汇总: cat $SUMMARY_FILE"

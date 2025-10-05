# analyze_data_distribution_fast.py - 只采样部分数据
import numpy as np
from glob import glob
import os.path as osp

def analyze_ct_data_fast(dataset='mayo_2016_sim', dose=5, test_id=9, sample_rate=0.1):
    """快速分析版本 - 只采样部分数据"""
    
    if dataset == 'mayo_2016_sim':
        data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
    elif dataset == 'mayo_2016':
        data_root = './data_preprocess/gen_data/mayo_2016_npy'
    
    patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
    patient_ids.pop(test_id)
    
    print(f"快速分析模式 (采样率: {sample_rate*100:.0f}%)")
    print(f"数据集: {dataset}, Dose: {dose}%")
    
    full_dose_values = []
    low_dose_values = []
    
    file_count = 0
    for id in patient_ids:
        target_files = sorted(glob(osp.join(data_root, f'L{id:03d}_target_*_img.npy')))
        
        # 只采样部分文件
        sample_indices = np.arange(1, len(target_files)-1)
        sample_indices = np.random.choice(sample_indices, 
                                         size=max(1, int(len(sample_indices)*sample_rate)), 
                                         replace=False)
        
        for idx in sample_indices:
            f = target_files[idx]
            img = np.load(f).astype(np.float32)
            img = img - 1024
            
            # 进一步采样像素（每隔N个像素采样）
            img_flat = img.flatten()[::10]  # 采样10%的像素
            full_dose_values.extend(img_flat)
            file_count += 1
        
        # Low dose 同理
        low_files = sorted(glob(osp.join(data_root, f'L{id:03d}_{dose}_*_img.npy')))
        for idx in sample_indices:
            f = low_files[idx]
            img = np.load(f).astype(np.float32)
            img = img - 1024
            img_flat = img.flatten()[::10]
            low_dose_values.extend(img_flat)
    
    print(f"已采样 {file_count} 个文件")
    
    full_dose_values = np.array(full_dose_values)
    low_dose_values = np.array(low_dose_values)
    
    # ... 后续统计代码完全一样 ...
    print("\n" + "=" * 70)
    print(f"数据集分析报告 (采样数据)")
    print("=" * 70)
    
    print("\n【Full Dose 统计】")
    print(f"  采样像素数: {len(full_dose_values):,}")
    print(f"  Min:        {full_dose_values.min():.2f} HU")
    print(f"  Max:        {full_dose_values.max():.2f} HU")
    print(f"  Mean:       {full_dose_values.mean():.2f} HU")
    print(f"  Median:     {np.median(full_dose_values):.2f} HU")
    print(f"  Std:        {full_dose_values.std():.2f} HU")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n  关键分位数:")
    for p in percentiles:
        val = np.percentile(full_dose_values, p)
        print(f"    {p:3d}%:  {val:8.2f} HU")
    
    p1, p99 = np.percentile(full_dose_values, [1, 99])
    p5, p95 = np.percentile(full_dose_values, [5, 95])
    
    print("\n" + "=" * 70)
    print("【推荐配置】")
    print("=" * 70)
    
    rec_range_width = p99 - p1
    rec_n_bins = int(rec_range_width / 17.5) + 1
    rec_actual_width = rec_range_width / (rec_n_bins - 1)
    
    print(f"\n⭐ 推荐配置:")
    print(f"  n_bins = {rec_n_bins}")
    print(f"  y_min = {p1:.0f}")
    print(f"  y_max = {p99:.0f}")
    print(f"  bin宽度: {rec_actual_width:.2f} HU")
    
    print(f"\n代码示例:")
    print(f"  DRL_Loss(n_bins={rec_n_bins}, y_min={p1:.0f}, y_max={p99:.0f})")

if __name__ == '__main__':
    analyze_ct_data_fast(
        dataset='mayo_2016',
        dose=25,
        test_id=9,
        sample_rate=0.2  # 采样20%文件
    )
import numpy as np
from glob import glob
import os.path as osp

def analyze_ct_data(dataset='mayo_2016_sim', dose=5, test_id=9):
    """分析CT数据的HU值分布（不绘图版本）"""
    
    if dataset == 'mayo_2016_sim':
        data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
    elif dataset == 'mayo_2016':
        data_root = './data_preprocess/gen_data/mayo_2016_npy'
    
    patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
    patient_ids.pop(test_id)  # 移除测试集
    
    print(f"正在加载数据...")
    print(f"数据集: {dataset}")
    print(f"Dose: {dose}%")
    print(f"训练集患者: {patient_ids}")
    print(f"测试集患者: {[67, 96, 109, 143, 192, 286, 291, 310, 333, 506][test_id]}")
    
    # 收集所有full_dose数据
    full_dose_values = []
    low_dose_values = []
    
    file_count = 0
    for id in patient_ids:
        # Full dose
        target_files = sorted(glob(osp.join(data_root, f'L{id:03d}_target_*_img.npy')))
        for f in target_files[1:-1]:  # 跳过首尾
            img = np.load(f).astype(np.float32)
            img = img - 1024  # 转换为HU值
            full_dose_values.extend(img.flatten())
            file_count += 1
        
        # Low dose
        low_files = sorted(glob(osp.join(data_root, f'L{id:03d}_{dose}_*_img.npy')))
        for f in low_files[1:-1]:
            img = np.load(f).astype(np.float32)
            img = img - 1024
            low_dose_values.extend(img.flatten())
    
    print(f"已加载 {file_count} 个full dose文件")
    
    full_dose_values = np.array(full_dose_values)
    low_dose_values = np.array(low_dose_values)
    
    print("\n" + "=" * 70)
    print(f"数据集分析报告: {dataset} (Dose: {dose}%)")
    print("=" * 70)
    
    # 基础统计
    print("\n【Full Dose 统计】")
    print(f"  总像素数:   {len(full_dose_values):,}")
    print(f"  Min:        {full_dose_values.min():.2f} HU")
    print(f"  Max:        {full_dose_values.max():.2f} HU")
    print(f"  Mean:       {full_dose_values.mean():.2f} HU")
    print(f"  Median:     {np.median(full_dose_values):.2f} HU")
    print(f"  Std:        {full_dose_values.std():.2f} HU")
    
    print(f"\n  分位数统计:")
    percentiles = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5, 99.9]
    for p in percentiles:
        val = np.percentile(full_dose_values, p)
        print(f"    {p:5.1f}%:  {val:8.2f} HU")
    
    print("\n【Low Dose 统计】")
    print(f"  总像素数:   {len(low_dose_values):,}")
    print(f"  Min:        {low_dose_values.min():.2f} HU")
    print(f"  Max:        {low_dose_values.max():.2f} HU")
    print(f"  Mean:       {low_dose_values.mean():.2f} HU")
    print(f"  Std:        {low_dose_values.std():.2f} HU")
    
    # 计算有效范围（去除极端值）
    p01, p999 = np.percentile(full_dose_values, [0.1, 99.9])
    p1, p99 = np.percentile(full_dose_values, [1, 99])
    p5, p95 = np.percentile(full_dose_values, [5, 95])
    p10, p90 = np.percentile(full_dose_values, [10, 90])
    
    print("\n" + "=" * 70)
    print("【推荐的Bins范围配置】")
    print("=" * 70)
    
    print(f"\n选项1 (极保守 - 覆盖99.8%数据):")
    print(f"  范围: [{p01:.0f}, {p999:.0f}] HU")
    print(f"  宽度: {p999 - p01:.0f} HU")
    
    print(f"\n选项2 (保守 - 覆盖98%数据) ⭐ 推荐")
    print(f"  范围: [{p1:.0f}, {p99:.0f}] HU")
    print(f"  宽度: {p99 - p1:.0f} HU")
    
    print(f"\n选项3 (平衡 - 覆盖90%数据):")
    print(f"  范围: [{p5:.0f}, {p95:.0f}] HU")
    print(f"  宽度: {p95 - p5:.0f} HU")
    
    print(f"\n选项4 (激进 - 覆盖80%数据):")
    print(f"  范围: [{p10:.0f}, {p90:.0f}] HU")
    print(f"  宽度: {p90 - p10:.0f} HU")
    
    # 显示常见CT窗口
    print("\n" + "-" * 70)
    print("【医学影像标准窗口参考】")
    print("-" * 70)
    print("  软组织窗: [-160, 240] HU (宽度: 400)")
    print("  腹部窗:   [-150, 250] HU (宽度: 400)")
    print("  肺窗:     [-1000, 400] HU (宽度: 1400)")
    print("  骨窗:     [-200, 1000] HU (宽度: 1200)")
    
    # Bins数量建议
    print("\n" + "=" * 70)
    print("【Bins数量建议】")
    print("=" * 70)
    
    target_bin_widths = [10, 15, 20, 25, 30]
    
    for option, (y_min, y_max, coverage) in enumerate([
        (p1, p99, "1%-99%"),
        (p5, p95, "5%-95%"),
        (-160, 240, "软组织窗"),
    ], 1):
        range_width = y_max - y_min
        print(f"\n配置选项 {option}: 范围 [{y_min:.0f}, {y_max:.0f}] HU ({coverage})")
        print(f"  范围宽度: {range_width:.0f} HU")
        print(f"\n  目标bin宽度 | 需要bins数 | 实际bin宽度")
        print(f"  {'-'*12}|{'-'*13}|{'-'*15}")
        
        for target_width in target_bin_widths:
            n_bins = int(range_width / target_width) + 1
            actual_width = range_width / (n_bins - 1)
            marker = " ⭐" if 15 <= target_width <= 20 else ""
            print(f"  {target_width:4d} HU      |  {n_bins:3d} bins   |  {actual_width:6.2f} HU{marker}")
    
    # 计算推荐配置
    print("\n" + "=" * 70)
    print("【最终推荐配置】")
    print("=" * 70)
    
    # 推荐1：基于1%-99%，bin宽度15-20 HU
    rec_range_width = p99 - p1
    rec_n_bins = int(rec_range_width / 17.5) + 1  # 目标17.5 HU/bin
    rec_actual_width = rec_range_width / (rec_n_bins - 1)
    
    print(f"\n⭐ 推荐配置 (覆盖98%数据，平衡精度):")
    print(f"  n_bins = {rec_n_bins}")
    print(f"  y_min = {p1:.0f}")
    print(f"  y_max = {p99:.0f}")
    print(f"  实际bin宽度: {rec_actual_width:.2f} HU")
    
    print(f"\n使用示例:")
    print(f"  class DRL_Loss(nn.Module):")
    print(f"      def __init__(self, n_bins={rec_n_bins}, y_min={p1:.0f}, y_max={p99:.0f}, lambda_df=0.02):")
    print(f"          ...")
    
    # 检查数据分布类型
    print("\n" + "=" * 70)
    print("【数据分布特征】")
    print("=" * 70)
    
    skewness = (full_dose_values.mean() - np.median(full_dose_values)) / full_dose_values.std()
    
    print(f"  偏度 (Skewness): {skewness:.3f}")
    if abs(skewness) < 0.5:
        print(f"    → 接近正态分布，bins可以均匀分布")
    elif skewness > 0.5:
        print(f"    → 右偏分布，可能需要在高值区域增加bins密度")
    else:
        print(f"    → 左偏分布，可能需要在低值区域增加bins密度")
    
    # 统计落在不同区间的像素比例
    print(f"\n  数据分布区间:")
    ranges = [
        (-np.inf, -500, "空气/肺"),
        (-500, -100, "肺组织"),
        (-100, 50, "软组织"),
        (50, 200, "血液/肌肉"),
        (200, 1000, "骨骼"),
        (1000, np.inf, "金属/异常"),
    ]
    
    for low, high, name in ranges:
        mask = (full_dose_values >= low) & (full_dose_values < high)
        percentage = mask.sum() / len(full_dose_values) * 100
        if percentage > 0.1:
            print(f"    [{low:6.0f}, {high:6.0f}) HU ({name:12s}): {percentage:5.2f}%")
    
    return {
        'full_dose_stats': {
            'min': full_dose_values.min(),
            'max': full_dose_values.max(),
            'mean': full_dose_values.mean(),
            'median': np.median(full_dose_values),
            'std': full_dose_values.std(),
            'p01': p01,
            'p1': p1,
            'p5': p5,
            'p10': p10,
            'p90': p90,
            'p95': p95,
            'p99': p99,
            'p999': p999,
        },
        'recommended_config': {
            'n_bins': rec_n_bins,
            'y_min': p1,
            'y_max': p99,
            'bin_width': rec_actual_width,
        }
    }


if __name__ == '__main__':
    # 分析你的数据集
    stats = analyze_ct_data(
        dataset='mayo_2016',  # 或 'mayo_2016'
        dose=25,  # 你的dose参数
        test_id=9
    )
    
    print("\n" + "=" * 70)
    print("分析完成！请根据上述推荐配置修改代码")
    print("=" * 70)

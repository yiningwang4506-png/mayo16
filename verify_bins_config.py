# verify_bins_config.py
import numpy as np
from glob import glob
import os.path as osp

print("验证bins配置是否合理...")

# 采样3个患者，每个10个slice
patient_ids = [67, 96, 109]
data_root = './data_preprocess/gen_data/mayo_2016_npy'

values = []
for id in patient_ids:
    files = sorted(glob(osp.join(data_root, f'L{id:03d}_target_*_img.npy')))
    for f in files[5:15]:  # 取中间10个slice
        img = np.load(f).astype(np.float32) - 1024
        values.extend(img.flatten()[::50])  # 采样2%像素

values = np.array(values)

# 统计
p1, p99 = np.percentile(values, [1, 99])
p5, p95 = np.percentile(values, [5, 95])
mean, std = values.mean(), values.std()

print(f"\n数据分布快速统计:")
print(f"  Mean: {mean:.1f} HU")
print(f"  Std:  {std:.1f} HU")
print(f"  1%-99%: [{p1:.0f}, {p99:.0f}] HU")
print(f"  5%-95%: [{p5:.0f}, {p95:.0f}] HU")

# 检查配置
y_min, y_max, n_bins = -160, 240, 23

print(f"\n配置 [{y_min}, {y_max}] 的覆盖情况:")

# 计算覆盖率
covered = ((values >= y_min) & (values <= y_max)).sum()
coverage = covered / len(values) * 100

print(f"  覆盖 {coverage:.2f}% 的像素")

if coverage >= 95:
    print(f"  ✅ 覆盖率很好，推荐使用此配置")
elif coverage >= 90:
    print(f"  ⚠️ 覆盖率尚可，建议调整为 [{p5:.0f}, {p95:.0f}]")
else:
    print(f"  ❌ 覆盖率不足，必须调整为 [{p1:.0f}, {p99:.0f}]")

# 检查bins数量
range_width = y_max - y_min
bin_width = range_width / (n_bins - 1)
data_range = p99 - p1

print(f"\nBins数量检查:")
print(f"  配置bin宽度: {bin_width:.2f} HU")
print(f"  数据实际范围: {data_range:.0f} HU")
print(f"  建议bins数量: {int(data_range / 17.5) + 1}")

if 15 <= bin_width <= 22:
    print(f"  ✅ Bin宽度合理")
else:
    print(f"  ⚠️ Bin宽度可能需要调整")

print(f"\n最终推荐:")
print(f"  n_bins = {n_bins}")
print(f"  y_min = {y_min}")
print(f"  y_max = {y_max}")
print(f"\n如果覆盖率<95%，请根据上述建议调整配置")

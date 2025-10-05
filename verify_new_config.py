# verify_new_config.py
import numpy as np
from glob import glob
import os.path as osp

print("验证新配置...")

patient_ids = [67, 96, 109]
data_root = './data_preprocess/gen_data/mayo_2016_npy'

values = []
for id in patient_ids:
    files = sorted(glob(osp.join(data_root, f'L{id:03d}_target_*_img.npy')))
    for f in files[5:15]:
        img = np.load(f).astype(np.float32) - 1024
        values.extend(img.flatten()[::50])

values = np.array(values)

# 测试新配置
configs = [
    (83, -1024, 423, "方案A-完全覆盖"),
    (69, -1013, 180, "方案B-平衡"),
    (81, -1000, 400, "方案C-肺窗"),
]

print(f"\n数据统计:")
print(f"  范围: [{values.min():.0f}, {values.max():.0f}] HU")
print(f"  1%-99%: [{np.percentile(values, 1):.0f}, {np.percentile(values, 99):.0f}] HU")

print(f"\n配置对比:")
print(f"{'方案':<20} {'覆盖率':<10} {'Bins':<8} {'Bin宽度':<10}")
print("-" * 50)

for n_bins, y_min, y_max, name in configs:
    covered = ((values >= y_min) & (values <= y_max)).sum()
    coverage = covered / len(values) * 100
    bin_width = (y_max - y_min) / (n_bins - 1)
    
    marker = "⭐" if coverage >= 95 else "⚠️" if coverage >= 85 else "❌"
    print(f"{name:<20} {coverage:>6.2f}% {marker}  {n_bins:<8} {bin_width:>6.2f} HU")

print(f"\n建议：选择覆盖率≥95%的方案")

"""
检查所有数据源的 HU 值范围
"""
import numpy as np
from glob import glob
import os.path as osp

data_root = './data'
datasets = [
    'LZU_PH_dose10',
    'LZU_PH_dose25', 
    'ZJU_GE_dose10',
    'ZJU_GE_dose25',
    'ZJU_UI_dose10',
    'ZJU_UI_dose25',
]

print("="*70)
print("🔍 检查各数据源的 HU 值范围")
print("="*70)

for ds in datasets:
    data_dir = osp.join(data_root, ds, ds)
    
    if not osp.exists(data_dir):
        print(f"\n❌ {ds}: 目录不存在")
        continue
    
    # 加载几个 target 文件
    target_files = sorted(glob(osp.join(data_dir, '*_target.npy')))[:10]
    # 加载几个 dose 文件
    dose_files = sorted(glob(osp.join(data_dir, '*_dose*.npy')))[:10]
    
    if not target_files:
        print(f"\n❌ {ds}: 没有找到文件")
        continue
    
    # 统计 target
    target_mins, target_maxs, target_means = [], [], []
    for f in target_files:
        img = np.load(f).astype(np.float32)
        target_mins.append(img.min())
        target_maxs.append(img.max())
        target_means.append(img.mean())
    
    # 统计 dose
    dose_mins, dose_maxs, dose_means = [], [], []
    for f in dose_files:
        img = np.load(f).astype(np.float32)
        dose_mins.append(img.min())
        dose_maxs.append(img.max())
        dose_means.append(img.mean())
    
    print(f"\n📊 {ds}:")
    print(f"   Target: min=[{min(target_mins):.0f}, {max(target_mins):.0f}], "
          f"max=[{min(target_maxs):.0f}, {max(target_maxs):.0f}], "
          f"mean={np.mean(target_means):.0f}")
    print(f"   Dose:   min=[{min(dose_mins):.0f}, {max(dose_mins):.0f}], "
          f"max=[{min(dose_maxs):.0f}, {max(dose_maxs):.0f}], "
          f"mean={np.mean(dose_means):.0f}")
    
    # 判断是否正常
    # Mayo16 的正常范围：原始值约 [0, 4096]，对应 HU [-1024, 3072]
    if min(target_mins) < -1500 or max(target_maxs) > 4500:
        print(f"   ⚠️  范围异常！可能需要调整归一化")
    elif max(target_maxs) < 500:
        print(f"   ⚠️  数值太小！可能已经被归一化过了")
    elif min(target_mins) > 500:
        print(f"   ⚠️  最小值太大！可能不是标准 CT 值")
    else:
        print(f"   ✅ 范围正常")

print("\n" + "="*70)
print("📋 参考：Mayo16 的正常范围")
print("="*70)
print("   原始值: [0, 4096] (存储值)")
print("   HU 值:  [-1024, 3072] (原始值 - 1024)")
print("   归一化后: [0, 1]")
print("\n   如果你的数据已经是 [0, 1] 范围，说明已经预处理过了")
print("   需要修改 Dataset 的 normalize_ 函数！")
print("="*70)
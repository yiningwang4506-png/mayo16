import os.path as osp
from glob import glob

print("=" * 70)
print("数据诊断")
print("=" * 70)

# 检查数据目录
data_roots = {
    'mayo_2016': './data_preprocess/gen_data/mayo_2016_npy',
    'mayo_2016_sim': './data_preprocess/gen_data/mayo_2016_sim_npy'
}

for name, root in data_roots.items():
    print(f"\n检查 {name}:")
    if not osp.exists(root):
        print(f"  ❌ 目录不存在: {root}")
        continue
    
    print(f"  ✅ 目录存在: {root}")
    
    # 检查不同剂量的文件
    for dose in [5, 10, 25, 50]:
        files = glob(osp.join(root, f'L067_{dose}_*.npy'))
        if files:
            print(f"  ✅ dose {dose}%: {len(files)} 个文件")
    
    # 检查target文件
    target_files = glob(osp.join(root, 'L067_target_*.npy'))
    if target_files:
        print(f"  ✅ target (full dose): {len(target_files)} 个文件")
    
    # 统计所有患者
    all_files = glob(osp.join(root, '*.npy'))
    if all_files:
        print(f"  总文件数: {len(all_files)}")
    else:
        print(f"  ❌ 没有任何.npy文件")

print("\n" + "=" * 70)
print("推荐配置:")
print("=" * 70)

# 推荐配置
mayo_2016_exists = osp.exists(data_roots['mayo_2016']) and \
                   len(glob(osp.join(data_roots['mayo_2016'], '*.npy'))) > 0
mayo_2016_sim_exists = osp.exists(data_roots['mayo_2016_sim']) and \
                       len(glob(osp.join(data_roots['mayo_2016_sim'], '*.npy'))) > 0

if mayo_2016_exists:
    # 检查有哪些剂量
    doses = []
    for d in [5, 10, 25, 50]:
        files = glob(osp.join(data_roots['mayo_2016'], f'L067_{d}_*.npy'))
        if files:
            doses.append(d)
    
    print(f"\n✅ 使用 mayo_2016")
    print(f"   可用剂量: {doses}")
    if 25 in doses:
        print(f"\n   推荐配置:")
        print(f"   --train_dataset mayo_2016")
        print(f"   --test_dataset mayo_2016")
        print(f"   --dose 25")
    else:
        print(f"\n   推荐配置:")
        print(f"   --train_dataset mayo_2016")
        print(f"   --test_dataset mayo_2016")
        print(f"   --dose {doses[0]}")

elif mayo_2016_sim_exists:
    # 检查有哪些剂量
    doses = []
    for d in [5, 10, 25, 50]:
        files = glob(osp.join(data_roots['mayo_2016_sim'], f'L067_{d}_*.npy'))
        if files:
            doses.append(d)
    
    print(f"\n✅ 使用 mayo_2016_sim")
    print(f"   可用剂量: {doses}")
    print(f"\n   推荐配置:")
    print(f"   --train_dataset mayo_2016_sim")
    print(f"   --test_dataset mayo_2016_sim")
    print(f"   --dose {doses[0]}")

else:
    print(f"\n❌ 没有找到任何可用数据！")
    print(f"\n请先准备数据:")
    print(f"   方法1: 预处理原始DICOM数据")
    print(f"   方法2: 生成模拟数据")

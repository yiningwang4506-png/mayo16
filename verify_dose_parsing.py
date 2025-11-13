# verify_dose_parsing.py
import re
from glob import glob
import os.path as osp
import numpy as np

def parse_dose(filename):
    """
    从文件名提取剂量
    
    支持格式:
        L067_001_dose25.npy -> 25
        L067_001_dose50.npy -> 50
        L067_001_target.npy -> 100
    """
    basename = osp.basename(filename)
    
    # 方案1: 匹配 dose25, dose50 等
    match = re.search(r'dose(\d+)', basename)
    if match:
        return int(match.group(1))
    
    # 方案2: 匹配 target（全剂量）
    if 'target' in basename:
        return 100
    
    # 如果都不匹配
    print(f"⚠️ Cannot parse dose from: {basename}")
    return None


def main():
    """验证剂量解析逻辑"""
    
    # 🔥 修改为你的实际路径
    data_root = './data/Mayo16_SM_dose25_and_dose50'
    
    print("="*70)
    print("🔍 验证剂量解析逻辑")
    print("="*70)
    
    # 检查目录是否存在
    if not osp.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        print("当前工作目录:", osp.abspath('.'))
        print("\n请检查路径是否正确！")
        print("提示：可能需要修改 data_root 变量")
        return
    
    print(f"✅ 数据目录: {data_root}\n")
    
    # 🔥 列出所有 .npy 文件（不限患者）
    all_files = sorted(glob(osp.join(data_root, '*.npy')))
    
    if len(all_files) == 0:
        print(f"❌ 没有找到 .npy 文件")
        print("请检查数据目录内容！")
        return
    
    print(f"找到 {len(all_files)} 个 .npy 文件\n")
    
    # 🔥 先看前 10 个文件名，了解格式
    print("📋 前 10 个文件名:")
    print("-"*70)
    for f in all_files[:10]:
        print(f"  {osp.basename(f)}")
    print()
    
    # 测试剂量解析
    print("📊 文件名 → 剂量解析结果（前 20 个）")
    print("-"*70)
    
    dose_stats = {}  # 统计每种剂量的数量
    
    for f in all_files[:20]:
        filename = osp.basename(f)
        dose = parse_dose(f)
        
        # 统计
        if dose is not None:
            dose_stats[dose] = dose_stats.get(dose, 0) + 1
        
        # 打印
        if dose is not None:
            print(f"{filename:40s} → {dose:3d}%")
        else:
            print(f"{filename:40s} → ❌ 解析失败")
    
    # 统计所有文件的剂量分布
    print("\n" + "="*70)
    print("📈 所有文件的剂量分布统计")
    print("="*70)
    
    all_dose_stats = {}
    for f in all_files:
        dose = parse_dose(f)
        if dose is not None:
            all_dose_stats[dose] = all_dose_stats.get(dose, 0) + 1
    
    for dose in sorted(all_dose_stats.keys()):
        count = all_dose_stats[dose]
        percentage = count / len(all_files) * 100
        print(f"  {dose:3d}% 剂量: {count:5d} 个文件 ({percentage:5.1f}%)")
    
    print(f"\n  总计: {len(all_files)} 个文件")
    
    # 验证图像可读性
    print("\n" + "="*70)
    print("🖼️  验证图像数据")
    print("="*70)
    
    # 尝试读取每种剂量的第一个文件
    test_samples = {}
    for dose in sorted(all_dose_stats.keys()):
        for f in all_files:
            if parse_dose(f) == dose:
                test_samples[dose] = f
                break
    
    for dose, f in sorted(test_samples.items()):
        try:
            img = np.load(f)
            print(f"✅ {dose:3d}% 剂量: {osp.basename(f):40s}")
            print(f"   Shape: {img.shape}, Dtype: {img.dtype}")
            print(f"   Range: [{img.min():.0f}, {img.max():.0f}]")
        except Exception as e:
            print(f"❌ {dose:3d}% 剂量: {osp.basename(f):40s}")
            print(f"   Error: {e}")
    
    # 检查患者数量
    print("\n" + "="*70)
    print("👥 患者统计")
    print("="*70)
    
    # 从文件名提取患者 ID（假设格式为 L067_xxx）
    patient_ids = set()
    for f in all_files:
        basename = osp.basename(f)
        match = re.search(r'(L\d+)_', basename)
        if match:
            patient_ids.add(match.group(1))
    
    if patient_ids:
        print(f"✅ 找到 {len(patient_ids)} 个患者")
        print(f"   患者 ID: {sorted(patient_ids)}")
    else:
        print("⚠️  无法从文件名提取患者 ID")
    
    # 检查是否有 context 所需的连续 slice
    print("\n" + "="*70)
    print("🔗 验证 Context 模式（需要每个患者有连续 slice）")
    print("="*70)
    
    if patient_ids:
        # 检查第一个患者的 dose25 文件
        first_patient = sorted(patient_ids)[0]
        dose25_files = sorted([f for f in all_files 
                               if first_patient in osp.basename(f) and 'dose25' in f])
        
        if len(dose25_files) >= 3:
            print(f"✅ 患者 {first_patient} 的 dose25 有 {len(dose25_files)} 个 slice")
            print(f"   示例连续3张:")
            for i in range(min(3, len(dose25_files))):
                print(f"     - {osp.basename(dose25_files[i])}")
        else:
            print(f"⚠️  患者 {first_patient} 的 dose25 只有 {len(dose25_files)} 个 slice")
            print("   Context 模式需要至少 3 张连续 slice")
    
    # 最终建议
    print("\n" + "="*70)
    print("✅ 验证完成！")
    print("="*70)
    
    if 25 in all_dose_stats and 50 in all_dose_stats:
        print("\n✅ 你的数据格式正确，包含:")
        print(f"   - 25% 低剂量: {all_dose_stats.get(25, 0)} 个文件")
        print(f"   - 50% 低剂量: {all_dose_stats.get(50, 0)} 个文件")
        if 100 in all_dose_stats:
            print(f"   - 100% 全剂量: {all_dose_stats[100]} 个文件")
        else:
            print("   ⚠️  缺少 100% 全剂量（target）数据")
            print("   建议：检查是否有 *_target.npy 文件")
        print("\n📝 下一步：修改 dataset.py 的 data_root 路径")
    else:
        print("\n⚠️  数据不完整")
        if 25 not in all_dose_stats:
            print("   ❌ 缺少 25% 剂量数据")
        if 50 not in all_dose_stats:
            print("   ❌ 缺少 50% 剂量数据")
        if 100 not in all_dose_stats:
            print("   ⚠️  缺少 100% 全剂量数据")
    
    # 生成可复制的解析函数
    print("\n" + "="*70)
    print("📋 用于 dataset.py 的解析函数")
    print("="*70)
    print("""
def _parse_dose_from_filename(self, filename):
    '''从文件名提取剂量'''
    basename = osp.basename(filename)
    
    # 匹配 dose25, dose50 等
    match = re.search(r'dose(\\d+)', basename)
    if match:
        return int(match.group(1))
    
    # 匹配 target（全剂量）
    if 'target' in basename:
        return 100
    
    # 解析失败，使用默认值python verify_dose_parsing.py
    return 100
    """)
    
    print("\n" + "="*70)
    print("⚠️  重要提醒")
    print("="*70)
    print("修改 utils/dataset.py 时，需要将 data_root 改为:")
    print(f"  data_root = '{data_root}'")
    print("或使用绝对路径:")
    print(f"  data_root = '{osp.abspath(data_root)}'")


if __name__ == '__main__':
    main()
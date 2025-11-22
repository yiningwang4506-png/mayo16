# diagnose_text_condition.py
import sys
sys.path.append('/root/autodl-tmp/CoreDiff-main')

print("="*70)
print("🔍 诊断 Text Condition 问题")
print("="*70)

# 1. 测试数据集
print("\n[1/4] 测试 TextConditionedCTDataset...")
from text_conditioned_dataset import TextConditionedCTDataset

dataset = TextConditionedCTDataset(
    dataset='mayo_2016',
    mode='test',
    test_id=9,
    dose=25,
    context=True,
    use_text=True  # 显式传入
)

sample = dataset[0]
print(f"✅ Dataset initialized")
print(f"  Sample type: {type(sample)}")
if isinstance(sample, dict):
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Has 'description': {'description' in sample}")
    if 'description' in sample:
        print(f"  Description (first 100 chars): {sample['description'][:100]}")
else:
    print(f"  ❌ Sample is not dict! It's {type(sample)}")

# 2. 测试 argparse
print("\n[2/4] 测试 argparse...")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--use_text_condition', action='store_true')
args = parser.parse_args(['--use_text_condition'])
print(f"✅ args.use_text_condition = {args.use_text_condition}")

# 3. 测试 DataLoader
print("\n[3/4] 测试 DataLoader...")
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
batch = next(iter(dataloader))
print(f"✅ DataLoader works")
print(f"  Batch type: {type(batch)}")
if isinstance(batch, dict):
    print(f"  Batch keys: {list(batch.keys())}")
    print(f"  Has 'description': {'description' in batch}")
    if 'description' in batch:
        print(f"  Description type: {type(batch['description'])}")
        print(f"  Description[0] (first 80 chars): {batch['description'][0][:80]}")

# 4. 模拟 corediff.train() 的逻辑
print("\n[4/4] 模拟 corediff.train() 逻辑...")
use_text = True
if isinstance(batch, dict):
    text_descriptions = batch.get('description', None)
    print(f"  text_descriptions is None: {text_descriptions is None}")
    
    if use_text and text_descriptions is not None:
        print(f"  ✅ Text condition should be ACTIVE")
        print(f"  ✅ Would encode: {text_descriptions[0][:80]}...")
    else:
        print(f"  ❌ Text condition would be DISABLED")
        print(f"     use_text={use_text}, text_descriptions={text_descriptions}")

print("\n" + "="*70)
print("诊断完成！")
print("="*70)
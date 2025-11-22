# test_embedding_diversity.py
import torch
import numpy as np
import sys
sys.path.append('/root/autodl-tmp/CoreDiff-main')
from text_conditioned_dataset import TextConditionedCTDataset

dataset_25 = TextConditionedCTDataset(
    dataset='mayo_2016', mode='test', test_id=9,
    dose=25, context=True, use_text=True
)

dataset_50 = TextConditionedCTDataset(
    dataset='mayo_2016', mode='test', test_id=9,
    dose=50, context=True, use_text=True
)

emb_25 = dataset_25[0]['text_embedding']
emb_50 = dataset_50[0]['text_embedding']

# ✅ 转回 Tensor
emb_25 = torch.from_numpy(emb_25)
emb_50 = torch.from_numpy(emb_50)

# 关键指标
l2_dist = torch.norm(emb_25 - emb_50).item()
cosine_sim = torch.nn.functional.cosine_similarity(
    emb_25.unsqueeze(0),
    emb_50.unsqueeze(0)
).item()

print("="*60)
print("🔍 Text Embedding 区分度分析")
print("="*60)
print(f"\n25% Embedding:")
print(f"  Shape: {emb_25.shape}")
print(f"  Norm:  {emb_25.norm().item():.4f}")
print(f"  Mean:  {emb_25.mean().item():.4f}")
print(f"  Std:   {emb_25.std().item():.4f}")

print(f"\n50% Embedding:")
print(f"  Shape: {emb_50.shape}")
print(f"  Norm:  {emb_50.norm().item():.4f}")
print(f"  Mean:  {emb_50.mean().item():.4f}")
print(f"  Std:   {emb_50.std().item():.4f}")

print(f"\n📊 区分度指标:")
print(f"  L2 Distance:  {l2_dist:.4f}")
print(f"  Cosine Sim:   {cosine_sim:.4f}")

# 判断
print("\n" + "="*60)
if l2_dist > 1.0:
    print("✅ L2距离 > 1.0 - 区分度良好")
else:
    print(f"❌ L2距离 = {l2_dist:.4f} < 1.0 - 区分度不足!")
    print("   建议:")
    print("   1. 使用更简洁的描述 (突出dose数值)")
    print("   2. 解冻BERT最后2层")
    print("   3. 增强FiLM调制强度")

if cosine_sim < 0.95:
    print("✅ 余弦相似度 < 0.95 - 方向差异OK")
else:
    print(f"⚠️  余弦相似度 = {cosine_sim:.4f} - 方向过于相似")

print("="*60)

# 额外检查: 打印文本描述
print("\n📝 文本描述对比:")
print("-"*60)
desc_25 = dataset_25[0]['description']
desc_50 = dataset_50[0]['description']

print(f"\n25% 描述 ({len(desc_25)} 字符):")
print(f"{desc_25[:200]}...")

print(f"\n50% 描述 ({len(desc_50)} 字符):")
print(f"{desc_50[:200]}...")

# 找出关键差异词
words_25 = set(desc_25.lower().split())
words_50 = set(desc_50.lower().split())
unique_25 = words_25 - words_50
unique_50 = words_50 - words_25

print(f"\n🔑 关键差异词:")
print(f"  仅在25%: {list(unique_25)[:10]}")
print(f"  仅在50%: {list(unique_50)[:10]}")
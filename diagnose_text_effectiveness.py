# diagnose_text_effectiveness.py
import torch
import sys
sys.path.append('/root/autodl-tmp/CoreDiff-main')

from text_conditioned_dataset import TextConditionedCTDataset

print("="*70)
print("🔍 诊断文本条件的有效性")
print("="*70)

# 1. 检查 embedding 区分度
dataset_25 = TextConditionedCTDataset(
    dataset='mayo_2016', mode='train', test_id=9,
    dose=25, context=True, use_text=True
)
dataset_50 = TextConditionedCTDataset(
    dataset='mayo_2016', mode='train', test_id=9,
    dose=50, context=True, use_text=True
)

sample_25 = dataset_25[0]
sample_50 = dataset_50[0]

emb_25 = torch.from_numpy(sample_25['text_embedding'])
emb_50 = torch.from_numpy(sample_50['text_embedding'])

# 计算区分度
l2_dist = torch.norm(emb_25 - emb_50).item()
cos_sim = torch.nn.functional.cosine_similarity(
    emb_25.unsqueeze(0), emb_50.unsqueeze(0)
).item()

print(f"\n📊 Text Embedding 区分度:")
print(f"  25% embedding norm: {emb_25.norm().item():.4f}")
print(f"  50% embedding norm: {emb_50.norm().item():.4f}")
print(f"  L2 distance: {l2_dist:.4f}")
print(f"  Cosine similarity: {cos_sim:.4f}")

if l2_dist < 0.5:
    print(f"\n❌ 区分度过低！L2距离 {l2_dist:.4f} < 0.5")
    print("   → 文本条件无法有效区分不同剂量")
elif cos_sim > 0.98:
    print(f"\n⚠️  方向过于相似！余弦相似度 {cos_sim:.4f} > 0.98")
    print("   → 文本条件方向性不够")
else:
    print(f"\n✅ 区分度合格")

print(f"\n📝 文本描述:")
print(f"  25%: {sample_25['description'][:100]}...")
print(f"  50%: {sample_50['description'][:100]}...")

# 2. 检查模型中文本条件的实际影响
print(f"\n" + "="*70)
print("🔍 检查模型中文本条件的影响")
print("="*70)

from models.corediff.corediff_wrapper import UNet

model = UNet(in_channels=3, text_emb_dim=256).cuda()

# 模拟输入
x = torch.randn(2, 3, 512, 512).cuda()
t = torch.tensor([5, 5]).cuda()
x_adjust = torch.randn(2, 2, 512, 512).cuda()

# 对比有无文本条件的输出
with torch.no_grad():
    out_no_text, _ = model(x, t, x_adjust, adjust=False, text_emb=None)
    out_with_text_25, _ = model(x, t, x_adjust, adjust=False, text_emb=emb_25.unsqueeze(0).cuda())
    out_with_text_50, _ = model(x, t, x_adjust, adjust=False, text_emb=emb_50.unsqueeze(0).cuda())

diff_25 = (out_with_text_25 - out_no_text).abs().mean().item()
diff_50 = (out_with_text_50 - out_no_text).abs().mean().item()
diff_25_50 = (out_with_text_25 - out_with_text_50).abs().mean().item()

print(f"\n输出差异:")
print(f"  无文本 vs 25%文本: {diff_25:.6f}")
print(f"  无文本 vs 50%文本: {diff_50:.6f}")
print(f"  25%文本 vs 50%文本: {diff_25_50:.6f}")

if diff_25 < 0.001 and diff_50 < 0.001:
    print(f"\n❌ 文本条件几乎不影响输出！")
    print(f"   → 需要增强文本注入机制")
elif diff_25_50 < 0.0001:
    print(f"\n❌ 不同剂量的文本条件产生相同输出！")
    print(f"   → 文本embedding区分度不够 或 注入方式有问题")
else:
    print(f"\n✅ 文本条件正常工作")

print(f"\n" + "="*70)
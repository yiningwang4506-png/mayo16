# diagnose_dose_embedding.py
# 把这个文件放到你的 CoreDiff-main 目录下运行

import torch
import glob
import os

print("=" * 60)
print("🔍 Dose Embedding 诊断")
print("=" * 60)

# 自动找到最新的checkpoint
ckpt_patterns = [
    './output/*/save_models/model-*',
    './output/*/save_models/ema_model-*'
]

all_ckpts = []
for pattern in ckpt_patterns:
    all_ckpts.extend(glob.glob(pattern))

if not all_ckpts:
    print("❌ 没有找到checkpoint文件!")
    print("请确认 output 目录下有保存的模型")
    exit(1)

# 按修改时间排序，取最新的
latest = max(all_ckpts, key=os.path.getmtime)
print(f"\n📁 加载最新checkpoint: {latest}\n")

ckpt = torch.load(latest, map_location='cpu')

# 分类统计
film_params = {}
dose_embed_params = {}
other_params = {}

for key, val in ckpt.items():
    if 'film' in key.lower():
        film_params[key] = val
    elif 'dose' in key.lower():
        dose_embed_params[key] = val

# ==================== FiLM 参数分析 ====================
print("=" * 60)
print("📊 FiLM 参数分析")
print("=" * 60)

if not film_params:
    print("⚠️  没有找到 FiLM 参数!")
else:
    for key, val in film_params.items():
        print(f"\n{key}:")
        print(f"  Shape: {val.shape}")
        
        if val.numel() == 1:  # 标量 (如 residual_weight)
            print(f"  Value: {val.item():.6f}")
            if 'residual_weight' in key:
                if val.item() < 0.01:
                    print(f"  ⚠️  太小了! FiLM 几乎没作用")
                elif val.item() < 0.05:
                    print(f"  📈 在增长中，继续训练")
                else:
                    print(f"  ✅ 正常，FiLM 在发挥作用")
        else:
            print(f"  Max:  {val.abs().max().item():.6f}")
            print(f"  Mean: {val.abs().mean().item():.6f}")
            print(f"  Std:  {val.std().item():.6f}")

# ==================== Dose Embedding 参数分析 ====================
print("\n" + "=" * 60)
print("📊 Dose Embedding 参数分析")
print("=" * 60)

if not dose_embed_params:
    print("⚠️  没有找到 Dose Embedding 参数!")
else:
    for key, val in dose_embed_params.items():
        print(f"\n{key}:")
        print(f"  Shape: {val.shape}")
        print(f"  Max:  {val.abs().max().item():.6f}")
        print(f"  Mean: {val.abs().mean().item():.6f}")
        print(f"  Std:  {val.std().item():.6f}")
        
        # 如果是 embedding 层，检查 25 和 50 的区分度
        if 'dose_embed.weight' in key and val.shape[0] >= 51:
            emb_25 = val[25]
            emb_50 = val[50]
            
            # 计算区分度
            l2_dist = torch.norm(emb_25 - emb_50).item()
            cos_sim = torch.nn.functional.cosine_similarity(
                emb_25.unsqueeze(0), emb_50.unsqueeze(0)
            ).item()
            
            print(f"\n  🎯 25% vs 50% 区分度:")
            print(f"     L2距离: {l2_dist:.4f}")
            print(f"     余弦相似度: {cos_sim:.4f}")
            
            if l2_dist > 1.0:
                print(f"     ✅ 区分度良好!")
            elif l2_dist > 0.5:
                print(f"     📈 区分度还行，继续训练")
            else:
                print(f"     ⚠️  区分度较低")

# ==================== 总结 ====================
print("\n" + "=" * 60)
print("📋 诊断总结")
print("=" * 60)

# 检查关键指标
has_residual_weight = any('residual_weight' in k for k in film_params.keys())
has_dose_embed = len(dose_embed_params) > 0

if has_residual_weight:
    rw_key = [k for k in film_params.keys() if 'residual_weight' in k][0]
    rw_val = film_params[rw_key].item()
    
    if rw_val < 0.01:
        print("\n⚠️  residual_weight 太小，FiLM 还没学到东西")
        print("   建议: 继续训练，或提高 FiLM 学习率")
    elif rw_val < 0.05:
        print("\n📈 residual_weight 在增长中")
        print("   建议: 继续训练，观察是否持续增长")
    else:
        print("\n✅ residual_weight 正常，FiLM 在发挥作用")

if has_dose_embed:
    print("\n✅ Dose Embedding 参数已加载")
else:
    print("\n⚠️  没有找到 Dose Embedding，请确认用的是正确的代码")

print("\n" + "=" * 60)
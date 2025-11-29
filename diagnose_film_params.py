# diagnose_film_params.py
import torch
import glob

# 找到最新的checkpoint
ckpt_files = sorted(glob.glob('./output/*/save_models/model-*'))
if ckpt_files:
    latest = ckpt_files[-1]
    print(f"📁 加载: {latest}\n")
    ckpt = torch.load(latest, map_location='cpu')
    
    print("=" * 60)
    print("🔍 FiLM 参数分析")
    print("=" * 60)
    
    for key, val in ckpt.items():
        if 'film' in key.lower():
            print(f"\n{key}:")
            print(f"  Shape: {val.shape}")
            print(f"  Max:   {val.abs().max().item():.6f}")
            print(f"  Mean:  {val.abs().mean().item():.6f}")
            print(f"  Std:   {val.std().item():.6f}")
            
            # 关键判断
            if 'residual_weight' in key:
                print(f"  → residual_weight = {val.item():.6f}")
                if val.item() < 0.02:
                    print(f"  ⚠️  太小了！FiLM几乎没作用")
                elif val.item() > 0.1:
                    print(f"  ✅ 正常，FiLM在发挥作用")
else:
    print("❌ 没找到checkpoint")
    
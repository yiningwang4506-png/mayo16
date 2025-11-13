import sys
sys.path.insert(0, './models/corediff')
import torch
import glob

# 加载checkpoint
ckpt_files = sorted(glob.glob('./experiments/*/ckpt/*.pth'), 
                   key=lambda x: int(x.split('/')[-1].replace('.pth', '')))
if ckpt_files:
    latest = ckpt_files[-1]
    print(f"📁 加载: {latest}\n")
    ckpt = torch.load(latest, map_location='cpu')
    
    print("=" * 60)
    print("🔍 FiLM参数状态")
    print("=" * 60)
    
    film_weights = []
    for key, value in ckpt['G'].items():
        if 'dose_film' in key and 'weight' in key:
            film_weights.append(value)
            print(f"\n{key}:")
            print(f"  Max abs: {value.abs().max():.6f}")
            print(f"  Mean abs: {value.abs().mean():.6f}")
    
    max_param = max([w.abs().max() for w in film_weights])
    
    print("\n" + "=" * 60)
    print("📊 诊断结论:")
    print("=" * 60)
    
    if max_param < 0.001:
        print("❌ FiLM参数几乎为0，没有被训练")
        print("   建议: 使用方案B (FiLM lr=1e-3)")
    elif max_param < 0.01:
        print("⚠️  FiLM参数很小，训练不充分")
        print("   建议: 使用方案A (FiLM lr=4e-4)")
    else:
        print(f"✅ FiLM参数正常 (max={max_param:.4f})")
        print("   问题可能不在学习率，而在单一dose值")
else:
    print("❌ 未找到checkpoint")

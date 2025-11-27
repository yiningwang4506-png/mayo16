import torch
import glob
import numpy as np

print("="*70)
print("🔍 诊断训练崩溃问题")
print("="*70)

# 1. 检查 checkpoint
print("\n[1/5] 检查最新的 checkpoint...")
ckpt_files = sorted(glob.glob('./output/corediff_text_conditionFILM/save_models/model-*'))
if not ckpt_files:
    print("❌ 没有找到 checkpoint!")
    exit(1)

latest = ckpt_files[-1]
print(f"✅ 找到: {latest}")

ckpt = torch.load(latest, map_location='cpu')

# 2. 检查 FiLM 参数
print("\n[2/5] 检查 FiLM 参数是否被训练...")
film_params = {}
for key, val in ckpt.items():
    if 'film' in key.lower():
        film_params[key] = val
        max_val = val.abs().max().item()
        mean_val = val.abs().mean().item()
        print(f"  {key}")
        print(f"    Max: {max_val:.6f}, Mean: {mean_val:.6f}")

if not film_params:
    print("  ⚠️  没有找到 FiLM 参数（可能没被保存）")
else:
    max_film = max([v.abs().max().item() for v in film_params.values()])
    if max_film < 0.001:
        print(f"\n  ❌ FiLM 参数几乎为 0 (max={max_film:.6f})")
        print(f"     → FiLM 层没有被训练！")
    else:
        print(f"\n  ✅ FiLM 参数正常 (max={max_film:.4f})")

# 3. 检查主网络参数
print("\n[3/5] 检查主网络参数范围...")
param_stats = []
for key, val in ckpt.items():
    if 'weight' in key and 'film' not in key.lower():
        max_val = val.abs().max().item()
        param_stats.append(max_val)

if param_stats:
    print(f"  主网络参数 max: {max(param_stats):.4f}")
    print(f"  主网络参数 mean: {np.mean(param_stats):.4f}")
    
    if max(param_stats) > 100:
        print(f"\n  ❌ 参数爆炸！max={max(param_stats):.2f} > 100")
    elif max(param_stats) < 0.01:
        print(f"\n  ❌ 参数消失！max={max(param_stats):.6f} < 0.01")
    else:
        print(f"  ✅ 参数范围正常")

# 4. 检查 text_proj 层（如果存在）
print("\n[4/5] 检查 text_proj 层...")
text_proj_found = False
for key, val in ckpt.items():
    if 'text_proj' in key:
        text_proj_found = True
        print(f"  {key}: shape={val.shape}, max={val.abs().max():.6f}")

if not text_proj_found:
    print("  ⚠️  没有找到 text_proj（已被 FiLM 替代，正常）")

# 5. 检查训练日志
print("\n[5/5] 检查训练日志...")
log_files = glob.glob('./output/corediff_text_conditionFILM/logs/*.log')
if log_files:
    with open(log_files[0], 'r') as f:
        lines = f.readlines()
    
    print(f"  最后 10 行日志:")
    for line in lines[-10:]:
        print(f"    {line.strip()}")
    
    # 检查 loss 值
    losses = []
    for line in lines[-100:]:
        if 'loss' in line.lower():
            try:
                parts = line.split(',')
                for part in parts:
                    if 'loss' in part.lower():
                        val = float(part.split()[-1])
                        losses.append(val)
            except:
                pass
    
    if losses:
        print(f"\n  最近 loss 统计:")
        print(f"    最小: {min(losses):.6f}")
        print(f"    最大: {max(losses):.6f}")
        print(f"    平均: {np.mean(losses):.6f}")
        
        if max(losses) > 1.0:
            print(f"\n  ❌ Loss 爆炸！max={max(losses):.4f} > 1.0")
        elif min(losses) < 1e-6:
            print(f"\n  ❌ Loss 消失！min={min(losses):.8f} < 1e-6")
        elif np.isnan(losses[-1]):
            print(f"\n  ❌ Loss 变成 NaN！")
else:
    print("  ❌ 没有找到日志文件")

# 6. 诊断结论
print("\n" + "="*70)
print("🎯 诊断结论:")
print("="*70)

# 检查是否使用了 text_proj
if 'text_proj' in str(ckpt.keys()):
    print("\n❌ 致命问题：仍在使用旧的 text_proj!")
    print("   → 应该使用 FiLM 层")
    print("   → 重新运行 fix_all_issues.py")

# 检查 DFL weight
print("\n⚠️  DFL weight = 0.2 可能太小")
print("   建议：改为 0.5 或 1.0")

print("\n" + "="*70)
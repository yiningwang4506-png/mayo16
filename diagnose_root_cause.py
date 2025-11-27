# diagnose_root_cause.py
import torch

print("="*70)
print("🔍 根本原因诊断")
print("="*70)

# 加载 checkpoint
ckpt = torch.load('./output/corediff_text_FILM1123/save_models/model-2500', map_location='cpu')

print("\n[1] 检查 checkpoint 键名:")
print("-"*70)

# 看前20个键
keys = sorted(ckpt.keys())
print("前 20 个键:")
for i, key in enumerate(keys[:20]):
    print(f"  {i+1}. {key}")

# 检查是否有 FiLM 相关的键
film_keys = [k for k in keys if 'film' in k.lower()]
print(f"\nFiLM 相关键数量: {len(film_keys)}")

if film_keys:
    print("FiLM 键示例:")
    for key in film_keys[:5]:
        print(f"  {key}")
else:
    print("❌ 没有 FiLM 键！")

# 检查架构
has_fc = any('fc.0' in k or 'fc.2' in k for k in keys)
has_film_gen = any('film_gen' in k for k in keys)

print(f"\n架构检查:")
print(f"  有 'fc' (新架构): {'✅' if has_fc else '❌'}")
print(f"  有 'film_gen' (旧架构): {'✅' if has_film_gen else '❌'}")

# 检查 text_proj（不应该存在）
has_text_proj = any('text_proj' in k for k in keys)
print(f"  有 'text_proj' (应该被删除): {'⚠️ 存在' if has_text_proj else '✅ 不存在'}")

print("\n" + "="*70)
print("[2] 检查实际参数值")
print("-"*70)

# 检查第一个卷积层的参数
conv_keys = [k for k in keys if 'conv' in k and 'weight' in k][:3]
for key in conv_keys:
    val = ckpt[key]
    print(f"\n{key}:")
    print(f"  Max: {val.abs().max().item():.6f}")
    print(f"  Mean: {val.abs().mean().item():.6f}")

print("\n" + "="*70)
print("[3] 诊断结论")
print("="*70)

if not film_keys:
    print("""
❌ 致命问题：checkpoint 里没有 FiLM 参数！

这说明：
  1. 训练时的代码没有 FiLM 层
  2. 或者 FiLM 层的变量名不对

修复方案：
  → 检查训练时的 corediff_wrapper.py 是否有 FiLM
  → 确保 FiLM 是在 UNet.__init__ 中创建的
""")
elif not has_fc and has_film_gen:
    print("""
❌ 架构不匹配：checkpoint 用的是旧架构 (film_gen)

这说明：
  1. 训练时用的是旧代码
  2. 但测试时用的是新代码

修复方案：
  → 删除旧实验，用新代码重新训练
  → 或者恢复旧代码来测试
""")
elif has_text_proj:
    print("""
⚠️  发现 text_proj：这是被替换的旧实现

这说明：
  1. 训练时还在用 text_proj
  2. 没有用 FiLM

修复方案：
  → 确认 corediff_wrapper.py 已经移除 text_proj
  → 确认添加了 FiLM 层
""")
else:
    print("""
⚠️  架构看起来正常，但 PSNR 极低

可能原因：
  1. 学习率太小
  2. Loss 函数有问题
  3. 数据加载有问题
  4. 初始化有问题

需要检查训练日志和 loss 曲线
""")
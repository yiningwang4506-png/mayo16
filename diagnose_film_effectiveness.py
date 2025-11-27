import torch
import glob

print("="*70)
print("🔍 深度诊断 FiLM 有效性")
print("="*70)

# 加载 checkpoint
ckpt_file = sorted(glob.glob('./output/corediff_text_conditionFILM/save_models/model-*'))[-1]
print(f"加载: {ckpt_file}")

ckpt = torch.load(ckpt_file, map_location='cpu')

# 检查 FiLM 最后一层的参数（决定 gamma 和 beta）
print("\n[关键] 检查 FiLM 输出层参数:")
print("-"*70)

for key in ckpt.keys():
    if 'film' in key and 'film_gen.2' in key:  # 第2层是输出层
        val = ckpt[key]
        print(f"\n{key}:")
        print(f"  Shape: {val.shape}")
        print(f"  Max: {val.abs().max().item():.8f}")
        print(f"  Mean: {val.abs().mean().item():.8f}")
        print(f"  Std: {val.std().item():.8f}")
        
        if val.abs().max().item() < 0.001:
            print(f"  ❌ 几乎为 0！这层没被训练")
        elif val.abs().max().item() < 0.01:
            print(f"  ⚠️  很小，训练不充分")
        else:
            print(f"  ✅ 正常")

# 模拟 FiLM 的实际影响
print("\n" + "="*70)
print("🧪 模拟 FiLM 的实际影响")
print("="*70)

from models.corediff.corediff_wrapper import Network

model = Network(in_channels=3, context=True, text_emb_dim=256).cuda()
model.load_state_dict(ckpt)
model.eval()

# 模拟输入
x = torch.randn(1, 3, 256, 256).cuda()
t = torch.tensor([5]).cuda()
y = torch.randn(1, 1, 256, 256).cuda()
x_end = torch.randn(1, 1, 256, 256).cuda()

# 两个不同的文本条件
text_25 = torch.randn(1, 256).cuda()
text_50 = torch.randn(1, 256).cuda()

with torch.no_grad():
    out_no_text, _ = model(x, t, y, x_end, adjust=False, text_emb=None)
    out_25, _ = model(x, t, y, x_end, adjust=False, text_emb=text_25)
    out_50, _ = model(x, t, y, x_end, adjust=False, text_emb=text_50)

diff_no_25 = (out_25 - out_no_text).abs().mean().item()
diff_no_50 = (out_50 - out_no_text).abs().mean().item()
diff_25_50 = (out_25 - out_50).abs().mean().item()

print(f"\n输出差异（绝对值）:")
print(f"  无文本 vs 25%: {diff_no_25:.8f}")
print(f"  无文本 vs 50%: {diff_no_50:.8f}")
print(f"  25% vs 50%:    {diff_25_50:.8f}")

# 计算相对差异
baseline = out_no_text.abs().mean().item()
print(f"\n相对差异（占输出的比例）:")
print(f"  无文本 vs 25%: {diff_no_25/baseline*100:.4f}%")
print(f"  无文本 vs 50%: {diff_no_50/baseline*100:.4f}%")
print(f"  25% vs 50%:    {diff_25_50/baseline*100:.4f}%")

print("\n" + "="*70)
print("🎯 诊断结论:")
print("="*70)

if diff_25_50 / baseline < 0.001:  # < 0.1%
    print("\n❌ FiLM 几乎不起作用！")
    print(f"   影响程度: {diff_25_50/baseline*100:.4f}% < 0.1%")
    print("\n可能原因:")
    print("  1. FiLM 输出层参数太小（需要检查初始化）")
    print("  2. FiLM 学习率太低")
    print("  3. Text embedding 变化太小")
    print("\n建议修复:")
    print("  → 方案A: 增大 FiLM 学习率 10 倍")
    print("  → 方案B: 移除 FiLM 零初始化")
    print("  → 方案C: 增强文本 embedding 的区分度")
    
elif diff_25_50 / baseline < 0.01:  # < 1%
    print("\n⚠️  FiLM 影响较弱")
    print(f"   影响程度: {diff_25_50/baseline*100:.4f}% < 1%")
    print("   建议：增大 FiLM 调制强度")
    
else:
    print("\n✅ FiLM 工作正常")
    print(f"   影响程度: {diff_25_50/baseline*100:.4f}%")
    print("   问题可能在其他地方（DFL loss、数据等）")

# 检查实际的 gamma 和 beta 值
print("\n" + "="*70)
print("🔬 检查实际的 gamma 和 beta 值")
print("="*70)

# Hook to capture FiLM outputs
gamma_values = []
beta_values = []

def film_hook(module, input, output):
    # 在 FiLMLayer.forward 中捕获
    pass

# 重新运行以获取中间值
with torch.no_grad():
    # 手动计算 FiLM
    for name, module in model.named_modules():
        if 'film_conv' in name and hasattr(module, 'fc'):
            params = module.fc(text_25)
            gamma, beta = torch.chunk(params, 2, dim=1)
            print(f"\n{name}:")
            print(f"  Gamma - Max: {gamma.abs().max().item():.6f}, Mean: {gamma.abs().mean().item():.6f}")
            print(f"  Beta  - Max: {beta.abs().max().item():.6f}, Mean: {beta.abs().mean().item():.6f}")
            
            if gamma.abs().max().item() < 0.01:
                print(f"  ❌ Gamma 太小！调制几乎无效")
            if beta.abs().max().item() < 0.01:
                print(f"  ❌ Beta 太小！偏置几乎无效")

print("\n" + "="*70)
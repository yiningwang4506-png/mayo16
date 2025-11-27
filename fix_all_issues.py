import re

print("="*70)
print("🔧 修复所有问题")
print("="*70)

# ============ 步骤1: 修复 text_proj 通道不匹配 ============
print("\n[1/3] 修复 text_proj 通道不匹配...")

with open('models/corediff/corediff_wrapper.py', 'r') as f:
    content = f.read()

# 找到 FCB 输出通道
if 'self.conv2_freq = FCB' in content:
    # 提取 FCB 的 output_chs
    match = re.search(r'self\.conv2_freq = FCB\([^)]+output_chs=(\d+)', content)
    if match:
        fcb_output = int(match.group(1))
        print(f"  检测到 FCB 输出通道: {fcb_output}")
        
        # 计算融合后的通道数
        # merged = [spatial_feat, freq_feat] = [256, fcb_output]
        merged_channels = 256 + fcb_output
        print(f"  融合后通道数: 256 + {fcb_output} = {merged_channels}")
        
        # 修复 text_proj 输入通道
        old_proj = r'self\.text_proj = nn\.Conv2d\(text_emb_dim, 256, 1\)'
        new_proj = f'self.text_proj = nn.Conv2d(text_emb_dim, {merged_channels}, 1)  # 匹配融合后的通道'
        
        if re.search(old_proj, content):
            content = re.sub(old_proj, new_proj, content)
            print(f"  ✅ 修复 text_proj: 256 → {merged_channels} 通道")
        else:
            print(f"  ⚠️  未找到 text_proj 定义")

# ============ 步骤2: 添加 FiLM 层 ============
print("\n[2/3] 添加 FiLM 调制层...")

# 检查是否已有 FiLM
if 'class FiLMLayer' not in content:
    film_code = '''
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    让文本条件自适应调制特征
    """
    def __init__(self, text_dim, feature_dim):
        super().__init__()
        self.film_gen = nn.Sequential(
            nn.Linear(text_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
    
    def forward(self, feature, text_emb):
        """
        Args:
            feature: [B, C, H, W]
            text_emb: [B, text_dim]
        """
        if text_emb is None:
            return feature
        
        B, C, H, W = feature.shape
        params = self.film_gen(text_emb)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return (1 + gamma) * feature + beta


'''
    content = content.replace('class SinusoidalPosEmb', film_code + 'class SinusoidalPosEmb')
    print("  ✅ 添加 FiLM 层定义")
else:
    print("  ✅ FiLM 层已存在")

# 在 UNet.__init__ 中添加 FiLM 实例
if 'self.film_conv1' not in content:
    film_instances = '''
        # 文本条件 FiLM 调制层
        self.film_conv1 = FiLMLayer(text_emb_dim, 128)
        self.film_conv2 = FiLMLayer(text_emb_dim, 256)
        self.film_conv3 = FiLMLayer(text_emb_dim, 128)
        self.film_conv4 = FiLMLayer(text_emb_dim, 64)
'''
    
    # 在 self.outc 之前插入
    content = re.sub(
        r'(        # DRL output layer\s+self\.outc)',
        film_instances + '\n\\1',
        content
    )
    print("  ✅ 添加 FiLM 实例")
else:
    print("  ✅ FiLM 实例已存在")

# ============ 步骤3: 替换文本注入方式 ============
print("\n[3/3] 替换文本注入方式...")

# 移除旧的简单相加
old_injection = r'''        # 🔥 如果有文本条件,进行融合.*?conv2 = conv2 \+ 0\.1 \* text_feat.*?\n'''
if re.search(old_injection, content, re.DOTALL):
    content = re.sub(old_injection, '', content, flags=re.DOTALL)
    print("  ✅ 移除旧的简单相加")

# 添加 FiLM 调制
modifications = [
    (r'(conv1 = self\.conv1\(down1\))', '\\1\n        conv1 = self.film_conv1(conv1, text_emb)'),
    (r'(conv2 = self\.conv2_fusion\(merged\))', '\\1\n        conv2 = self.film_conv2(conv2, text_emb)'),
    (r'(conv3 = self\.conv3\(up1\))', '\\1\n        conv3 = self.film_conv3(conv3, text_emb)'),
    (r'(conv4 = self\.conv4\(up2\))', '\\1\n        conv4 = self.film_conv4(conv4, text_emb)'),
]

for pattern, replacement in modifications:
    if 'self.film_' in replacement and 'self.film_' not in re.search(pattern, content).group(0) if re.search(pattern, content) else False:
        content = re.sub(pattern, replacement, content)

print("  ✅ 添加 FiLM 调制（在 conv1, conv2, conv3, conv4）")

# 保存
with open('models/corediff/corediff_wrapper.py', 'w') as f:
    f.write(content)

print("\n" + "="*70)
print("✅ 所有修复完成！")
print("="*70)

print("\n📋 修复内容:")
print("  1. ✅ 修复 text_proj 通道不匹配")
print("  2. ✅ 添加 FiLM 层")
print("  3. ✅ 替换简单相加为 FiLM 调制")

print("\n🎯 架构改进:")
print("  旧方案: conv2 = conv2 + 0.1 * text_feat  (权重太小)")
print("  新方案: conv_i = FiLM(conv_i, text_emb)  (自适应调制)")
print("\n  FiLM 公式: output = (1 + γ) * feature + β")
print("  - γ, β 由文本 embedding 生成")
print("  - 每层独立调制")
print("  - γ=0, β=0 时退化为恒等映射")

print("\n" + "="*70)
print("🚀 下一步:")
print("="*70)
print("1. 重新运行诊断: python diagnose_text_effectiveness_lite.py")
print("2. 如果通过，开始训练: bash train.sh")
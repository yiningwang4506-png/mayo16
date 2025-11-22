import re

print("🔧 修复 FiLM 通道不匹配问题...")

with open('models/corediff/corediff_wrapper.py', 'r') as f:
    content = f.read()

# 方案1: 调整 film_down1 的通道数
content = re.sub(
    r"self\.film_down1 = FiLMLayer\(text_dim, 128, alpha=0\.6\)",
    "self.film_down1 = FiLMLayer(text_dim, 64, alpha=0.6)  # 匹配down1的64通道",
    content
)

with open('models/corediff/corediff_wrapper.py', 'w') as f:
    f.write(content)

print("✅ 已修复 film_down1 通道数: 128 → 64")
print("现在可以继续训练了！")
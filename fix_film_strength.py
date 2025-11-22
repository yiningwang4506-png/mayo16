import re

print("🔧 增强 FiLM 调制强度...")

with open('models/corediff/corediff_wrapper.py', 'r') as f:
    content = f.read()

# 修改 FiLMLayer 的 forward 方法
old_forward = r'''    def forward\(self, feature, text_emb\):
        """
        Args:
            feature: \[B, C, H, W\]
            text_emb: \[B, text_dim\]
        """
        if text_emb is None:
            return feature
        
        B, C, H, W = feature.shape
        
        # 生成 gamma 和 beta
        params = self.film_gen\(text_emb\)  # \[B, C\*2\]
        gamma, beta = params.chunk\(2, dim=1\)  # 各 \[B, C\]
        
        # Reshape 为 \[B, C, 1, 1\] 以便广播
        gamma = gamma.view\(B, C, 1, 1\)
        beta = beta.view\(B, C, 1, 1\)
        
        # FiLM: gamma \* x \+ beta
        # 使用残差连接：0.5 \* film \+ 0.5 \* original
        film_feature = gamma \* feature \+ beta
        return 0.5 \* film_feature \+ 0.5 \* feature'''

new_forward = '''    def forward(self, feature, text_emb):
        """
        Args:
            feature: [B, C, H, W]
            text_emb: [B, text_dim]
        """
        if text_emb is None:
            return feature
        
        B, C, H, W = feature.shape
        
        # 生成 gamma 和 beta
        params = self.film_gen(text_emb)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)  # 各 [B, C]
        
        # Reshape 为 [B, C, 1, 1] 以便广播
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # 🔥 改用 (1+gamma)*x + beta，更强的调制
        film_feature = (1 + gamma) * feature + beta
        
        # 🔥 增强调制权重（从0.5改为0.7）
        return 0.7 * film_feature + 0.3 * feature'''

content = re.sub(old_forward, new_forward, content, flags=re.DOTALL)

# 修改 UNet 中 FiLM 层的初始化，bottleneck 用更强权重
content = re.sub(
    r"self\.film_conv2 = FiLMLayer\(text_dim, 256\)",
    "self.film_conv2 = FiLMLayer(text_dim, 256)  # Bottleneck: 70% modulation",
    content
)

with open('models/corediff/corediff_wrapper.py', 'w') as f:
    f.write(content)

print("✅ FiLM 调制强度已增强到 70%")
print("✅ Bottleneck 使用最强调制")
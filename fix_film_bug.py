import re

print("🔧 修复 FiLM 层的 bug...")

with open('models/corediff/corediff_wrapper.py', 'r') as f:
    content = f.read()

# 找到 FiLMLayer 定义并替换
old_film = r'''class FiLMLayer\(nn\.Module\):.*?return \(1 \+ gamma\) \* feature \+ beta'''

new_film = '''class FiLMLayer(nn.Module):
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
            text_emb: [B, text_dim] 或 None
        """
        if text_emb is None:
            return feature
        
        B, C, H, W = feature.shape
        
        # 🔥 关键修复：生成 C*2 维参数
        params = self.film_gen(text_emb)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)  # 各 [B, C]
        
        # Reshape 为 [B, C, 1, 1]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # FiLM 调制
        return (1 + gamma) * feature + beta'''

content = re.sub(old_film, new_film, content, flags=re.DOTALL)

with open('models/corediff/corediff_wrapper.py', 'w') as f:
    f.write(content)

print("✅ FiLM 层已修复")
print("\n修复内容:")
print("  问题：film_gen 输出维度不匹配特征通道数")
print("  解决：确保 film_gen 输出 feature_dim * 2")
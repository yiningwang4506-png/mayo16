# test_correct_size.py
import torch
import sys
sys.path.append('.')
from models.corediff.corediff_wrapper import Network

print('创建模型...')
model = Network(in_channels=3, text_emb_dim=256).cuda()
print('✅ 模型创建成功')

# 🔥 使用正确的尺寸：512×512
x = torch.randn(1, 3, 512, 512).cuda()
t = torch.tensor([5]).cuda()
y = torch.randn(1, 1, 512, 512).cuda()
x_end = torch.randn(1, 1, 512, 512).cuda()
text_emb = torch.randn(1, 256).cuda()

print('测试前向传播（512×512）...')
out, out_dist = model(x, t, y, x_end, adjust=True, text_emb=text_emb)

print('✅ 前向传播成功！')
print(f'   输入: {x.shape}')
print(f'   输出: {out.shape}')
print(f'   分布: {out_dist.shape}')
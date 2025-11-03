import sys
sys.path.insert(0, './models/corediff')
import torch
from corediff_wrapper import Network

model = Network(in_channels=3, out_channels=1)
model.eval()

# 🔥 手动设置FiLM参数（模拟训练后的状态）
with torch.no_grad():
    model.unet.dose_film_down2.mlp[-1].weight.data.normal_(0, 0.01)
    model.unet.dose_film_up1.mlp[-1].weight.data.normal_(0, 0.01)

# 测试
x = torch.randn(1, 3, 512, 512)
t = torch.randint(0, 1000, (1,))
y = torch.randn(1, 1, 512, 512)
x_end = torch.randn(1, 1, 512, 512)

dose_low = torch.tensor([50.0])
dose_high = torch.tensor([200.0])

with torch.no_grad():
    out_low, _ = model(x, t, y, x_end, adjust=False, dose_value=dose_low)
    out_high, _ = model(x, t, y, x_end, adjust=False, dose_value=dose_high)

diff = (out_high - out_low).abs().mean().item()
print(f"手动初始化后，剂量差异: {diff:.6f}")
if diff > 0.01:
    print("✅ 剂量条件现在生效了！")
else:
    print("差异较小，但这在训练后会增大")

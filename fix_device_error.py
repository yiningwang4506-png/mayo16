import re

print("修复设备不匹配错误...")

with open('utils/drl_loss.py', 'r') as f:
    content = f.read()

# 在 distribution_focal_loss 方法开头添加设备转换
old_pattern = r'(def distribution_focal_loss\(self, probs, target\):.*?\n.*?B, C, H, W = probs\.shape\s*\n)'
new_code = r'\1        \n        # 确保bins和target在同一设备上\n        bins = self.bins.to(target.device)\n        \n'

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

# 替换所有的 self.bins 为 bins（在 distribution_focal_loss 方法内）
# 简单替换
content = content.replace('torch.clamp(target, self.bins[0], self.bins[-1])', 
                         'torch.clamp(target, bins[0], bins[-1])')
content = content.replace('bins_tensor = self.bins.view(1, -1, 1, 1)', 
                         'bins_tensor = bins.view(1, -1, 1, 1)')

with open('utils/drl_loss.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！")
print("\n请重新运行训练: bash train.sh")

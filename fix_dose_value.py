import re

print("开始修复...")

# 修复 diffusion_modules.py
print("\n1. 修复 diffusion_modules.py")
with open('models/corediff/diffusion_modules.py', 'r') as f:
    content = f.read()

# 添加 dose_value 参数到 sample 方法
content = re.sub(
    r'def sample\(self, batch_size=4, img=None, t=None, sampling_routine=\'ddim\', n_iter=1, start_adjust_iter=1\)',
    r'def sample(self, batch_size=4, img=None, t=None, sampling_routine=\'ddim\', n_iter=1, start_adjust_iter=1, dose_value=None)',
    content
)

with open('models/corediff/diffusion_modules.py', 'w') as f:
    f.write(content)
print("   ✅ diffusion_modules.py 修复完成")

# 修复 corediff.py
print("\n2. 修复 corediff.py")
with open('models/corediff/corediff.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    
    # 在 generate_images 方法开始后添加 dose_value
    if 'def generate_images(self, n_iter):' in line:
        # 跳过接下来的几行直到找到 low_dose, full_dose
        i += 1
        while i < len(lines) and 'low_dose, full_dose = self.test_images' not in lines[i]:
            new_lines.append(lines[i])
            i += 1
        if i < len(lines):
            new_lines.append(lines[i])  # 添加 low_dose, full_dose 那行
            # 添加 dose_value 初始化
            indent = '        '
            new_lines.append(f'\n{indent}# Create dose_value tensor\n')
            new_lines.append(f'{indent}dose_value = torch.tensor([self.dose], device=low_dose.device).float()\n')
    
    # 在 test 方法的循环内添加 dose_value
    elif 'for low_dose, full_dose in tqdm.tqdm(self.test_loader' in line:
        i += 1
        new_lines.append(lines[i])  # 添加下一行（通常是 low_dose.cuda()）
        # 添加 dose_value 初始化
        indent = '        '
        new_lines.append(f'\n{indent}# Create dose_value tensor\n')
        new_lines.append(f'{indent}dose_value = torch.tensor([self.dose], device=low_dose.device).float()\n')
    
    # 在 sample 调用中添加 dose_value 参数
    elif 'start_adjust_iter=opt.start_adjust_iter,' in line and i+1 < len(lines) and ')' in lines[i+1]:
        new_lines.append('            dose_value=dose_value\n')
    
    i += 1

with open('models/corediff/corediff.py', 'w') as f:
    f.writelines(new_lines)
print("   ✅ corediff.py 修复完成")

print("\n✅ 所有修复完成！")
print("\n现在可以运行: bash train.sh")

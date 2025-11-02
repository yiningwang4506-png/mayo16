# 读取文件
with open('models/corediff/corediff.py', 'r') as f:
    lines = f.readlines()

# 修复第267行附近的缩进
fixed_lines = []
for i, line in enumerate(lines, 1):
    # 如果是 dose_value 相关的行，确保缩进正确（8个空格）
    if 'dose_value = torch.tensor' in line or 'dose_value = torch.full' in line:
        # 去掉所有前导空格，重新添加8个空格
        content = line.lstrip()
        fixed_lines.append('        ' + content)  # 8个空格
    else:
        fixed_lines.append(line)

# 写回文件
with open('models/corediff/corediff.py', 'w') as f:
    f.writelines(fixed_lines)

print("✅ 缩进修复完成")

import re

print("开始修复所有缩进问题...")

with open('models/corediff/corediff.py', 'r') as f:
    content = f.read()

# 先备份
with open('models/corediff/corediff.py.backup2', 'w') as f:
    f.write(content)
print("✅ 已备份到 corediff.py.backup2")

# 方法：逐行处理，确保在方法内的所有代码缩进一致
lines = content.split('\n')
fixed_lines = []
current_indent = 0
in_method = False

for i, line in enumerate(lines):
    stripped = line.lstrip()
    
    # 检测方法定义
    if stripped.startswith('def '):
        in_method = True
        current_indent = 4  # 方法内代码缩进4个空格
        fixed_lines.append(line)
        continue
    
    # 检测类定义
    if stripped.startswith('class '):
        in_method = False
        current_indent = 0
        fixed_lines.append(line)
        continue
    
    # 空行保持原样
    if not stripped:
        fixed_lines.append('')
        continue
    
    # 如果在方法内
    if in_method:
        # 计算应该的缩进级别
        if stripped.startswith(('return ', 'raise ', 'pass')):
            # return/raise/pass 语句
            indent_level = 2  # 8个空格
        elif stripped.startswith(('if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except', 'finally:', 'with ')):
            # 控制流语句
            indent_level = 2  # 8个空格
        elif any(stripped.startswith(x) for x in ['# ', 'dose_value', 'gen_full_dose', 'low_dose', 'full_dose', 'opt', 'self.']):
            # 普通语句
            # 检查是否在 for/if 等块内
            prev_stripped = lines[i-1].lstrip() if i > 0 else ''
            if prev_stripped.startswith(('for ', 'if ', 'elif ', 'else:', 'try:', 'except', 'with ')):
                indent_level = 3  # 12个空格（块内）
            else:
                indent_level = 2  # 8个空格
        else:
            # 其他情况，保持相对缩进
            indent_level = 2
        
        # 重建行
        fixed_lines.append('    ' * indent_level + stripped)
    else:
        # 不在方法内，保持原缩进
        fixed_lines.append(line)

# 写回文件
with open('models/corediff/corediff.py', 'w') as f:
    f.write('\n'.join(fixed_lines))

print("✅ 缩进修复完成")
print("\n请运行: python main.py ... 测试")

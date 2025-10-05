import re

print("正在应用DRL配置...")

# 1. 修改 utils/drl_loss.py
print("\n[1/3] 修改 utils/drl_loss.py")
with open('utils/drl_loss.py', 'r') as f:
    content = f.read()

# 替换初始化参数
content = re.sub(
    r'def __init__\(self, n_bins=\d+, y_min=-?\d+, y_max=\d+',
    'def __init__(self, n_bins=83, y_min=-1024, y_max=423',
    content
)

with open('utils/drl_loss.py', 'w') as f:
    f.write(content)
print("  ✅ 已更新为 n_bins=83, y_min=-1024, y_max=423")

# 2. 修改 models/corediff/corediff_wrapper.py
print("\n[2/3] 修改 models/corediff/corediff_wrapper.py")
with open('models/corediff/corediff_wrapper.py', 'r') as f:
    content = f.read()

# 替换outconv的__init__
content = re.sub(
    r'def __init__\(self, in_ch, out_ch, n_bins=\d+\)',
    'def __init__(self, in_ch, out_ch, n_bins=83)',
    content
)

# 替换y_min, y_max
content = re.sub(
    r'y_min, y_max = -?\d+, \d+',
    'y_min, y_max = -1024, 423',
    content
)

with open('models/corediff/corediff_wrapper.py', 'w') as f:
    f.write(content)
print("  ✅ 已更新outconv类配置")

# 3. 修改 models/corediff/corediff.py
print("\n[3/3] 修改 models/corediff/corediff.py")
with open('models/corediff/corediff.py', 'r') as f:
    content = f.read()

# 替换DRL_Loss初始化
content = re.sub(
    r'DRL_Loss\(n_bins=\d+, y_min=-?\d+, y_max=\d+',
    'DRL_Loss(n_bins=83, y_min=-1024, y_max=423',
    content
)

with open('models/corediff/corediff.py', 'w') as f:
    f.write(content)
print("  ✅ 已更新loss函数配置")

print("\n" + "="*60)
print("配置应用完成！")
print("="*60)
print("\n最终配置:")
print("  n_bins = 83")
print("  y_min = -1024 HU")
print("  y_max = 423 HU")
print("  覆盖率 = 99.00%")
print("  bin宽度 = 17.65 HU")
print("\n现在可以开始训练: bash train.sh")

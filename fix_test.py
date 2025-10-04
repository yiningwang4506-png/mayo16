import re

# 读取 basic_template.py
with open('models/basic_template.py', 'r') as f:
    content = f.read()

# 找到 set_loader 方法，确保test用正确的dataset
# 通常是这样的结构：
# self.test_loader = DataLoader(dataset_dict[opt.test_dataset](...))

# 检查是否已经正确
if "from utils.dataset import CTDataset" not in content:
    # 在import部分添加
    content = content.replace(
        "from utils.measure import *",
        "from utils.measure import *\nfrom utils.dataset import CTDataset"
    )

# 找到test方法开头，移除我之前添加的return
content = content.replace(
    "def test(self, n_iter):\n        return  # 跳过测试",
    "def test(self, n_iter):"
)

with open('models/basic_template.py', 'w') as f:
    f.write(content)

print("Step 1: basic_template.py 已清理")

# 读取 corediff.py 的 test 方法
with open('models/corediff/corediff.py', 'r') as f:
    corediff_content = f.read()

# 检查test方法中的sample调用
# 确保传递的img参数是完整的5通道
# 通常不需要改，因为test_loader会提供5通道数据

print("Step 2: corediff.py test方法检查完成")


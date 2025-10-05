print("修复 dataset_dict 配置...")

with open('utils/dataset.py', 'r') as f:
    content = f.read()

# 找到并替换 dataset_dict
old_dict = """dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016', mode='train', test_id=9, dose=25, context=True),
    'test': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),"""

new_dict = """dataset_dict = {
    'train': partial(CTDataset, mode='train', test_id=9, context=True),
    'test': partial(CTDataset, mode='test', test_id=9, context=True),"""

if old_dict in content:
    content = content.replace(old_dict, new_dict)
    print("  ✅ 已修复 'train' 和 'test' 条目")
else:
    print("  ⚠️ 未找到标准格式，尝试通用替换...")
    # 通用替换
    import re
    content = re.sub(
        r"'train': partial\(CTDataset, dataset='[^']+', mode='train'",
        "'train': partial(CTDataset, mode='train'",
        content
    )
    content = re.sub(
        r"'train': partial\(CTDataset, mode='train', test_id=\d+, dose=\d+",
        "'train': partial(CTDataset, mode='train', test_id=9",
        content
    )
    content = re.sub(
        r"'test': partial\(CTDataset, dataset='[^']+', mode='test'",
        "'test': partial(CTDataset, mode='test'",
        content
    )
    content = re.sub(
        r"'test': partial\(CTDataset, mode='test', test_id=\d+, dose=\d+",
        "'test': partial(CTDataset, mode='test', test_id=9",
        content
    )

with open('utils/dataset.py', 'w') as f:
    f.write(content)

print("✅ 修复完成！")
print("\n现在运行: bash train.sh")

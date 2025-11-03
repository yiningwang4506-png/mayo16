import re

with open('main.py', 'r', encoding='utf-8') as f:
    content = f.read()

print("=" * 60)
print("🔍 检查 main.py 中的剂量处理")
print("=" * 60)

# 检查1: 是否有--dose参数定义
if '--dose' in content or 'args.dose' in content:
    print("\n✅ 找到 --dose 参数定义")
    dose_matches = re.findall(r'.*args\.dose.*', content)
    for match in dose_matches[:5]:
        print(f"   {match.strip()}")
else:
    print("\n❌ 未找到 --dose 参数")

# 检查2: 是否传递给模型
if 'dose_value' in content:
    print("\n✅ 找到 dose_value 参数传递")
    dose_value_matches = re.findall(r'.*dose_value.*', content)
    for match in dose_value_matches[:5]:
        print(f"   {match.strip()}")
else:
    print("\n⚠️  未找到 dose_value 传递给模型")
    print("   需要添加 dose_value 参数！")

# 检查3: model forward调用
forward_pattern = r'model\([^)]+\)'
forward_calls = re.findall(forward_pattern, content)
if forward_calls:
    print(f"\n📝 找到 {len(forward_calls)} 处模型调用")
    for i, call in enumerate(forward_calls[:3], 1):
        print(f"\n   调用 {i}: {call[:100]}...")
        if 'dose_value' in call:
            print("      ✅ 包含 dose_value")
        else:
            print("      ⚠️  缺少 dose_value")

print("\n" + "=" * 60)

import torch

# 加载checkpoint
ckpt = torch.load('output/corediff_dose25_mayo2016_FCB_and_drl0.50_and_medclipFiLM/save_models/model-150000', map_location='cpu')
opt = torch.load('output/corediff_dose25_mayo2016_FCB_and_drl0.50_and_medclipFiLM/save_models/optimizer-150000', map_location='cpu')

# 找到FiLM参数在optimizer中的索引
film_param_groups = []
for i, param_group in enumerate(opt['param_groups']):
    # 检查是否包含FiLM参数
    if len(param_group['params']) > 0:
        # 这里需要根据实际optimizer结构调整
        pass

print("✅ 请在训练脚本中修改optimizer创建部分")
print("将FiLM参数的学习率设置为主网络的5-10倍")

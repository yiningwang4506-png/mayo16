#!/usr/bin/env python
"""
检查 PubMed-BERT 模型是否已下载且完整
"""
import os
from pathlib import Path

print("=" * 60)
print("🔍 检查 PubMed-BERT 模型")
print("=" * 60)

# 可能的缓存路径
possible_paths = [
    "./pretrained_models",
    "./pretrained_models/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    os.path.expanduser("~/.cache/huggingface/hub"),
    os.path.expanduser("~/.cache/huggingface/hub/models--microsoft--BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"),
]

found_path = None

for path in possible_paths:
    if os.path.exists(path):
        print(f"\n✅ 路径存在: {path}")
        
        # 检查是否包含模型文件
        for root, dirs, files in os.walk(path):
            for f in files:
                if f in ["pytorch_model.bin", "model.safetensors", "config.json"]:
                    full_path = os.path.join(root, f)
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    print(f"   📦 {f}: {size_mb:.1f} MB")
                    
                    if f in ["pytorch_model.bin", "model.safetensors"] and size_mb > 400:
                        found_path = path
    else:
        print(f"❌ 不存在: {path}")

print("\n" + "=" * 60)

if found_path:
    print("✅ BERT 模型已找到！")
    print(f"📁 位置: {found_path}")
    
    # 尝试加载
    print("\n🔄 测试加载模型...")
    try:
        import os
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        from transformers import AutoTokenizer, AutoModel
        
        model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        
        # 确定正确的 cache_dir
        if "pretrained_models" in found_path:
            cache_dir = "./pretrained_models"
        else:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        print(f"   使用缓存目录: {cache_dir}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("   ✅ Tokenizer 加载成功")
        
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )
        print("   ✅ Model 加载成功")
        
        # 测试编码
        import torch
        text = "This is a 25% low-dose CT scan from a GE scanner."
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]
        
        print(f"\n✅ 模型功能正常！")
        print(f"   测试文本: {text}")
        print(f"   CLS embedding shape: {cls_emb.shape}")
        print(f"   Embedding norm: {cls_emb.norm().item():.4f}")
        
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 未找到 BERT 模型！")
    print("\n请先下载模型：")
    print("python -c \"from transformers import AutoModel, AutoTokenizer; ")
    print("           AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', cache_dir='./pretrained_models');")
    print("           AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', cache_dir='./pretrained_models')\"")

print("=" * 60)
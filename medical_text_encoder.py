"""
Medical Text Encoder using PubMed-BERT
使用 PubMed-BERT 将医学描述文本编码为条件向量
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os


class MedicalTextEncoder(nn.Module):
    """
    医学文本编码器
    使用预训练的 PubMed-BERT / BioLinkBERT 将医学描述编码为条件向量
    """
    
    def __init__(self, 
                 model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 output_dim=256,
                 freeze_bert=True,
                 cache_dir='./pretrained_models'):
        """
        Args:
            model_name: Hugging Face 模型名称
                - 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' (推荐)
                - 'michiyasunaga/BioLinkBERT-base'
                - 'dmis-lab/biobert-base-cased-v1.1'
            output_dim: 输出embedding维度 (默认256,与CoreDiff兼容)
            freeze_bert: 是否冻结BERT参数 (推荐True以节省显存)
            cache_dir: 模型缓存目录
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # 加载预训练tokenizer和模型
        print(f"Loading medical text encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        self.bert = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        
        # 冻结BERT参数
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print("  ✓ BERT parameters frozen")
        
        # BERT输出维度 (通常是768)
        bert_dim = self.bert.config.hidden_size
        
        # 投影层: 768D → output_dim
        self.projection = nn.Sequential(
            nn.Linear(bert_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        print(f"  ✓ Text encoder initialized: {bert_dim}D → {output_dim}D")
    
    def forward(self, text_descriptions):
        """
        Args:
            text_descriptions: List[str] or str - 医学描述句子
            
        Returns:
            torch.Tensor: [batch_size, output_dim] - 文本embedding
        """
        # 确保输入是列表
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]
        
        # Tokenize
        inputs = self.tokenizer(
            text_descriptions,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128  # CT描述通常不超过128 tokens
        )
        
        # 移动到正确设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # BERT编码
        with torch.set_grad_enabled(not self.bert.training):
            outputs = self.bert(**inputs)
        
        # 使用 [CLS] token 的输出作为句子表示
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, 768]
        
        # 投影到目标维度
        text_embedding = self.projection(cls_embedding)  # [B, output_dim]
        
        return text_embedding
    
    def encode_batch(self, dose_list, site_list, **kwargs):
        """
        便捷方法:直接从元数据编码
        
        Args:
            dose_list: List[int] - 剂量列表
            site_list: List[str] - 站点列表
            **kwargs: 其他参数传递给 TextDescriptionGenerator
            
        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        from text_description_generator import TextDescriptionGenerator
        
        # 生成描述
        descriptions = TextDescriptionGenerator.batch_generate_descriptions(
            dose_list, site_list, **kwargs
        )
        
        # 编码
        return self.forward(descriptions)


class CachedTextEncoder(MedicalTextEncoder):
    """
    带缓存的文本编码器
    对于训练集中的固定描述,缓存embedding以加速训练
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}  # {description: embedding}
        self.use_cache = True
    
    def forward(self, text_descriptions):
        """
        带缓存的前向传播
        """
        if not self.use_cache:
            return super().forward(text_descriptions)
        
        # 确保输入是列表
        if isinstance(text_descriptions, str):
            text_descriptions = [text_descriptions]
        
        # 检查缓存
        cached_embeddings = []
        uncached_descriptions = []
        uncached_indices = []
        
        for i, desc in enumerate(text_descriptions):
            if desc in self.cache:
                cached_embeddings.append(self.cache[desc])
            else:
                uncached_descriptions.append(desc)
                uncached_indices.append(i)
        
        # 如果全部命中缓存
        if len(uncached_descriptions) == 0:
            return torch.stack(cached_embeddings)
        
        # 编码未缓存的描述
        new_embeddings = super().forward(uncached_descriptions)
        
        # 更新缓存
        for desc, emb in zip(uncached_descriptions, new_embeddings):
            self.cache[desc] = emb.detach()
        
        # 合并结果
        all_embeddings = []
        cached_idx = 0
        uncached_idx = 0
        
        for i in range(len(text_descriptions)):
            if i in uncached_indices:
                all_embeddings.append(new_embeddings[uncached_idx])
                uncached_idx += 1
            else:
                all_embeddings.append(cached_embeddings[cached_idx])
                cached_idx += 1
        
        return torch.stack(all_embeddings)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
    
    def save_cache(self, path):
        """保存缓存到文件"""
        torch.save(self.cache, path)
    
    def load_cache(self, path):
        """从文件加载缓存"""
        self.cache = torch.load(path)


# ============== 示例用法 ==============
if __name__ == '__main__':
    # 创建编码器
    encoder = MedicalTextEncoder(
        output_dim=256,
        freeze_bert=True
    )
    encoder.eval()
    
    # 示例1: 单个描述
    desc = "This CT scan was acquired using a 25% low-dose protocol. The image comes from Mayo Clinic 2016 dataset."
    embedding = encoder(desc)
    print(f"单个描述编码: {embedding.shape}")  # [1, 256]
    
    # 示例2: 批量描述
    descs = [
        "This CT scan was acquired using a 25% low-dose protocol.",
        "This CT scan was acquired using a 10% ultra-low-dose protocol.",
        "This CT scan was acquired using a full-dose protocol."
    ]
    embeddings = encoder(descs)
    print(f"批量描述编码: {embeddings.shape}")  # [3, 256]
    
    # 示例3: 从元数据直接编码
    embeddings = encoder.encode_batch(
        dose_list=[25, 25, 10],
        site_list=['mayo_2016', 'mayo_2020', 'mayo_2016']
    )
    print(f"从元数据编码: {embeddings.shape}")  # [3, 256]
    
    # 示例4: 验证embedding的语义相似性
    with torch.no_grad():
        emb_25 = encoder("25% low-dose protocol with increased noise")
        emb_10 = encoder("10% ultra-low-dose protocol with high noise")
        emb_full = encoder("full-dose protocol with standard quality")
        
        # 计算余弦相似度
        sim_25_10 = torch.nn.functional.cosine_similarity(emb_25, emb_10)
        sim_25_full = torch.nn.functional.cosine_similarity(emb_25, emb_full)
        
        print(f"\n语义相似度:")
        print(f"  25% vs 10% dose: {sim_25_10.item():.4f}")  # 应该较高
        print(f"  25% vs full dose: {sim_25_full.item():.4f}")  # 应该较低
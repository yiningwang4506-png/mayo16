"""
Medical Text Encoder using PubMed-BERT
使用 PubMed-BERT 将医学描述文本编码为条件向量
优化版：强制使用本地缓存，避免 Hugging Face 网络超时
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
                 cache_dir='./pretrained_models',
                 local_files_only=True):  # ← 新增参数，默认True
        """
        Args:
            model_name: Hugging Face 模型名称
                - 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext' (推荐)
                - 'michiyasunaga/BioLinkBERT-base'
                - 'dmis-lab/biobert-base-cased-v1.1'
            output_dim: 输出embedding维度 (默认256,与CoreDiff兼容)
            freeze_bert: 是否冻结BERT参数 (推荐True以节省显存)
            cache_dir: 模型缓存目录
            local_files_only: 是否仅使用本地缓存（默认True，避免网络超时）
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # ============ 关键修改：设置离线模式 ============
        if local_files_only:
            print("📦 Using LOCAL CACHE ONLY (offline mode)")
            # 双重保险：设置环境变量
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'
        
        # 加载预训练tokenizer和模型
        print(f"🔄 Loading medical text encoder: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=local_files_only  # ← 强制使用本地
            )
            self.bert = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=local_files_only  # ← 强制使用本地
            )
            print("✅ Loaded from local cache successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load from local cache!")
            print(f"   Error: {e}")
            print(f"\n💡 Please download the model first:")
            print(f"   python medical_text_encoder.py --download")
            raise
        
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
        前向传播（训练时使用）
        
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
    
    def encode(self, text_descriptions):
        """
        便捷编码方法（Dataset推理专用）
        在推理时自动处理维度，适合在Dataset中使用
        
        Args:
            text_descriptions: str or List[str] - 医学描述
            
        Returns:
            torch.Tensor: 
                - 如果输入是 str: 返回 [output_dim]
                - 如果输入是 List[str]: 返回 [batch_size, output_dim]
        """
        was_training = self.training
        self.eval()  # 切换到评估模式
        
        with torch.no_grad():
            embeddings = self.forward(text_descriptions)
            
            # 如果输入是单个字符串，去掉batch维度
            if isinstance(text_descriptions, str):
                embeddings = embeddings.squeeze(0)  # [output_dim]
        
        # 恢复原来的模式
        if was_training:
            self.train()
        
        return embeddings
    
    def encode_batch(self, dose_list, site_list, **kwargs):
        """
        便捷方法：直接从元数据编码
        
        Args:
            dose_list: List[int] - 剂量列表
            site_list: List[str] - 站点列表
            **kwargs: 其他参数传递给 TextDescriptionGenerator
            
        Returns:
            torch.Tensor: [batch_size, output_dim]
        """
        from text_description_generator import TextDescriptionGenerator
        
        # 生成描述
        generator = TextDescriptionGenerator()
        descriptions = [
            generator.generate_description(dose=dose, site=site, **kwargs)
            for dose, site in zip(dose_list, site_list)
        ]
        
        # 编码
        return self.encode(descriptions)


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
        print(f"✓ Cache cleared")
    
    def save_cache(self, path):
        """保存缓存到文件"""
        torch.save(self.cache, path)
        print(f"✓ Cache saved to {path} ({len(self.cache)} entries)")
    
    def load_cache(self, path):
        """从文件加载缓存"""
        self.cache = torch.load(path)
        print(f"✓ Cache loaded from {path} ({len(self.cache)} entries)")


# ============== 工具函数：首次下载模型 ==============
def download_model_if_needed(model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                             cache_dir='./pretrained_models'):
    """
    首次下载模型到本地缓存
    运行一次后，后续可以离线使用
    
    Usage:
        python medical_text_encoder.py --download
    """
    print(f"📥 Downloading model: {model_name}")
    print(f"📁 Cache directory: {cache_dir}")
    print(f"⏳ This may take a few minutes...")
    
    try:
        # 临时允许网络访问
        print("\n🔄 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False  # 允许下载
        )
        
        print("🔄 Downloading model...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False  # 允许下载
        )
        
        print("\n✅ Model downloaded successfully!")
        print(f"   Tokenizer vocab size: {tokenizer.vocab_size}")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        print(f"   Cache location: {cache_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print(f"💡 Please check your network connection and try again")
        return False


# ============== 示例用法 ==============
if __name__ == '__main__':
    import sys
    
    # 检查是否是下载模式
    if len(sys.argv) > 1 and sys.argv[1] == '--download':
        print("\n" + "="*60)
        print("🚀 Model Download Mode")
        print("="*60 + "\n")
        success = download_model_if_needed()
        sys.exit(0 if success else 1)
    
    # 正常测试模式
    print("\n" + "="*60)
    print("🧪 Testing MedicalTextEncoder (Offline Mode)")
    print("="*60 + "\n")
    
    try:
        # 创建编码器（使用本地缓存）
        encoder = MedicalTextEncoder(
            output_dim=256,
            freeze_bert=True,
            local_files_only=True  # 强制离线
        )
        encoder.eval()
        
        # 示例1: 单个描述（使用 encode 方法）
        print("\n📝 Test 1: Single Description (encode method)")
        desc = "This CT scan was acquired using a 25% low-dose protocol. The image comes from Mayo Clinic 2016 dataset."
        embedding = encoder.encode(desc)
        print(f"✅ Single description encoding: {embedding.shape}")  # [256]
        print(f"   Embedding norm: {embedding.norm().item():.4f}")
        
        # 示例2: 批量描述（使用 encode 方法）
        print("\n📝 Test 2: Batch Descriptions (encode method)")
        descs = [
            "This CT scan was acquired using a 25% low-dose protocol.",
            "This CT scan was acquired using a 10% ultra-low-dose protocol.",
            "This CT scan was acquired using a full-dose protocol."
        ]
        embeddings = encoder.encode(descs)
        print(f"✅ Batch description encoding: {embeddings.shape}")  # [3, 256]
        
        # 示例3: forward 方法（训练时）
        print("\n📝 Test 3: Forward Method (for training)")
        with torch.no_grad():
            embeddings_forward = encoder.forward(descs)
        print(f"✅ Forward method output: {embeddings_forward.shape}")  # [3, 256]
        
        # 示例4: 从元数据直接编码
        print("\n📝 Test 4: Encoding from Metadata")
        embeddings = encoder.encode_batch(
            dose_list=[25, 25, 10],
            site_list=['mayo_2016', 'mayo_2020', 'mayo_2016']
        )
        print(f"✅ Encoding from metadata: {embeddings.shape}")  # [3, 256]
        
        # 示例5: 验证embedding的语义相似性
        print("\n📝 Test 5: Semantic Similarity")
        with torch.no_grad():
            emb_25 = encoder.encode("25% low-dose protocol with increased noise")
            emb_10 = encoder.encode("10% ultra-low-dose protocol with high noise")
            emb_full = encoder.encode("full-dose protocol with standard quality")
            
            # 计算余弦相似度
            sim_25_10 = torch.nn.functional.cosine_similarity(
                emb_25.unsqueeze(0), emb_10.unsqueeze(0)
            ).item()
            sim_25_full = torch.nn.functional.cosine_similarity(
                emb_25.unsqueeze(0), emb_full.unsqueeze(0)
            ).item()
            
            print(f"  25% vs 10% dose: {sim_25_10:.4f} (should be high)")
            print(f"  25% vs full dose: {sim_25_full:.4f} (should be lower)")
        
        # 示例6: 测试维度处理
        print("\n📝 Test 6: Dimension Handling")
        single_emb = encoder.encode("Single description")
        batch_emb = encoder.encode(["Batch description"])
        print(f"  Single string → shape: {single_emb.shape}")  # [256]
        print(f"  List with 1 item → shape: {batch_emb.shape}")  # [1, 256]
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)
        print("\n💡 Usage in your code:")
        print("   text_embedding = encoder.encode(description)  # Returns [256]")
        print("   text_embeddings = encoder.encode([desc1, desc2])  # Returns [2, 256]")
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ Test failed!")
        print("="*60)
        print(f"\nError: {e}")
        print("\n💡 If model not found, please run:")
        print("   python medical_text_encoder.py --download")
        import traceback
        traceback.print_exc()
        sys.exit(1)
# utils/prompt_builder.py
"""
Prompt Builder for Multi-Dose CT Denoising
根据剂量等级构建医学文本描述
"""

class PromptBuilder:
    """
    单站点多剂量场景的 Prompt 构建器
    
    支持剂量: 25%, 50%, 100%
    """
    
    # 剂量等级的详细描述模板
    DOSE_TEMPLATES = {
        100: "This is a full-dose CT scan with standard radiation exposure, showing clear anatomical structures and minimal noise.",
        
        50: "This is a low-dose CT scan with 50% radiation dose, showing moderate noise levels and acceptable image quality.",
        
        25: "This is a low-dose CT scan with 25% radiation dose, showing significant noise and reduced image quality that requires denoising."
    }
    
    # 简化版本（用于消融实验）
    DOSE_TEMPLATES_SIMPLE = {
        100: "This is a full-dose CT scan.",
        50: "This is a low-dose CT scan with 50% radiation dose.",
        25: "This is a low-dose CT scan with 25% radiation dose."
    }
    
    @staticmethod
    def build_prompt(dose: int, mode: str = 'detailed') -> str:
        """
        根据剂量构建文本描述
        
        Args:
            dose: 剂量百分比 (25, 50, 100)
            mode: 'detailed' 或 'simple'
        
        Returns:
            prompt: 医学文本描述
        
        Examples:
            >>> builder = PromptBuilder()
            >>> builder.build_prompt(25)
            "This is a low-dose CT scan with 25% radiation dose, ..."
            >>> builder.build_prompt(100, mode='simple')
            "This is a full-dose CT scan."
        """
        templates = (PromptBuilder.DOSE_TEMPLATES if mode == 'detailed' 
                    else PromptBuilder.DOSE_TEMPLATES_SIMPLE)
        
        # 容错：如果剂量不在预定义范围，使用最接近的
        if dose not in templates:
            if dose >= 75:
                dose = 100
            elif dose >= 37:
                dose = 50
            else:
                dose = 25
        
        return templates[dose]
    
    @staticmethod
    def get_dose_category(dose: int) -> str:
        """
        获取剂量类别（用于分析和可视化）
        
        Args:
            dose: 剂量百分比
        
        Returns:
            category: 'full_dose', 'medium_dose', 'low_dose'
        """
        if dose == 100:
            return "full_dose"
        elif dose >= 50:
            return "medium_dose"
        else:
            return "low_dose"
    
    @staticmethod
    def batch_build_prompts(doses: list, mode: str = 'detailed') -> list:
        """
        批量构建 prompts
        
        Args:
            doses: 剂量列表, e.g., [25, 50, 100]
            mode: 'detailed' 或 'simple'
        
        Returns:
            prompts: 文本描述列表
        """
        return [PromptBuilder.build_prompt(d, mode) for d in doses]


# 测试代码
if __name__ == '__main__':
    builder = PromptBuilder()
    
    print("="*70)
    print("测试 Prompt Builder")
    print("="*70)
    
    doses = [25, 50, 100]
    
    print("\n【详细模式】")
    for dose in doses:
        prompt = builder.build_prompt(dose, mode='detailed')
        print(f"\n{dose}% 剂量:")
        print(f"  {prompt}")
    
    print("\n" + "="*70)
    print("【简单模式】")
    for dose in doses:
        prompt = builder.build_prompt(dose, mode='simple')
        print(f"\n{dose}% 剂量:")
        print(f"  {prompt}")
    
    print("\n" + "="*70)
    print("【批量构建】")
    prompts = builder.batch_build_prompts([25, 25, 50, 100])
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:50]}...")
    
    print("\n✅ Prompt Builder 测试完成！")
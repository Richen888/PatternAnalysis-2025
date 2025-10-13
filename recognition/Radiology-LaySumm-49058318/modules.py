import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class RadiologySimplifierModel:
    def __init__(self, model_name="google/flan-t5-base", use_lora=True):
        self.model_name = model_name
        self.use_lora = use_lora
        
        # 初始化tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # 应用LoRA如果启用
        if use_lora:
            self._setup_lora()
        
        # 计算参数数量
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _setup_lora(self):
        """设置LoRA配置"""
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q", "v"]
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def get_model_info(self):
        """返回模型信息"""
        return {
            "model_name": self.model_name,
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "use_lora": self.use_lora
        }
    
    def save_model(self, save_path):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path):
        """加载模型"""
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

def create_model(model_name="google/flan-t5-base", use_lora=True):
    """创建模型实例的工厂函数"""
    return RadiologySimplifierModel(model_name, use_lora)
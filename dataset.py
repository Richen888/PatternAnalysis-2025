from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class RadiologyDataset:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset = None
    
    def load_data(self):
        """加载BioLaySumm数据集"""
        logger.info("Loading BioLaySumm dataset...")
        try:
            self.dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
            logger.info("Dataset loaded successfully")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_function(self, examples):
        """预处理函数"""
        inputs = examples["expert_report"]
        targets = examples["lay_report"]
        
        # 对输入进行tokenize
        model_inputs = self.tokenizer(
            inputs, 
            max_length=512, 
            padding=False, 
            truncation=True
        )
        
        # 对目标进行tokenize
        labels = self.tokenizer(
            targets, 
            max_length=256, 
            padding=False, 
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def get_tokenized_datasets(self):
        """获取tokenized数据集"""
        if self.dataset is None:
            self.load_data()
        
        # 预处理数据集
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names,
        )
        
        return tokenized_datasets
    
    def get_dataset_info(self):
        """返回数据集信息"""
        if self.dataset is None:
            self.load_data()
        
        info = {
            "train_samples": len(self.dataset["train"]),
            "validation_samples": len(self.dataset["validation"]),
            "test_samples": len(self.dataset["test"]),
            "columns": self.dataset["train"].column_names
        }
        return info

def create_dataset(model_name="google/flan-t5-base"):
    """创建数据集实例的工厂函数"""
    return RadiologyDataset(model_name)
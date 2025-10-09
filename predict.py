import torch
import logging
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate
import pandas as pd

from modules import create_model
from utils import setup_logging

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

class RadiologySimplifierInference:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rouge = evaluate.load('rouge')
        
        logger.info(f"Model loaded from: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def simplify_report(self, expert_report, max_length=256, num_beams=4):
        """简化放射学报告"""
        inputs = self.tokenizer(
            expert_report, 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                temperature=0.7,
                do_sample=True
            )
        
        simplified = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return simplified
    
    def evaluate_on_test_set(self, num_examples=50):
        """在测试集上评估并生成示例"""
        try:
            dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
            test_data = dataset["test"]
        except Exception as e:
            logger.error(f"Error loading test dataset: {e}")
            return None, []
        
        # 随机选择一些样本
        indices = random.sample(range(len(test_data)), min(num_examples, len(test_data)))
        
        predictions = []
        references = []
        examples = []
        
        logger.info(f"Evaluating on {len(indices)} examples...")
        
        for i, idx in enumerate(indices[:10]):  # 只处理前10个用于示例
            expert = test_data[idx]["expert_report"]
            reference_lay = test_data[idx]["lay_report"]
            
            prediction = self.simplify_report(expert)
            
            predictions.append(prediction)
            references.append(reference_lay)
            
            examples.append({
                "expert": expert,
                "predicted_lay": prediction,
                "reference_lay": reference_lay
            })
            
            if i % 5 == 0:
                logger.info(f"Processed {i+1} examples")
        
        # 计算ROUGE分数
        if predictions and references:
            rouge_scores = self.rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            rouge_scores = {key: value * 100 for key, value in rouge_scores.items()}
        else:
            rouge_scores = {}
        
        return rouge_scores, examples
    
    def generate_examples(self, num_examples=5):
        """生成代表性示例"""
        try:
            dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
            test_data = dataset["test"]
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
        
        examples = []
        indices = random.sample(range(len(test_data)), min(num_examples, len(test_data)))
        
        for idx in indices:
            expert = test_data[idx]["expert_report"]
            reference_lay = test_data[idx]["lay_report"]
            prediction = self.simplify_report(expert)
            
            examples.append({
                "expert": expert,
                "predicted_lay": prediction,
                "reference_lay": reference_lay
            })
        
        return examples

def main():
    """主推理函数"""
    model_path = "./radiology-simplifier-output"  # 修改为您的模型路径
    
    # 初始化推理器
    inference = RadiologySimplifierInference(model_path)
    
    # 评估模型
    rouge_scores, examples = inference.evaluate_on_test_set(num_examples=30)
    
    # 打印结果
    print("=" * 80)
    print("最终评估结果")
    print("=" * 80)
    if rouge_scores:
        print(f"ROUGE-1: {rouge_scores.get('rouge1', 0):.2f}")
        print(f"ROUGE-2: {rouge_scores.get('rouge2', 0):.2f}")
        print(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.2f}")
        print(f"ROUGE-Lsum: {rouge_scores.get('rougeLsum', 0):.2f}")
    else:
        print("无法计算ROUGE分数")
    
    print("\n" + "=" * 80)
    print("代表性示例")
    print("=" + 80)
    
    for i, example in enumerate(examples[:5], 1):
        print(f"\n示例 {i}:")
        print(f"专家报告: {example['expert'][:200]}...")
        print(f"预测简化: {example['predicted_lay']}")
        print(f"参考简化: {example['reference_lay']}")
        print("-" * 80)
    
    # 错误分析
    print("\n" + "=" * 80)
    print("错误分析")
    print("=" * 80)
    print("""
    常见错误类型:
    1. 医学术语简化不足 - 模型可能保留了某些专业术语
    2. 信息丢失 - 简化过程中可能遗漏重要临床细节
    3. 过度简化 - 可能过度简化导致信息不准确
    4. 流畅性问题 - 生成的文本可能不够自然流畅
    5. 长度控制 - 生成的摘要可能过长或过短
    """)

if __name__ == "__main__":
    main()
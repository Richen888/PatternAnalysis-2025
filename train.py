import torch
import logging
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import os
import time

from modules import create_model
from dataset import create_dataset
from utils import setup_logging, get_gpu_info

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 下载NLTK数据
nltk.download('punkt')

class RadiologyTrainer:
    def __init__(self, model_name="google/flan-t5-base", use_lora=True, output_dir="./output"):
        self.model_name = model_name
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.trainer = None
        
        # 初始化模型和数据集
        self.model = create_model(model_name, use_lora)
        self.dataset = create_dataset(model_name)
        
        # 初始化ROUGE评估器
        self.rouge = evaluate.load('rouge')
        
        # 记录训练开始时间
        self.start_time = None
        
    def compute_metrics(self, eval_pred):
        """计算评估指标"""
        predictions, labels = eval_pred
        
        # 解码预测
        decoded_preds = self.model.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # 将labels中的-100替换为pad_token_id
        labels = np.where(labels != -100, labels, self.model.tokenizer.pad_token_id)
        decoded_labels = self.model.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # 简单的后处理：确保每个句子以句号结束
        decoded_preds = [pred.strip() + '.' if pred.strip() and not pred.strip().endswith('.') else pred.strip() 
                        for pred in decoded_preds]
        decoded_labels = [label.strip() + '.' if label.strip() and not label.strip().endswith('.') else label.strip() 
                         for label in decoded_labels]
        
        # 计算ROUGE分数
        result = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
            use_aggregator=True
        )
        
        # 提取分数
        result = {key: value * 100 for key, value in result.items()}
        
        # 添加生成文本的长度
        prediction_lens = [np.count_nonzero(pred != self.model.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return result
    
    def setup_training(self, training_args=None):
        """设置训练参数"""
        # 获取tokenized数据集
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        
        # 默认训练参数
        if training_args is None:
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                overwrite_output_dir=True,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=5e-5,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=4,
                weight_decay=0.01,
                save_total_limit=3,
                num_train_epochs=10,
                predict_with_generate=True,
                logging_dir=f"{self.output_dir}/logs",
                logging_steps=50,
                load_best_model_at_end=True,
                metric_for_best_model="rougeL",
                greater_is_better=True,
                report_to="none",
                fp16=torch.cuda.is_available(),
                dataloader_num_workers=4,
                gradient_accumulation_steps=2,
                warmup_steps=100,
            )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        # 创建trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        return self.trainer
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            self.setup_training()
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 打印GPU信息
        gpu_info = get_gpu_info()
        logger.info(f"Training on device: {gpu_info}")
        
        # 打印模型信息
        model_info = self.model.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # 打印数据集信息
        dataset_info = self.dataset.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")
        
        # 开始训练
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # 计算总训练时间
        training_time = time.time() - self.start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # 保存最终模型
        self.trainer.save_model()
        self.model.tokenizer.save_pretrained(self.output_dir)
        
        # 记录训练结果
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        logger.info(f"Final model saved to: {self.output_dir}")
        
        return train_result, metrics
    
    def evaluate(self):
        """在测试集上评估模型"""
        if self.trainer is None:
            raise ValueError("Trainer not initialized. Please run setup_training() first.")
        
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(
            eval_dataset=tokenized_datasets["test"],
            metric_key_prefix="test"
        )
        
        logger.info(f"Test results: {test_results}")
        return test_results

def main():
    """主训练函数"""
    # 初始化训练器
    trainer = RadiologyTrainer(
        model_name="google/flan-t5-base",
        use_lora=True,
        output_dir="./radiology-simplifier-output"
    )
    
    # 开始训练
    train_result, metrics = trainer.train()
    
    # 在测试集上评估
    test_results = trainer.evaluate()
    
    # 打印最终结果
    logger.info("=== FINAL RESULTS ===")
    logger.info(f"Training metrics: {metrics}")
    logger.info(f"Test results: {test_results}")

if __name__ == "__main__":
    main()
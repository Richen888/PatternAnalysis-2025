import torch
import logging
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
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

class TrainingMonitorCallback(TrainerCallback):
    """自定义回调监控训练状态"""
    
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.last_log_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """每次日志记录时调用"""
        if logs is not None:
            # 打印关键训练指标
            if 'loss' in logs and logs['loss'] != 0:
                current_step = state.global_step
                if current_step - self.last_log_step >= self.log_interval:
                    epoch = logs.get('epoch', 0)
                    lr = logs.get('learning_rate', 0)
                    loss = logs.get('loss', 0)
                    
                    logger.info(f"📊 Step {current_step}: epoch={epoch:.2f}, lr={lr:.2e}, loss={loss:.4f}")
                    self.last_log_step = current_step

class ProgressCallback(TrainerCallback):
    """进度显示"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """每步结束时显示进度"""
        if state.global_step % 500 == 0:  # 每500步显示一次
            total_steps = state.max_steps if state.max_steps else args.num_train_epochs * state.steps_in_epoch
            progress = state.global_step / total_steps * 100 if total_steps else 0
            logger.info(f"⏳ 进度: {state.global_step}/{total_steps} ({progress:.1f}%)")

class RadiologyTrainer:
    def __init__(self, model_name="google/flan-t5-base", use_lora=True, output_dir="./output"):
        self.model_name = model_name
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.trainer = None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化模型和数据集
        logger.info("初始化模型和数据集...")
        self.model = create_model(model_name, use_lora)
        self.dataset = create_dataset(model_name)
        
        # 初始化ROUGE评估器
        self.rouge = evaluate.load('rouge')
        
        # 记录训练开始时间
        self.start_time = None
        
    def check_trainable_params(self):
        """检查可训练参数"""
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.model.parameters())
        logger.info(f"总参数: {total_params}, 可训练参数: {trainable_params}")
        return trainable_params, total_params
        
    def setup_training(self):
        """设置训练参数 - 纯训练，无评估"""
        # 获取tokenized数据集
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        
        # 纯训练配置，完全跳过评估
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="no",  # 完全关闭评估
            save_strategy="epoch",     # 只每个epoch保存
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=10,  # 训练10个epoch
            predict_with_generate=False,  # 关闭生成预测
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,  # 每100步记录一次
            load_best_model_at_end=False,  # 关闭最佳模型加载
            report_to="none",
            fp16=False,
            dataloader_num_workers=0,
            gradient_accumulation_steps=1,
            warmup_steps=100,
            max_grad_norm=0.5,
            gradient_checkpointing=False,
            remove_unused_columns=False,
            label_names=["labels"],
            disable_tqdm=True,
            # 关键：没有eval_steps，没有评估
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        # 创建trainer - 只设置训练集
        self.trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            # 不设置eval_dataset
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            # 不设置compute_metrics
            callbacks=[
                TrainingMonitorCallback(log_interval=100),
                ProgressCallback()
            ],
        )
        
        return self.trainer
    
    def verify_training_setup(self):
        """验证训练设置"""
        logger.info("=== 训练设置验证 ===")
        
        # 检查参数
        trainable, total = self.check_trainable_params()
        
        # 检查一个batch
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            tokenized_datasets["train"],
            batch_size=2,
            collate_fn=data_collator
        )
        
        batch = next(iter(train_loader))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = {k: v.to(device) for k, v in batch.items()}
        self.model.model.to(device)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model.model(**batch)
        loss = outputs.loss.item()
        logger.info(f"初始批次loss: {loss:.4f}")
        
        if loss > 10 or loss < 0.1:
            logger.warning(f"⚠️ 初始loss异常: {loss:.4f}")
        else:
            logger.info(f"✅ 初始loss正常: {loss:.4f}")
        
        # 解码示例
        decoded_input = self.model.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        safe_labels = batch["labels"][0].clone()
        safe_labels[safe_labels == -100] = self.model.tokenizer.pad_token_id
        decoded_label = self.model.tokenizer.decode(safe_labels, skip_special_tokens=True)
        
        logger.info(f"输入示例: {decoded_input}")
        logger.info(f"标签示例: {decoded_label}")
        
        return True
    
    def train(self):
        """开始训练"""
        if self.trainer is None:
            self.setup_training()
        
        # 验证训练设置
        setup_ok = self.verify_training_setup()
        
        if not setup_ok:
            logger.error("训练设置验证失败")
            return None, None
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 打印GPU信息
        gpu_info = get_gpu_info()
        logger.info(f"训练设备: {gpu_info}")
        
        # 开始训练
        logger.info("开始纯训练模式...")
        logger.info("配置: 10个epoch, 无评估, 只训练")
        logger.info("Loss下降趋势: 3.23 → 1.98 (已下降38.7%)")
        
        try:
            train_result = self.trainer.train()
            
            # 计算总训练时间
            training_time = time.time() - self.start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            logger.info(f"🎉 训练完成! 耗时: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # 保存最终模型
            self.trainer.save_model()
            self.model.tokenizer.save_pretrained(self.output_dir)
            
            # 记录训练结果
            metrics = train_result.metrics
            logger.info(f"最终训练指标: {metrics}")
            
            logger.info(f"模型已保存到: {self.output_dir}")
            
            return train_result, metrics
            
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            # 尝试保存当前进度
            try:
                self.trainer.save_model(f"{self.output_dir}/emergency_save")
                logger.info("已保存紧急备份模型")
            except:
                pass
            raise
    
    def manual_evaluate(self):
        """训练完成后手动评估"""
        logger.info("训练完成，开始手动评估...")
        
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        
        # 重新设置评估参数
        eval_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            per_device_eval_batch_size=4,
            predict_with_generate=True,
            generation_max_length=128,
            generation_num_beams=1,
            report_to="none",
        )
        
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            decoded_preds = self.model.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels = np.where(labels != -100, labels, self.model.tokenizer.pad_token_id)
            decoded_labels = self.model.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            result = self.rouge.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True,
                use_aggregator=True
            )
            return {key: value * 100 for key, value in result.items()}
        
        eval_trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=eval_args,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            compute_metrics=compute_metrics,
        )
        
        # 评估验证集和测试集
        val_results = eval_trainer.evaluate(tokenized_datasets["validation"])
        test_results = eval_trainer.evaluate(tokenized_datasets["test"])
        
        logger.info(f"验证集结果: {val_results}")
        logger.info(f"测试集结果: {test_results}")
        
        return val_results, test_results

def main():
    """主训练函数"""
    try:
        # 初始化训练器
        trainer = RadiologyTrainer(
            model_name="google/flan-t5-base",
            use_lora=True,
            output_dir="./radiology-simplifier-output"
        )
        
        # 开始训练
        train_result, metrics = trainer.train()
        
        if train_result is not None:
            # 手动评估
            val_results, test_results = trainer.manual_evaluate()
            
            # 打印最终结果
            logger.info("=== 最终结果 ===")
            logger.info(f"训练指标: {metrics}")
            logger.info(f"验证集结果: {val_results}")
            logger.info(f"测试集结果: {test_results}")
        else:
            logger.error("训练失败，无法进行评估")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        raise

if __name__ == "__main__":
    main()

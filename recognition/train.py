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

# Setup logging configuration
setup_logging()
logger = logging.getLogger(__name__)

# Download NLTK punkt tokenizer for sentence segmentation
nltk.download('punkt')

class TrainingMonitorCallback(TrainerCallback):
    """
    Custom callback for monitoring training progress and logging key metrics.
    
    This callback tracks training metrics and logs them at specified intervals
    to provide visibility into the training process without overwhelming the logs.
    """
    
    def __init__(self, log_interval=100):
        """
        Initialize the training monitor callback.
        
        Args:
            log_interval (int): Number of steps between each log entry. Defaults to 100.
        """
        self.log_interval = log_interval
        self.last_log_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when training logs are updated.
        
        Args:
            args: Training arguments
            state: Training state
            control: Training control object
            logs: Dictionary containing current training metrics
        """
        if logs is not None:
            if 'loss' in logs and logs['loss'] != 0:
                current_step = state.global_step
                if current_step - self.last_log_step >= self.log_interval:
                    epoch = logs.get('epoch', 0)
                    lr = logs.get('learning_rate', 0)
                    loss = logs.get('loss', 0)
                    
                    logger.info(f"📊 Step {current_step}: epoch={epoch:.2f}, lr={lr:.2e}, loss={loss:.4f}")
                    self.last_log_step = current_step

class ProgressCallback(TrainerCallback):
    """
    Callback for displaying training progress at regular intervals.
    
    Provides periodic updates on training progress including completion percentage
    and current step count.
    """
    
    def on_step_end(self, args, state, control, **kwargs):
        """
        Display progress information every 500 training steps.
        
        Args:
            args: Training arguments
            state: Training state containing current step information
            control: Training control object
        """
        if state.global_step % 500 == 0:
            total_steps = state.max_steps if state.max_steps else args.num_train_epochs * state.steps_in_epoch
            progress = state.global_step / total_steps * 100 if total_steps else 0
            logger.info(f"⏳ Progress: {state.global_step}/{total_steps} ({progress:.1f}%)")

class RadiologyTrainer:
    """
    Main trainer class for fine-tuning radiology report simplification models.
    
    This class handles the complete training pipeline including model initialization,
    dataset preparation, training configuration, and training execution. It supports
    LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
    """
    
    def __init__(self, model_name="google/flan-t5-base", use_lora=True, output_dir="./output"):
        """
        Initialize the radiology trainer.
        
        Args:
            model_name (str): Name of the pre-trained model to use. Defaults to "google/flan-t5-base".
            use_lora (bool): Whether to apply LoRA for efficient fine-tuning. Defaults to True.
            output_dir (str): Directory path where model checkpoints and outputs will be saved.
        """
        self.model_name = model_name
        self.use_lora = use_lora
        self.output_dir = output_dir
        self.trainer = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model and dataset
        logger.info("Initializing model and dataset...")
        self.model = create_model(model_name, use_lora)
        self.dataset = create_dataset(model_name)
        
        # Record training start time for duration calculation
        self.start_time = None
        
    def check_trainable_params(self):
        """
        Calculate and log the number of trainable and total parameters.
        
        Returns:
            tuple: (trainable_parameters, total_parameters)
        """
        trainable_params = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.model.parameters())
        logger.info(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
        return trainable_params, total_params
        
    def setup_training(self):
        """
        Configure and setup the training pipeline.
        
        This method sets up the training arguments, data collator, and trainer instance
        for a training-only configuration without evaluation during training.
        
        Returns:
            Seq2SeqTrainer: Configured trainer instance ready for training
        """
        # Get tokenized datasets from the dataset module
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        
        # Training-only configuration - completely disables evaluation during training
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="no",  # Completely disable evaluation during training
            save_strategy="epoch",     # Save checkpoints only at the end of each epoch
            learning_rate=5e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            weight_decay=0.01,
            save_total_limit=3,        # Keep only the 3 most recent checkpoints
            num_train_epochs=10,       # Train for 10 complete epochs
            predict_with_generate=False,  # Disable text generation during training for efficiency
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=100,         # Log training metrics every 100 steps
            load_best_model_at_end=False,  # Disable best model loading since no evaluation
            report_to="none",          # Disable external reporting (e.g., Weights & Biases)
            fp16=False,                # Disable mixed-precision training
            dataloader_num_workers=0,  # Disable multiprocessing for data loading
            gradient_accumulation_steps=1,
            warmup_steps=100,          # Number of warmup steps for learning rate scheduler
            max_grad_norm=0.5,         # Gradient clipping threshold
            gradient_checkpointing=False,  # Disable gradient checkpointing for memory efficiency
            remove_unused_columns=False,  # Keep all columns from the dataset
            label_names=["labels"],    # Specify which columns contain the labels
            disable_tqdm=True,         # Disable progress bars for cleaner logs
        )
        
        # Data collator for dynamic padding and preparing batches for seq2seq training
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        # Create trainer instance with only the training dataset
        self.trainer = Seq2SeqTrainer(
            model=self.model.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],  # Use only training dataset
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
            callbacks=[
                TrainingMonitorCallback(log_interval=100),
                ProgressCallback()
            ],
        )
        
        return self.trainer
    
    def verify_training_setup(self):
        """
        Perform comprehensive verification of the training setup.
        
        This method checks parameter counts, tests a forward pass with a sample batch,
        and verifies that data processing is working correctly before starting full training.
        
        Returns:
            bool: True if setup verification passes, False otherwise
        """
        logger.info("=== Training Setup Verification ===")
        
        # Check parameter counts to verify LoRA configuration
        trainable, total = self.check_trainable_params()
        
        # Test data loading and model forward pass with a sample batch
        tokenized_datasets = self.dataset.get_tokenized_datasets()
        data_collator = DataCollatorForSeq2Seq(
            self.model.tokenizer,
            model=self.model.model,
            padding=True,
        )
        
        # Create a small data loader for testing
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            tokenized_datasets["train"],
            batch_size=2,
            collate_fn=data_collator
        )
        
        # Get a sample batch and move to appropriate device
        batch = next(iter(train_loader))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch = {k: v.to(device) for k, v in batch.items()}
        self.model.model.to(device)
        
        # Perform a forward pass to verify model can process the data
        with torch.no_grad():
            outputs = self.model.model(**batch)
        loss = outputs.loss.item()
        logger.info(f"Initial batch loss: {loss:.4f}")
        
        # Check if initial loss is within expected range
        if loss > 10 or loss < 0.1:
            logger.warning(f"⚠️ Unusual initial loss: {loss:.4f}")
        else:
            logger.info(f"✅ Normal initial loss: {loss:.4f}")
        
        # Decode and log examples to verify data processing
        decoded_input = self.model.tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
        safe_labels = batch["labels"][0].clone()
        safe_labels[safe_labels == -100] = self.model.tokenizer.pad_token_id
        decoded_label = self.model.tokenizer.decode(safe_labels, skip_special_tokens=True)
        
        logger.info(f"Input example: {decoded_input[:100]}...")
        logger.info(f"Label example: {decoded_label[:100]}...")
        
        return True
    
    def train(self):
        """
        Execute the training process.
        
        This method performs setup verification, starts training, monitors progress,
        and handles training completion including model saving and logging.
        
        Returns:
            TrainOutput: Training results including metrics and state, or None if training fails
        """
        if self.trainer is None:
            self.setup_training()
        
        # Verify training setup before starting
        setup_ok = self.verify_training_setup()
        
        if not setup_ok:
            logger.error("Training setup verification failed")
            return None
        
        # Record start time for training duration calculation
        self.start_time = time.time()
        
        # Print GPU information for resource monitoring
        gpu_info = get_gpu_info()
        logger.info(f"Training device: {gpu_info}")
        
        # Start training process
        logger.info("Starting training-only mode...")
        logger.info("Configuration: 10 epochs, no evaluation, training only")
        
        try:
            # Execute the training loop
            train_result = self.trainer.train()
            
            # Calculate and log total training time
            training_time = time.time() - self.start_time
            hours = int(training_time // 3600)
            minutes = int((training_time % 3600) // 60)
            seconds = int(training_time % 60)
            
            logger.info(f"🎉 Training completed! Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Save the final trained model and tokenizer
            self.trainer.save_model()
            self.model.tokenizer.save_pretrained(self.output_dir)
            
            # Log final training metrics
            metrics = train_result.metrics
            logger.info(f"Final training metrics: {metrics}")
            
            logger.info(f"Model saved to: {self.output_dir}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            # Attempt to save current progress in case of failure
            try:
                self.trainer.save_model(f"{self.output_dir}/emergency_save")
                logger.info("Emergency model backup saved")
            except:
                logger.warning("Failed to save emergency backup")
            raise

def main():
    """
    Main execution function for training the radiology report simplification model.
    
    This function orchestrates the complete training pipeline including initialization,
    training execution, and result logging.
    """
    try:
        # Initialize the radiology trainer with specified configuration
        trainer = RadiologyTrainer(
            model_name="google/flan-t5-base",
            use_lora=True,
            output_dir="./radiology-simplifier-output"
        )
        
        # Execute the training process
        train_result = trainer.train()
        
        # Log final results
        if train_result is not None:
            logger.info("=== Training Completed Successfully ===")
            logger.info(f"Training metrics: {train_result.metrics}")
        else:
            logger.error("Training failed")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    # Entry point for script execution
    main()
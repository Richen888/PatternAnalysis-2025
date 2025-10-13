from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

class RadiologyDataset:
    """
    Dataset class for handling radiology report simplification data.
    Loads and preprocesses the BioLaySumm dataset for training.
    """
    
    def __init__(self, model_name="google/flan-t5-base"):
        """Initialize the dataset with tokenizer for the specified model."""
        # Load tokenizer for the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Placeholder for the dataset - will be loaded later
        self.dataset = None
    
    def load_data(self):
        """Load BioLaySumm dataset from Hugging Face Hub."""
        logger.info("Loading BioLaySumm dataset...")
        try:
            # Load the dataset from Hugging Face dataset hub
            self.dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
            logger.info("Dataset loaded successfully")
            return self.dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_function(self, examples):
        """
        Preprocessing function for tokenizing the dataset.
        Ensures that tokenized labels are properly set for T5 training.
        """
        inputs = examples["radiology_report"]   # 原文报告
        targets = examples["layman_report"]     # 简化摘要

        # 对输入文本进行 tokenization
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            padding=False,   # 不在这里 padding
            truncation=True
        )

        # 对目标文本进行 tokenization，确保不全是 -100
        labels = self.tokenizer(
            targets,
            max_length=256,
            padding=False,
            truncation=True
        )["input_ids"]

        # 只将 pad token 设置为 -100，其他保留
        labels = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label_seq]
            for label_seq in labels
        ]

        model_inputs["labels"] = labels
        return model_inputs

    def get_tokenized_datasets(self):
        """
        Apply tokenization to the entire dataset.
        
        Returns:
            DatasetDict: Tokenized datasets for train/validation/test splits
        """
        # Load data if not already loaded
        if self.dataset is None:
            self.load_data()
        
        # Preprocess datasets by applying tokenization function
        tokenized_datasets = self.dataset.map(
            self.preprocess_function,
            batched=True,                    # Process in batches for efficiency
            remove_columns=self.dataset["train"].column_names,  # Remove original columns
        )
        
        return tokenized_datasets
    
    def get_dataset_info(self):
        """
        Get basic information about the dataset.
        
        Returns:
            dict: Dataset statistics including sample counts and columns
        """
        # Load data if not already loaded
        if self.dataset is None:
            self.load_data()
        
        # Compile dataset information
        info = {
            "train_samples": len(self.dataset["train"]),          # Number of training samples
            "validation_samples": len(self.dataset["validation"]), # Number of validation samples
            "test_samples": len(self.dataset["test"]),            # Number of test samples
            "columns": self.dataset["train"].column_names         # Original column names
        }
        return info

def create_dataset(model_name="google/flan-t5-base"):
    """
    Factory function to create dataset instance.
    
    Args:
        model_name (str): Name of the model to use for tokenization
        
    Returns:
        RadiologyDataset: Initialized dataset instance
    """
    return RadiologyDataset(model_name)
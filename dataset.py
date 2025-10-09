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
        
        Args:
            examples: Batch of examples from the dataset
            
        Returns:
            dict: Tokenized inputs with labels
        """
        # Extract the source (radiology report) and target (layman summary) texts
        inputs = examples["radiology_report"]   # Source: original radiology report
        targets = examples["layman_report"]     # Target: simplified layman report
        
        # Tokenize the input radiology reports
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,          # Limit input length to 512 tokens
            padding=False,           # No padding here (will be done later in DataCollator)
            truncation=True          # Truncate texts longer than max_length
        )

        # Tokenize the target layman summaries
        labels = self.tokenizer(
            targets,
            max_length=256,          # Shorter max length for targets (summaries)
            padding=False,           # No padding for labels
            truncation=True          # Truncate long summaries
        )

        # Add tokenized target IDs as labels for the model
        model_inputs["labels"] = labels["input_ids"]

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

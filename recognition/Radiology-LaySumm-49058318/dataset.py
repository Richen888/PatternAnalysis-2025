from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)

class RadiologyDataset:
    """
    Dataset class for handling radiology report simplification data.
    
    This class manages the loading, preprocessing, and tokenization of the BioLaySumm dataset
    for training sequence-to-sequence models on radiology report simplification tasks.
    
    Attributes:
        tokenizer: Hugging Face tokenizer for text processing
        dataset: Loaded dataset object containing train/validation/test splits
    """
    
    def __init__(self, model_name="google/flan-t5-base"):
        """
        Initialize the dataset with tokenizer for the specified model.
        
        Args:
            model_name (str): Name of the pre-trained model to use for tokenization.
                            Defaults to "google/flan-t5-base".
        """
        # Load tokenizer for the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Placeholder for the dataset - will be loaded later
        self.dataset = None
    
    def load_data(self):
        """
        Load BioLaySumm dataset from Hugging Face Hub.
        
        Returns:
            DatasetDict: Loaded dataset with train/validation/test splits
            
        Raises:
            Exception: If dataset loading fails
        """
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
        
        This function tokenizes both input radiology reports and target layman summaries,
        and prepares the labels for sequence-to-sequence training with proper masking.
        
        Args:
            examples (dict): Batch of examples containing 'radiology_report' and 'layman_report'
            
        Returns:
            dict: Tokenized inputs with labels ready for model training
        """
        # Extract source and target texts from the batch
        inputs = examples["radiology_report"]   # Original radiology reports
        targets = examples["layman_report"]     # Simplified layman summaries

        # Tokenize input texts (radiology reports)
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,    # Maximum sequence length for inputs
            padding=False,     # Defer padding to data collator
            truncation=True    # Truncate sequences longer than max_length
        )

        # Tokenize target texts (layman summaries)
        labels = self.tokenizer(
            targets,
            max_length=256,    # Maximum sequence length for targets
            padding=False,     # Defer padding to data collator
            truncation=True    # Truncate sequences longer than max_length
        )["input_ids"]

        # Replace padding tokens with -100 for loss calculation
        # -100 is ignored by the loss function in PyTorch
        labels = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label_seq]
            for label_seq in labels
        ]

        model_inputs["labels"] = labels
        return model_inputs

    def get_tokenized_datasets(self):
        """
        Apply tokenization to the entire dataset.
        
        This method processes all dataset splits (train/validation/test) through
        the preprocessing function and returns tokenized versions ready for training.
        
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
        Get basic information and statistics about the dataset.
        
        Returns:
            dict: Dataset statistics including sample counts and column names
        """
        # Load data if not already loaded
        if self.dataset is None:
            self.load_data()
        
        # Compile dataset information and statistics
        info = {
            "train_samples": len(self.dataset["train"]),          # Number of training samples
            "validation_samples": len(self.dataset["validation"]), # Number of validation samples
            "test_samples": len(self.dataset["test"]),            # Number of test samples
            "columns": self.dataset["train"].column_names         # Original column names
        }
        return info

def create_dataset(model_name="google/flan-t5-base"):
    """
    Factory function to create and initialize a RadiologyDataset instance.
    
    This function provides a convenient interface for creating dataset instances
    with the specified tokenizer configuration.
    
    Args:
        model_name (str): Name of the pre-trained model to use for tokenization.
                        Defaults to "google/flan-t5-base".
                        
    Returns:
        RadiologyDataset: Initialized dataset instance ready for data loading
    """
    return RadiologyDataset(model_name)

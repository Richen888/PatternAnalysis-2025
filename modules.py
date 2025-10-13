import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class RadiologySimplifierModel:
    """
    A model for simplifying radiology reports using sequence-to-sequence transformers.
    
    This class wraps a transformer model (default: FLAN-T5) with optional LoRA (Low-Rank Adaptation)
    for efficient fine-tuning. It provides methods for model setup, training, and inference
    specifically tailored for radiology report simplification tasks.
    
    Attributes:
        model_name (str): Name of the pre-trained model to use
        use_lora (bool): Whether to apply LoRA for parameter-efficient fine-tuning
        tokenizer: Hugging Face tokenizer for the model
        model: The sequence-to-sequence model instance
        total_params (int): Total number of parameters in the model
        trainable_params (int): Number of trainable parameters (affected by LoRA)
    """
    
    def __init__(self, model_name="google/flan-t5-base", use_lora=True):
        """
        Initialize the radiology report simplification model.
        
        Args:
            model_name (str): Name or path of the pre-trained model to use. 
                            Defaults to "google/flan-t5-base".
            use_lora (bool): Whether to apply LoRA for efficient fine-tuning. 
                           Defaults to True.
        """
        self.model_name = model_name
        self.use_lora = use_lora
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Apply LoRA if enabled
        if use_lora:
            self._setup_lora()
        
        # Calculate parameter counts
        self.total_params = sum(p.numel() for p in self.model.parameters())
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _setup_lora(self):
        """
        Configure and apply LoRA (Low-Rank Adaptation) to the model.
        
        LoRA reduces the number of trainable parameters by adding low-rank adapters
        to specific model components, making fine-tuning more efficient while
        maintaining performance.
        """
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # Task type for sequence-to-sequence language modeling
            inference_mode=False,              # Set to False for training, True for inference
            r=16,                             # LoRA rank (dimensionality of adaptation matrices)
            lora_alpha=32,                    # LoRA scaling factor
            lora_dropout=0.1,                 # Dropout probability for LoRA layers
            target_modules=["q", "v"]         # Model components to apply LoRA to (query and value layers)
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def get_model_info(self):
        """
        Get comprehensive information about the model configuration.
        
        Returns:
            dict: Dictionary containing model metadata including:
                - model_name: Name of the base model
                - total_parameters: Total number of parameters
                - trainable_parameters: Number of trainable parameters
                - use_lora: Whether LoRA is enabled
        """
        return {
            "model_name": self.model_name,
            "total_parameters": self.total_params,
            "trainable_parameters": self.trainable_params,
            "use_lora": self.use_lora
        }
    
    def save_model(self, save_path):
        """
        Save the model and tokenizer to the specified directory.
        
        Args:
            save_path (str): Directory path where model and tokenizer will be saved
        """
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
    
    def load_model(self, load_path):
        """
        Load a previously saved model and tokenizer from disk.
        
        Args:
            load_path (str): Directory path from which to load model and tokenizer
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)

def create_model(model_name="google/flan-t5-base", use_lora=True):
    """
    Factory function to create a RadiologySimplifierModel instance.
    
    This function provides a convenient way to instantiate the radiology
    simplification model with specified configuration.
    
    Args:
        model_name (str): Name or path of the pre-trained model. 
                         Defaults to "google/flan-t5-base".
        use_lora (bool): Whether to apply LoRA fine-tuning. Defaults to True.
    
    Returns:
        RadiologySimplifierModel: Configured instance of the radiology simplification model
    """
    return RadiologySimplifierModel(model_name, use_lora)

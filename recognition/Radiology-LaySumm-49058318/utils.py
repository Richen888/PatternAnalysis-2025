import torch
import logging
import sys
import os

def setup_logging(log_level=logging.INFO):
    """
    Configure logging settings for the application.
    
    Sets up both console and file logging with timestamped messages.
    
    Args:
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG). 
                        Defaults to logging.INFO.
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Output to console
            logging.FileHandler('training.log')  # Output to file
        ]
    )

def get_gpu_info():
    """
    Retrieve GPU hardware information if available.
    
    Returns detailed information about the GPU including name, memory, 
    and CUDA version when GPU is available.
    
    Returns:
        dict: Dictionary containing GPU information with keys:
            - gpu_name: Name of the GPU device
            - vram_gb: Total VRAM in gigabytes
            - cuda_version: CUDA version string
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "gpu_name": gpu_name,
            "vram_gb": vram,
            "cuda_version": torch.version.cuda
        }
    else:
        return {"gpu_name": "CPU", "vram_gb": 0, "cuda_version": None}

def count_parameters(model):
    """
    Calculate the total and trainable parameters of a model.
    
    Args:
        model (torch.nn.Module): PyTorch model to analyze
        
    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def ensure_dir(directory):
    """
    Ensure that a directory exists, create it if necessary.
    
    Args:
        directory (str): Path to the directory to check/create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def format_time(seconds):
    """
    Convert seconds into a formatted time string (HH:MM:SS).
    
    Args:
        seconds (float): Time duration in seconds
        
    Returns:
        str: Formatted time string in HH:MM:SS format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

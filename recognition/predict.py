import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import evaluate

# ==== CONFIGURATION ====
BASE_MODEL = "google/flan-t5-base"  # Base model name
ADAPTER_PATH = "./radiology-simplifier-output"  # Path to fine-tuned adapter
MAX_INPUT = 512  # Maximum input token length
MAX_OUTPUT = 256  # Maximum output token length
BATCH_SIZE = 4  # Batch size for generation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Device selection
# =======================

def verify_test_set():
    """Verify if test set has reference summaries"""
    print("🔹 Loading BioLaySumm dataset to verify test set...")
    
    # Load the dataset
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
    
    print("Dataset structure:")
    for split_name, split_data in dataset.items():
        print(f"{split_name}: {len(split_data)} samples")
        print(f"  Columns: {split_data.column_names}")
        if 'layman_report' in split_data.column_names:
            # Check if layman_report column has actual content
            first_sample = split_data[0]
            if first_sample['layman_report'] and first_sample['layman_report'].strip():
                print(f"  ✅ 'layman_report' contains data: {first_sample['layman_report'][:100]}...")
            else:
                print(f"  ❌ 'layman_report' is empty or missing")
        else:
            print(f"  ❌ 'layman_report' column not found")
        print()

def quick_test_with_small_sample():
    """Run a quick test with small sample to confirm ROUGE works"""
    print("🔹 Running quick test with small sample...")
    
    # Load only a small sample from test set
    dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="test")
    small_sample = dataset.select(range(min(10, len(dataset))))  # Take first 10 samples
    
    # Check if references exist
    if 'layman_report' not in small_sample.column_names:
        print("❌ ERROR: 'layman_report' column not found in test set")
        return False
    
    references = small_sample['layman_report']
    if not all(ref and ref.strip() for ref in references):
        print("❌ ERROR: Some reference summaries are empty")
        return False
    
    print("✅ Test set has valid reference summaries")
    
    # Load model and generate predictions for small sample
    print("🔹 Loading model for quick test...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.to(DEVICE)
    model.eval()
    
    # Generate predictions for small sample
    predictions = []
    reports = small_sample['radiology_report']
    
    for i in range(0, len(reports), BATCH_SIZE):
        batch_reports = reports[i:i+BATCH_SIZE]
        inputs = tokenizer(
            batch_reports, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_INPUT
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_OUTPUT,
                num_beams=4,
                early_stopping=True
            )
        
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(decoded)
    
    # Try to compute ROUGE scores
    try:
        rouge = evaluate.load("rouge")
        rouge_scores = rouge.compute(
            predictions=predictions, 
            references=references[:len(predictions)], 
            use_stemmer=True
        )
        
        print("✅ SUCCESS: ROUGE scores computed successfully!")
        print(f"ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}")
        print(f"ROUGE-2: {rouge_scores.get('rouge2', 0):.4f}")
        print(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to compute ROUGE scores: {e}")
        return False

# Run verification
if __name__ == "__main__":
    verify_test_set()
    print("\n" + "="*50)
    success = quick_test_with_small_sample()
    
    if success:
        print("\n🎉 Test set is ready for full evaluation!")
        print("You can now run the full evaluation on the entire test set.")
    else:
        print("\n⚠️ WARNING: Test set may not have reference summaries")
        print("Consider using validation set instead.")
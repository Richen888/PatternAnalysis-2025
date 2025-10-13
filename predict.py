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
ADAPTER_PATH = "./radiology-simplifier-output"  # Path to fine-tuned adapter weights
OUTPUT_FILE = "biolaysumm_validation_results.csv"  # Output CSV file name
MAX_INPUT_LENGTH = 512  # Maximum token length for input reports
MAX_OUTPUT_LENGTH = 256  # Maximum token length for generated summaries
BATCH_SIZE = 4  # Batch size for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available
NUM_EXAMPLES = 5  # Number of representative examples to display
# =======================

# Load the BioLaySumm validation dataset
print("🔹 Loading BioLaySumm validation set...")
dataset = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track", split="validation")

# Load tokenizer and base model with PEFT adapter
print("🔹 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(DEVICE)
model.eval()  # Set model to evaluation mode

# Generate predictions on the validation set
print("🔹 Generating predictions on validation set...")
predictions, references, reports = [], [], []

for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
    # Extract batch of radiology reports and corresponding layman summaries
    batch_reports = dataset[i:i+BATCH_SIZE]["radiology_report"]
    batch_refs = dataset[i:i+BATCH_SIZE]["layman_report"]
    reports.extend(batch_reports)
    references.extend(batch_refs)

    # Tokenize input reports with padding and truncation
    inputs = tokenizer(
        batch_reports, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=MAX_INPUT_LENGTH
    ).to(DEVICE)

    # Generate summaries without gradient computation for efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_OUTPUT_LENGTH,
            num_beams=4,  # Use beam search for better quality
            early_stopping=True  # Stop generation when all beams have finished
        )

    # Decode generated token IDs back to text
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    predictions.extend(decoded)

# ==== Compute ROUGE scores ====
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

# Print ROUGE scores in official 0-1 scale format
print("\n✅ ROUGE Scores on Validation Set:")
print(f"ROUGE-1: {rouge_scores.get('rouge1', 0):.4f}")
print(f"ROUGE-2: {rouge_scores.get('rouge2', 0):.4f}")
print(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.4f}")
print(f"ROUGE-Lsum: {rouge_scores.get('rougeLsum', 0):.4f}")

# ==== Save results to CSV ====
print("\n💾 Saving validation results CSV...")
df = pd.DataFrame({
    "id": list(range(len(reports))),
    "radiology_report": reports,
    "reference_lay_summary": references,
    "generated_summary": predictions
})
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ Validation results saved to {OUTPUT_FILE}")

# ==== Print representative examples ====
print("\n" + "="*80)
print(f"Representative Examples (First {NUM_EXAMPLES})")
print("="*80)

for i in range(min(NUM_EXAMPLES, len(reports))):
    print(f"\nExample {i+1}:")
    print(f"Radiology Report: {reports[i][:300]}{'...' if len(reports[i])>300 else ''}")
    print(f"Reference Lay Summary: {references[i]}")
    print(f"Generated Summary: {predictions[i]}")
    print("-"*80)

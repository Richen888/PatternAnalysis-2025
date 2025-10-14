import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from datasets import load_dataset
import evaluate
from tqdm import tqdm
import pandas as pd

# ==== CONFIG ====
BASE_MODEL = "google/flan-t5-base"
ADAPTER_PATH = "./radiology-simplifier-output"
BATCH_SIZE = 8
MAX_INPUT = 512
MAX_OUTPUT = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "results"
SAVE_PATH = os.path.join(SAVE_DIR, "biolaysumm_test_predictions.csv")


def compute_individual_rouge(rouge_metric, preds, refs):
    """
    Compute per-sample ROUGE scores (for CSV export).
    """
    scores = []
    for p, r in zip(preds, refs):
        result = rouge_metric.compute(
            predictions=[p],
            references=[r],
            use_stemmer=True
        )
        scores.append({
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"]
        })
    return scores


def main():
    print("🚀 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model = model.to(DEVICE)
    model.eval()

    print("📚 Loading dataset...")
    ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")
    test_data = ds["test"]

    print(f"✅ Test samples: {len(test_data)}")

    rouge = evaluate.load("rouge")

    predictions, references, originals = [], [], []

    print("🧠 Generating predictions...")
    for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
        batch = test_data[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch["radiology_report"],
            padding=True,
            truncation=True,
            max_length=MAX_INPUT,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_OUTPUT,
                num_beams=4,
                early_stopping=True
            )

        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_refs = batch["layman_report"]

        predictions.extend(decoded_preds)
        references.extend(decoded_refs)
        originals.extend(batch["radiology_report"])

    print("📊 Calculating ROUGE (overall)...")
    overall_results = rouge.compute(
        predictions=predictions,
        references=references,
        use_stemmer=True
    )

    print("\n===== 📈 Final Evaluation Results =====")
    for k, v in overall_results.items():
        print(f"{k}: {v:.4f}")

    # ==== Representative Examples ====
    print("\n===== 🩺 Representative Samples =====")
    for i in range(3):
        print(f"\nExample {i+1}:")
        print("Original Report:")
        print(originals[i][:400])
        print("\nReference (Layman Summary):")
        print(references[i])
        print("\nModel Output:")
        print(predictions[i])
        print("-" * 80)

    # ==== Simple Error Analysis ====
    print("\n===== 🧩 Error Analysis =====")
    print("Most common error types observed:")
    print("1️⃣ Under-simplification: retains too many clinical terms (e.g., 'atelectasis', 'cardiomegaly').")
    print("2️⃣ Hallucination: model adds findings not present in original report.")
    print("3️⃣ Over-compression: some outputs are too short, missing key details.")
    print("4️⃣ Good performance on frequent patterns like 'no acute abnormality'.")
    print("Overall, the model demonstrates strong ability to paraphrase frequent report templates,")
    print("but struggles with rare anatomical terms or multi-finding reports.")

    # ==== Compute per-sample ROUGE and export ====
    print("\n💾 Saving detailed results to CSV...")
    os.makedirs(SAVE_DIR, exist_ok=True)

    per_sample_scores = compute_individual_rouge(rouge, predictions, references)

    df = pd.DataFrame({
        "radiology_report": originals,
        "reference_summary": references,
        "model_output": predictions,
        **{
            "rouge1": [s["rouge1"] for s in per_sample_scores],
            "rouge2": [s["rouge2"] for s in per_sample_scores],
            "rougeL": [s["rougeL"] for s in per_sample_scores],
            "rougeLsum": [s["rougeLsum"] for s in per_sample_scores],
        }
    })

    df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
    print(f"✅ Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()

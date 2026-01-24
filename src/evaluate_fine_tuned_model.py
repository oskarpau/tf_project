import os
import time
import csv
import torch
import pandas as pd
from datetime import datetime, timedelta

# Unsloth and Transformers libraries
from unsloth import FastVisionModel
from transformers import AutoProcessor

# Your existing modules
from model import Model
from config import COLUMN_SCHEMA
import model_fine_tuning  # To load cleaned datasets

# --- CONFIGURATION ---
RESULTS_PATH = "finetuned_results_50epochs_holedataset.csv"
CSV_SEP = ";"
FINETUNED_MODEL_DIR = "../hpc/qwen3_vl_lora_finetuned"  # Output from your training script
BASE_MODEL_NAME = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit" # Base model

class UnslothEvalModel(Model):
    """
    Inherits from your original 'Model' class to reuse:
    - run_batch_and_compute_confidence
    - preprocess_prompt
    - calculate_conf_and_filter_correct
    But overrides __init__ to load with Unsloth.
    """
    def __init__(self):
        # Skip parent class __init__ for custom loading
        self.model = None
        self.processor = None
        
        print(f"[Init] Loading Unsloth LoRA model from: {FINETUNED_MODEL_DIR}")
        start_time = time.perf_counter()

        # 1. MODEL LOADING (Unsloth)
        # It's crucial to use FastVisionModel because that's how it was trained, even if you only use text.
        model, _ = FastVisionModel.from_pretrained(
            model_name=FINETUNED_MODEL_DIR, # Automatically loads adapter + base
            load_in_4bit=True,
            max_seq_length=8192,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Native Unsloth inference optimization
        FastVisionModel.for_inference(model)
        self.model = model
        print("Model loaded successfully")

        # 2. PROCESSOR LOADING (Transformers)
        # 'confidence_methods.py' needs access to self.processor.tokenizer.decode.
        # Unsloth returns a "bare" tokenizer, so we instantiate the full AutoProcessor 
        # to maintain compatibility with your legacy code.
        print("[Init] Loading AutoProcessor for tokenization compatibility...")
        try:
            self.processor = AutoProcessor.from_pretrained(FINETUNED_MODEL_DIR)
        except:
            self.processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)

        init_time = time.perf_counter() - start_time
        print(f"[Init] Model loaded in {init_time:.2f} seconds")


def _get_last_written_index(path: str, sep: str = CSV_SEP) -> int:
    """Helper to resume evaluation if interrupted."""
    # Check if file exists and is not empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return -1
    last_row = None
    # Read CSV file to find last written row
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        try:
            next(reader)  # Skip header
        except StopIteration:
            return -1
        # Iterate to get last row
        for row in reader:
            if row:
                last_row = row
    # Extract index from last row
    if not last_row:
        return -1
    try:
        return int(str(last_row[0]).strip())
    except:
        return -1

def run_evaluation():
    
    # Reset results file if it exists so we always start fresh
    if os.path.exists(RESULTS_PATH):
        print(f"Resetting results file: {RESULTS_PATH}")
        os.remove(RESULTS_PATH)

    # 1. Load Datasets (using clean logic from model_fine_tuning)
    print("Loading and processing datasets...")
    all_datasets = model_fine_tuning.load_processed_datasets()

    '''# Optional: limit rows per dataset for faster test runs #######################FORTESTING
    TEST_ROWS = 6  # Set to None to disable
    print(f"Test mode: limiting to head({TEST_ROWS}) rows per dataset")
    all_datasets = {name: df.head(TEST_ROWS) for name, df in all_datasets.items()}
    '''
    ''' # Optional: Subsample datasets for quicker evaluation
    print("Subsampling: Keeping random 50% of each dataset...")
    for name, df in all_datasets.items():
        original_count = len(df)
        # frac=0.5 selecciona el 50%. 
        # random_state=42 asegura que siempre elija las mismas filas si reinicias el script.
        sampled_df = df.sample(frac=0.5, random_state=42)
        all_datasets[name] = sampled_df
        print(f"  -> {name}: Reduced from {original_count} to {len(sampled_df)} rows")
    '''

    # 2. Add 'answer_type' column 
    # (Needed for model.py to know which prompt to use: True/False vs Multi)
    print("Adding response type metadata...")
    for name, df in all_datasets.items():
        if 'answer' in df.columns:
            def determine_type(ans_list):
                # If all possible answers are variants of true/false
                if all(str(x).lower() in ['true', 'false'] for x in ans_list):
                    return 'true/false'
                return 'multi_str'
            
            df['answer_type'] = df['answer'].apply(determine_type)

    # 3. Initialize Model
    model = UnslothEvalModel()

    # 4. Prepare CSV
    last_written_index = _get_last_written_index(RESULTS_PATH, sep=CSV_SEP)
    if last_written_index >= 0:
        print(f"Resuming from index {last_written_index + 1}")
    else:
        print(f"Starting new file: {RESULTS_PATH}")

    # Initialize counters and timing
    total_rows = sum(len(df) for df in all_datasets.values())
    global_row_idx = 0
    processed_count = 0
    start_perf = time.perf_counter()
    
    is_first_write = (not os.path.exists(RESULTS_PATH)) or (os.path.getsize(RESULTS_PATH) == 0)

    # 5. Evaluation Loop
    for dataset_name, df in all_datasets.items():
        print(f"\n---> Dataset: {dataset_name} ({len(df)} rows)")
        
        for index, row in df.iterrows():
            # Skip already processed rows
            if global_row_idx <= last_written_index:
                global_row_idx += 1
                continue

            q_start = time.perf_counter()
            
            # --- CALL TO YOUR CODE CORE ---
            # model.py handles tokenization (text only), generation, and confidence calculation
            try:
                dataframe_result = model.run_batch_and_compute_confidence(
                    dataset_name=dataset_name, 
                    categories=[row['categorie']], 
                    subcategories=[row['subcategorie']],
                    questions=[row['question']],
                    right_answers=[row['answer']],
                    question_types=[row['answer_type']]
                )
                
                # Save result to CSV
                dataframe_result.index = [global_row_idx]
                write_mode = "w" if is_first_write else "a"
                dataframe_result.to_csv(
                    RESULTS_PATH, mode=write_mode, header=is_first_write,
                    index=True, index_label="index", sep=CSV_SEP
                )
                is_first_write = False
                
            except Exception as e:
                print(f"Error in row {global_row_idx}: {e}")
                # Optional: save empty row or continue

            # Timing metrics
            processed_count += 1
            global_row_idx += 1
            
            q_time = time.perf_counter() - q_start
            total_elapsed = time.perf_counter() - start_perf
            avg_time = total_elapsed / processed_count
            remaining_rows = total_rows - (last_written_index + 1 + processed_count)
            eta = timedelta(seconds=int(avg_time * remaining_rows))

            # Print progress
            print(
                f"[{global_row_idx}/{total_rows}] "
                f"T: {q_time:.2f}s | ETA: {eta} | "
                f"Dataset: {dataset_name}"
            )

    print(f"\nEvaluation completed. Results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
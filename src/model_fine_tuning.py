"""
Fine-tuning script for Qwen3-VL using Unsloth + LoRA.

This script is intentionally SEPARATE from model.py.
- model.py        -> inference + confidence evaluation
- model_fine_tuning.py -> supervised fine-tuning (SFT)

This separation is important for clean experimental methodology.

Update: Adds dataset loading mirroring `run_model_on_dataset.py` and a CLI
entrypoint to launch training from the selected dataset(s).
"""
from unsloth.trainer import UnslothVisionDataCollator
import os
import re
import argparse
import pandas as pd
import torch
from datasets import Dataset
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig

############################################
# 1. MODEL CONFIGURATION
############################################

MODEL_NAME = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
OUTPUT_DIR = "qwen3_vl_lora_finetuned"

MAX_SEQ_LENGTH = 8192
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True

############################################
# 2. LOAD MODEL + TOKENIZER (UNSLOTH)
############################################

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=False,      # no images in your datasets
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=64,
    lora_alpha=128,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
    use_gradient_checkpointing="unsloth",
)

FastVisionModel.for_training(model)

############################################
# 3. DATA LOADING (mirrors run_model_on_dataset.py)
############################################

# Base path relative to this file, matching run_model_on_dataset.py
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASETS_BASE = os.path.join(_SCRIPT_DIR, os.pardir, "datasets")


def _load_all_raw_datasets() -> dict:
    """
    Load raw datasets into pandas DataFrames, mirroring logic in
    eneko_trying_out_things/run_model_on_dataset.py.

    Returns a dict[str, pd.DataFrame].
    """
    import pandas as pd  # Local import to avoid hard dependency when importing module.

    # 1. strategyqa_dataset
    strategyqa_dev_df = pd.read_json(os.path.join(_DATASETS_BASE, "strategyqa_dataset", "dev.json"))
    strategyqa_train_df = pd.read_json(os.path.join(_DATASETS_BASE, "strategyqa_dataset", "train.json"))
    strategyqa_train_df["categorie"], strategyqa_dev_df["categorie"] = "NA", "NA"
    strategyqa_train_df["subcategorie"], strategyqa_dev_df["subcategorie"] = "NA", "NA"

    # 2. gsm8k_datasets
    gsm8k_test_df = pd.read_parquet(os.path.join(_DATASETS_BASE, "gsm8k_datasets", "test-00000-of-00001.parquet"))
    gsm8k_train_df = pd.read_parquet(os.path.join(_DATASETS_BASE, "gsm8k_datasets", "train-00000-of-00001.parquet"))
    gsm8k_test_df["categorie"], gsm8k_train_df["categorie"] = "math", "math"
    gsm8k_test_df["subcategorie"], gsm8k_train_df["subcategorie"] = "NA", "NA"

    # 3. maqa_datasets
    maqa_commonsense_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "MAQA_commonsense_reasoning.json"))
    maqa_commonsense_df["categorie"], maqa_commonsense_df["subcategorie"] = "commonsense single", "NA"

    maqa_math_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "MAQA_mathmatical_reasoning.json"))
    maqa_math_df["categorie"], maqa_math_df["subcategorie"] = "math", "multi"

    maqa_world_hls_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "MAQA_world_knowledge_HLS.json"))
    maqa_world_hls_df["categorie"], maqa_world_hls_df["subcategorie"] = "world knowledge", "multi HLS"

    maqa_world_nq_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "MAQA_world_knowledge_nq.json"))
    maqa_world_nq_df["categorie"], maqa_world_nq_df["subcategorie"] = "world knowledge", "multi NQ"

    maqa_single_commonsense_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "single_commonsens_reasoning(StrategyQA).json"))
    maqa_single_commonsense_df["categorie"], maqa_single_commonsense_df["subcategorie"] = "commonsense single", "NA"

    maqa_single_math_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "single_mathematical_reasoning(gsm8k).json"))
    maqa_single_math_df["categorie"], maqa_single_math_df["subcategorie"] = "math", "single"

    maqa_single_world_nq_df = pd.read_json(os.path.join(_DATASETS_BASE, "maqa_datasets", "single_world_knowledge(NQ).json"))
    maqa_single_world_nq_df["categorie"], maqa_single_world_nq_df["subcategorie"] = "world knowledge", "single NQ"

    return {
        "strategyqa_dev": strategyqa_dev_df,
        "strategyqa_train": strategyqa_train_df,
        "gsm8k_test": gsm8k_test_df,
        "gsm8k_train": gsm8k_train_df,
        "maqa_commonsense_reasoning": maqa_commonsense_df,
        "maqa_mathematical_reasoning": maqa_math_df,
        "maqa_world_knowledge_hls": maqa_world_hls_df,
        "maqa_world_knowledge_nq": maqa_world_nq_df,
        "maqa_single_commonsense_reasoning": maqa_single_commonsense_df,
        "maqa_single_math_reasoning": maqa_single_math_df,
        "maqa_single_world_knowledge_nq": maqa_single_world_nq_df,
    }


def _get_processed_answer_list(answer_input):
    if isinstance(answer_input, list):
        return [str(item) for item in answer_input]
    elif isinstance(answer_input, (bool, int, float)):
        return [str(answer_input)]
    elif isinstance(answer_input, str):
        answer_string = answer_input
        expression_result_matches = re.findall(r"<<.*?=(-?\d+(?:\.\d+)?)>>", answer_string)
        if expression_result_matches:
            return [expression_result_matches[-1]]
        final_answer_match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer_string)
        if final_answer_match:
            return [final_answer_match.group(1)]
        if answer_string.strip().lower() == "true":
            return ["true"]
        if answer_string.strip().lower() == "false":
            return ["false"]
        return [answer_string]
    else:
        return [str(answer_input)]


def _split_multi_part_questions(row: pd.Series):
    question_text = row["question"]
    original_answers = row["answer"]
    parts = re.findall(r"\((\w)\)\s*(.*?)(?=\(\w\)|$)", question_text, re.DOTALL)
    if not parts:
        return [{**row.to_dict(), "question": question_text, "answer": original_answers}]
    new_rows = []
    for letter, q_text in parts:
        is_true = str(letter) in original_answers
        new_row = row.to_dict().copy()
        new_row["question"] = q_text.strip()
        new_row["answer"] = ["true"] if is_true else ["false"]
        for col in ["qid", "term", "description", "facts", "decomposition", "evidence"]:
            if col in new_row:
                del new_row[col]
        new_rows.append(new_row)
    return new_rows


def _round_answers_to_integers(answer_list):
    processed_answers = []
    for item in answer_list:
        try:
            processed_answers.append(str(int(float(item))))
        except (ValueError, TypeError):
            processed_answers.append(item)
    return processed_answers


def load_processed_datasets() -> dict:
    """Load and process datasets like run_model_on_dataset.py returns.

    Returns dict[name, DataFrame] with columns at least: question, answer (list[str]).
    """
    all_datasets = _load_all_raw_datasets()

    # Apply initial processing of 'answer' to list[str]
    for _, df in all_datasets.items():
        if "answer" in df.columns:
            df["answer"] = df["answer"].apply(_get_processed_answer_list)

    # Expand MAQA commonsense multi-part into single true/false items
    if "maqa_commonsense_reasoning" in all_datasets:
        maqa_df = all_datasets["maqa_commonsense_reasoning"]
        expanded_rows = []
        for _, row in maqa_df.iterrows():
            expanded_rows.extend(_split_multi_part_questions(row))
        expanded_maqa = pd.DataFrame(expanded_rows)
        expanded_maqa["categorie"], expanded_maqa["subcategorie"] = "commonsense single", "NA"
        all_datasets["maqa_commonsense_reasoning"] = expanded_maqa

    # Round numeric answers like 18.0 -> 18, keep non-numeric as-is
    for _, df in all_datasets.items():
        if "answer" in df.columns:
            df["answer"] = df["answer"].apply(_round_answers_to_integers)

    # Ensure final format of answers is list[str]
    for _, df in all_datasets.items():
        if "answer" in df.columns:
            df["answer"] = df["answer"].apply(_get_processed_answer_list)

    return all_datasets


def get_training_dataframe(dataset_keys: list[str] | None = None) -> pd.DataFrame:
    """
    Build a single training pandas DataFrame from selected dataset keys.
    - dataset_keys: iterable of keys from load_processed_datasets(); if None or
      empty, defaults to ["gsm8k_train"].
    Output columns:
      - question (str)
      - right_answer (list[str])
    """
    all_ds = load_processed_datasets()
    if not dataset_keys:
        dataset_keys = sorted(all_ds.keys())

    missing = [k for k in dataset_keys if k not in all_ds]
    if missing:
        raise ValueError(f"Unknown dataset keys: {missing}. Available: {sorted(all_ds.keys())}")

    frames = []
    for k in dataset_keys:
        df = all_ds[k]
        cols_needed = []
        if "question" not in df.columns:
            cols_needed.append("question")
        if "answer" not in df.columns:
            cols_needed.append("answer")
        if cols_needed:
            raise ValueError(f"Dataset '{k}' missing required columns: {cols_needed}")
        sub = df[["question", "answer"]].copy()
        sub.rename(columns={"answer": "right_answer"}, inplace=True)
        frames.append(sub)

    if not frames:
        raise ValueError("No data selected for training.")

    combined = pd.concat(frames, ignore_index=True)
    return combined

############################################
# 4. DATASET CONVERSION TO CONVERSATION
############################################

INSTRUCTION = (
    "Instruction: Answer the following question as briefly as possible. "
    "Return ONLY the final answer in the required format."
)


def convert_row_to_conversation(row: dict) -> dict:
    """
    Converts one QA sample to Unsloth conversation format.

    Expected row format:
    {
        'question': str,
        'right_answer': list[str]
    }
    """

    question = row["question"].strip()
    answers = row["right_answer"]

    # Ensure list[str]
    if isinstance(answers, str):
        answers = [answers]

    answer_text = "||" + ", ".join(answers) + "||"

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{INSTRUCTION}\nQuestion: {question}"}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": answer_text}
                ],
            },
        ]
    }


############################################
# 5. BUILD HF DATASET
############################################

def build_training_dataset(df) -> Dataset:
    """
    df is expected to be a pandas DataFrame with at least:
    - question
    - right_answer
    """

    conversations = []
    for _, row in df.iterrows():
        conversations.append(convert_row_to_conversation(row))

    return Dataset.from_list(conversations)


############################################
# 6. TRAIN FUNCTION WITH DYNAMIC ARGS
############################################

def run_fine_tuning(df_train):
    """
    df_train: pandas DataFrame with QA pairs
    """
    
    # Parámetros de batch
    PER_DEVICE_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 4
    
    train_dataset = build_training_dataset(df_train)
    
    # --- CÁLCULO PARA GUARDAR CADA 5 EPOCHS ---
    # Calculamos cuántos pasos (steps) hay en una epoch
    num_samples = len(train_dataset)
    effective_batch_size = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION
    
    # Steps por epoch = Total datos / (Batch por dispositivo * Acumulación)
    steps_per_epoch = num_samples // effective_batch_size
    
    # Aseguramos que sea al menos 1 paso si el dataset es muy pequeño
    if steps_per_epoch < 1:
        steps_per_epoch = 1
        
    # Queremos guardar cada 5 epochs
    save_steps_interval = steps_per_epoch * 5
    
    print(f"Dataset size: {num_samples}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Saving checkpoint every {save_steps_interval} steps (approx every 5 epochs).")
    
    # Definimos los argumentos aquí para usar la variable 'save_steps_interval'
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=50.0,
        logging_steps=10,
        
        # --- CAMBIOS REALIZADOS ---
        save_strategy="steps",        # Estrategia por pasos para poder calcular "cada 5 epochs"
        save_steps=save_steps_interval,
        save_total_limit=None,        # None = Guardar TODOS los checkpoints (no borrar antiguos)
        # --------------------------
        
        seed=3407,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=training_args,
        train_dataset=train_dataset,
    )

    train_result = trainer.train()

    # Save LoRA adapters final
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return train_result


############################################
# 7. CLI ENTRY POINT
############################################

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL with LoRA using Unsloth.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help=(
            "One or more dataset keys to train on. If omitted, trains on ALL available datasets "
            "discovered under datasets/ (mirrors run_model_on_dataset.py)."
        ),
    )
    args = parser.parse_args()

    print(f"Loading datasets: {args.datasets or 'ALL (default)'}")
    df_train = get_training_dataframe(args.datasets)
    print(f"Training samples: {len(df_train)}")

    result = run_fine_tuning(df_train)
    print("Training completed.")
    print(result)


if __name__ == "__main__":
    main()
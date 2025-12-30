import os
import subprocess
import sys
#from google.colab import drive
import pandas as pd
import time
from datetime import datetime, timedelta
import csv
import re

from model import Model

# SRC/RUN_DATASETS
RESULTS_PATH = "initial_results.csv"
BATCH_SIZE = 1
CSV_SEP = ";"

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Print it
print("Directory of this script:", script_dir)

'''
dataset_name
categories
subcategories
question
rightanswers list
question type = "true/false", "multi string" list
'''

base_path = os.path.join(script_dir, os.pardir, 'datasets') # os.pardir equals '..' (parent path).

# 1. For strategyqa_dataset:
# No categorie specified
# categorie: NA, subcategorie: NA
strategyqa_dev_df = pd.read_json(os.path.join(base_path, 'strategyqa_dataset', 'dev.json'))
strategyqa_train_df = pd.read_json(os.path.join(base_path, 'strategyqa_dataset', 'train.json'))
strategyqa_train_df['categorie'], strategyqa_dev_df['categorie'] = 'NA', 'NA'
strategyqa_train_df['subcategorie'], strategyqa_dev_df['subcategorie'] = 'NA', 'NA'

# 2. For gsm8k_datasets:
# categorie: math, subcategorie: NA
gsm8k_test_df = pd.read_parquet(os.path.join(base_path, 'gsm8k_datasets', 'test-00000-of-00001.parquet'))
gsm8k_train_df = pd.read_parquet(os.path.join(base_path, 'gsm8k_datasets', 'train-00000-of-00001.parquet'))
gsm8k_test_df['categorie'], gsm8k_train_df['categorie'] = 'math', 'math'
gsm8k_test_df['subcategorie'], gsm8k_train_df['subcategorie'] = 'NA', 'NA'

# 3. For maqa_datasets:
# categorie: math, subcategorie: single
# categorie: math, subcategorie: multi
# categorie: world knowledge, subcategorie: multi HLS
# categorie: world knowledge, subcategorie: multi NQ
# categorie: world knowledge, subcategorie: single NQ
# categorie: commonsense single, subcategorie: NA
maqa_commonsense_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'MAQA_commonsense_reasoning.json'))
# The questions are actually single answer despite MAQA prefix
maqa_commonsense_df['categorie'], maqa_commonsense_df['subcategorie'] = 'commonsense single', 'NA'

maqa_math_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'MAQA_mathmatical_reasoning.json'))
maqa_math_df['categorie'], maqa_math_df['subcategorie'] = 'math', 'multi'

maqa_world_hls_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'MAQA_world_knowledge_HLS.json'))
maqa_world_hls_df['categorie'], maqa_world_hls_df['subcategorie'] = 'world knowledge', 'multi HLS'

maqa_world_nq_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'MAQA_world_knowledge_nq.json'))
maqa_world_nq_df['categorie'], maqa_world_nq_df['subcategorie'] = 'world knowledge', 'multi NQ'

maqa_single_commonsense_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'single_commonsens_reasoning(StrategyQA).json'))
maqa_single_commonsense_df['categorie'], maqa_single_commonsense_df['subcategorie'] = 'commonsense single', 'NA'

maqa_single_math_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'single_mathematical_reasoning(gsm8k).json'))
maqa_single_math_df['categorie'], maqa_single_math_df['subcategorie'] = 'math', 'single'

maqa_single_world_nq_df = pd.read_json(os.path.join(base_path, 'maqa_datasets', 'single_world_knowledge(NQ).json'))
maqa_single_world_nq_df['categorie'], maqa_single_world_nq_df['subcategorie'] = 'world knowledge', 'single NQ'

all_datasets = {
    'strategyqa_dev': strategyqa_dev_df,
    'strategyqa_train': strategyqa_train_df,
    'gsm8k_test': gsm8k_test_df,
    'gsm8k_train': gsm8k_train_df,
    'maqa_commonsense_reasoning': maqa_commonsense_df,
    'maqa_mathematical_reasoning': maqa_math_df,
    'maqa_world_knowledge_hls': maqa_world_hls_df,
    'maqa_world_knowledge_nq': maqa_world_nq_df,
    'maqa_single_commonsense_reasoning': maqa_single_commonsense_df,
    'maqa_single_math_reasoning': maqa_single_math_df,
    'maqa_single_world_knowledge_nq': maqa_single_world_nq_df

}

# maqa_dfs = {}
# maqa_dataset_path = os.path.join(base_path, 'maqa_datasets')
# for filename in os.listdir(maqa_dataset_path):
#     if filename.endswith('.json'):
#         file_path = os.path.join(maqa_dataset_path, filename)
#         df_name = os.path.splitext(filename)[0] # Get filename without extension
#         maqa_dfs[df_name] = pd.read_json(file_path)

# # Add MAQA datasets to the main dictionary
# for name, df in maqa_dfs.items():
#     all_datasets[f'maqa_{name}'] = df

# Set display option to show full column content
pd.set_option('display.max_colwidth', None)

'''
print("All loaded DataFrames are now organized into the 'all_datasets' dictionary.")
print(f"Total DataFrames in dictionary: {len(all_datasets)}")
print("Keys in all_datasets:")
for key in all_datasets.keys():
    print(f"- {key}")
'''


#DATA CLEANING AND TRANSFORMATION STARTS HERE
#####################

# DATA CLEANING TRANSFORMATION
# 1. SOLVE << ... >>>

def get_processed_answer_list(answer_input):

    if isinstance(answer_input, list):
        # If it's already a list, ensure all items are strings
        return [str(item) for item in answer_input]
    elif isinstance(answer_input, (bool, int, float)):
        # Convert booleans, ints, floats to string and wrap in a list
        return [str(answer_input)]
    elif isinstance(answer_input, str):
        answer_string = answer_input

        # 1. Try to find the result from the last '<<expression=result>>' (GSM8K intermediate/final)
        # This regex looks for <<...=NUMBER>> and captures the number
        expression_result_matches = re.findall(r'<<.*?=(-?\d+(?:\.\d+)?)>>', answer_string)
        if expression_result_matches:
            # Return the last captured numerical result from <<...>>
            return [expression_result_matches[-1]]

        # 2. If no '<<...>>' result found, try to find the final answer after '####' (GSM8K specific)
        # Captures integers and floats
        final_answer_match = re.search(r'####\s*(-?\d+(?:\.\d+)?)', answer_string)
        if final_answer_match:
            return [final_answer_match.group(1)]

        # 3. If neither numerical pattern is found, check for 'true' or 'false' (StrategyQA style)
        if answer_string.strip().lower() == 'true':
            return ['true']
        if answer_string.strip().lower() == 'false':
            return ['false']

        # 4. If none of the above specific patterns, return the original string wrapped in a list
        return [answer_string]
    else:
        # Fallback for unexpected types
        return [str(answer_input)]

# Apply the processing function to the 'answer' column of all dataframes in all_datasets
#print("Processing 'answer' column for all datasets...")
for dataset_name, df in all_datasets.items():
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(get_processed_answer_list)
        #print(f"  '{dataset_name}' 'answer' column processed.")


# data CLEANINS
# 2. SPLIT (A) (B) (C)

def split_multi_part_questions(row):
    question_text = row['question']
    original_answers = row['answer'] # This is expected to be a list like ['a', 'b']
    parts = re.findall(r'\((\w)\)\s*(.*?)(?=\(\w\)|$)', question_text, re.DOTALL)

    if not parts:
        return [{**row.to_dict(), 'question': question_text, 'answer': original_answers}]

    new_rows = []
    for letter, q_text in parts:
        is_true = str(letter) in original_answers
        new_row = row.to_dict().copy()
        new_row['question'] = q_text.strip() # Remove leading/trailing whitespace
        new_row['answer'] = ['true'] if is_true else ['false']
        # Remove other columns not relevant for the new simplified question
        if 'qid' in new_row: del new_row['qid']
        if 'term' in new_row: del new_row['term']
        if 'description' in new_row: del new_row['description']
        if 'facts' in new_row: del new_row['facts']
        if 'decomposition' in new_row: del new_row['decomposition']
        if 'evidence' in new_row: del new_row['evidence']

        new_rows.append(new_row)

    return new_rows

#print("The function `split_multi_part_questions` has been defined.")


# data cleaning and processing
###################
# 2 CONTINUES

# Get the maqa_MAQA_commonsense_reasoning DataFrame
maqa_commonsense_df = all_datasets['maqa_commonsense_reasoning']

# Apply the function to each row and collect the results
expanded_rows = []
for index, row in maqa_commonsense_df.iterrows():
    expanded_rows.extend(split_multi_part_questions(row))

# Create a new DataFrame from the expanded rows
expanded_maqa_commonsense_df = pd.DataFrame(expanded_rows)
expanded_maqa_commonsense_df['categorie'], expanded_maqa_commonsense_df['subcategorie'] = 'commonsense single', 'NA' 

# Replace the original DataFrame in all_datasets with the expanded one
all_datasets['maqa_commonsense_reasoning'] = expanded_maqa_commonsense_df


#data processing
# 3
# 18.0 -> 18 ROUND

def round_answers_to_integers(answer_list):
    """
    Rounds numerical strings in a list of answers to integer strings.
    Non-numerical strings are left unchanged.
    """
    processed_answers = []
    for item in answer_list:
        try:
            # Try to convert to float first, then to int, then back to string
            # This handles cases like '18.0' or '70000.0'
            processed_answers.append(str(int(float(item))))
        except (ValueError, TypeError):
            # If it's not a number, or cannot be converted, keep it as is
            processed_answers.append(item)
    return processed_answers


#print("Rounding numerical answers to integers for all datasets...")
for dataset_name, df in all_datasets.items():
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(round_answers_to_integers)
        print(f"  '{dataset_name}' 'answer' column rounded.")



#cleaning
#4 all in format ['string']


#print("Re-processing 'answer' column to ensure ['string'] format for all datasets...")
for dataset_name, df in all_datasets.items():
    if 'answer' in df.columns:
        df['answer'] = df['answer'].apply(get_processed_answer_list)
        #print(f"  '{dataset_name}' 'answer' column re-processed to ['string'] format.")


print("\nVerification of 'answer' format for a few samples:")
# Sample and verify the format for a few dataframes

if 'gsm8k_test' in all_datasets:
    #print("\n--- gsm8k_test sample ---")
    sample_df = all_datasets['gsm8k_test'].head(2)
    for index, row in sample_df.iterrows():
        answer_value = row['answer']
        #print(f"  Question: {row['question'][:50]}...")
        #print(f"  Answer: {answer_value}, Type: {type(answer_value)}, Item Type: {type(answer_value[0]) if answer_value else 'N/A'}")

if 'maqa_MAQA_commonsense_reasoning' in all_datasets:
    #print("\n--- maqa_MAQA_commonsense_reasoning sample ---")
    sample_df = all_datasets['maqa_MAQA_commonsense_reasoning'].head(10)
    for index, row in sample_df.iterrows():
        answer_value = row['answer']
        print(f"  Question: {row['question']}")
        print(f"  Answer: {answer_value}, Type: {type(answer_value)}, Item Type: {type(answer_value[0]) if answer_value else 'N/A'}")

if 'strategyqa_dev' in all_datasets:
    #print("\n--- strategyqa_dev sample ---")
    sample_df = all_datasets['strategyqa_dev'].head(2)
    for index, row in sample_df.iterrows():
        answer_value = row['answer']
        #print(f"  Question: {row['question'][:50]}...")
        #print(f"  Answer: {answer_value}, Type: {type(answer_value)}, Item Type: {type(answer_value[0]) if answer_value else 'N/A'}")

# Function to determine answer type
def get_answer_type(answer_list):
    if all(isinstance(item, str) and item.lower() in ['true', 'false'] for item in answer_list):
        return 'true/false'
    return 'multi_str'

# Add 'answer_type' column to all dataframes in all_datasets
print("Adding 'answer_type' column to all datasets...")
for dataset_name, df in all_datasets.items():
    if 'answer' in df.columns:
        df['answer_type'] = df['answer'].apply(get_answer_type)
        print(f"  '{dataset_name}' 'answer_type' column added.")

print(df.head(2))

#################################
# END OF CLEANING
################################

def _get_last_written_index(path: str, sep: str = CSV_SEP) -> int:
    """Return the last written integer index from the results CSV.

    Assumes the first column is the index and the first row is a header.
    Returns -1 when the file does not exist or has no data rows.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return -1
    last_row = None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=sep)
        try:
            next(reader)  # header
        except StopIteration:
            return -1

        for row in reader:
            if row:
                last_row = row

    if not last_row:
        return -1
    try:
        return int(str(last_row[0]).strip())
    except (ValueError, TypeError, IndexError):
        raise ValueError(
            f"Could not parse last index from {path}. Expected integer in first column, got: {last_row[:1]!r}"
        )

def run_model_on_dataset():
    """
    This function loops over all datasets, their categories and subcategories
    """

    last_written_index = _get_last_written_index(RESULTS_PATH, sep=CSV_SEP)
    if last_written_index >= 0:
        print(f"Resuming from {RESULTS_PATH} at index {last_written_index + 1}")
    else:
        print(f"Starting fresh output to {RESULTS_PATH}")

    model = Model()

    start_perf = time.perf_counter()

    total_number_rows = 0
    for dataset_name, df in all_datasets.items():
        total_number_rows += df.shape[0]

    processed_this_run = 0
    global_row_idx = 0
    is_first_write = (not os.path.exists(RESULTS_PATH)) or (os.path.getsize(RESULTS_PATH) == 0)
    for dataset_name, df in all_datasets.items():
        for index, row in df.iterrows():
            if global_row_idx <= last_written_index:
                global_row_idx += 1
                continue

            question_start = time.perf_counter()
            dataframe_to_append = model.run_batch_and_compute_confidence(
                dataset_name= dataset_name, 
                categories = [row['categorie']], 
                subcategories = [row['subcategorie']],
                questions = [row['question']],
                right_answers = [row['answer']],
                question_types = [row['answer_type']]
                )

            # Ensure the CSV "index" column is labeled and unique across streamed writes.
            dataframe_to_append.index = [global_row_idx]

            write_header = is_first_write
            write_mode = "w" if is_first_write else "a"
            dataframe_to_append.to_csv(
                RESULTS_PATH,
                mode=write_mode,
                header=write_header,
                index=True,
                index_label="index",
                sep=CSV_SEP,
            )
            is_first_write = False

            processed_this_run += 1
            global_row_idx += 1

            rows_done_global = (last_written_index + 1) + processed_this_run

            question_seconds = time.perf_counter() - question_start
            elapsed_seconds = time.perf_counter() - start_perf
            avg_seconds = elapsed_seconds / processed_this_run if processed_this_run else 0.0
            remaining = max(0, total_number_rows - rows_done_global)
            eta_seconds = max(0.0, avg_seconds * remaining)
            eta_td = timedelta(seconds=int(eta_seconds))
            finish_wall = datetime.now() + timedelta(seconds=eta_seconds)

            print(
                f"Rows processed: {rows_done_global} / {total_number_rows} | "
                f"this question: {question_seconds:.2f}s | avg/question: {avg_seconds:.2f}s | "
                f"Total est. running time: {eta_td} | now: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
                f"est. finish wall time: {finish_wall.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    print(f"Finished! Total time: {timedelta(seconds=int(time.perf_counter() - start_perf))}")
    


if __name__ == "__main__":
    run_model_on_dataset()

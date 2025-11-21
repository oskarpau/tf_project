from model import Model


# !pip install -U bitsandbytes
import os
from google.colab import drive
import pandas as pd

# todo
# kõik vastused ühele kujule . stringide list
# answerist <<=5>> välja

'''
dataset_name
categories
subcategories
question
rightanswers list
question type = "true/false", "multi string" list
'''

base_path ="../datasets/"

# 3. For strategyqa_dataset:
strategyqa_dev_df = pd.read_json(os.path.join(base_path, 'strategyqa_dataset', 'dev.json'))
strategyqa_train_df = pd.read_json(os.path.join(base_path, 'strategyqa_dataset', 'train.json'))

# 4. For gsm8k_datasets:
gsm8k_test_df = pd.read_parquet(os.path.join(base_path, 'gsm8k_datasets', 'test-00000-of-00001.parquet'))
gsm8k_train_df = pd.read_parquet(os.path.join(base_path, 'gsm8k_datasets', 'train-00000-of-00001.parquet'))

# 5. For maqa_datasets:
maqa_dfs = {}
maqa_dataset_path = os.path.join(base_path, 'maqa_datasets')
for filename in os.listdir(maqa_dataset_path):
    if filename.endswith('.json'):
        file_path = os.path.join(maqa_dataset_path, filename)
        df_name = os.path.splitext(filename)[0] # Get filename without extension
        maqa_dfs[df_name] = pd.read_json(file_path)

# Set display option to show full column content
pd.set_option('display.max_colwidth', None)

all_datasets = {
    'strategyqa_dev': strategyqa_dev_df,
    'strategyqa_train': strategyqa_train_df,
    'gsm8k_test': gsm8k_test_df,
    'gsm8k_train': gsm8k_train_df
}

# Add MAQA datasets to the main dictionary
for name, df in maqa_dfs.items():
    all_datasets[f'maqa_{name}'] = df

'''
print("All loaded DataFrames are now organized into the 'all_datasets' dictionary.")
print(f"Total DataFrames in dictionary: {len(all_datasets)}")
print("Keys in all_datasets:")
for key in all_datasets.keys():
    print(f"- {key}")
'''


#DATA CLEANSING AND TRANSFORMATION STARTS HERE
#####################




# DATA CLEANING TRANSFORMATION
# 1. SOLVE << ... >>>

import re
import pandas as pd

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

import re

def split_multi_part_questions(row):
    question_text = row['question']
    original_answers = row['answer'] # This is expected to be a list like ['a', 'b']
    new_rows = []
    parts = re.findall(r'\((\w)\)\s*(.*?)(?=\(\w\)|$)', question_text, re.DOTALL)

    if not parts:
        return [{**row.to_dict(), 'question': question_text, 'answer': original_answers}]

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

import pandas as pd

# Get the maqa_MAQA_commonsense_reasoning DataFrame
maqa_commonsense_df = all_datasets['maqa_MAQA_commonsense_reasoning']

# Apply the function to each row and collect the results
expanded_rows = []
for index, row in maqa_commonsense_df.iterrows():
    expanded_rows.extend(split_multi_part_questions(row))

# Create a new DataFrame from the expanded rows
expanded_maqa_commonsense_df = pd.DataFrame(expanded_rows)

# Replace the original DataFrame in all_datasets with the expanded one
all_datasets['maqa_MAQA_commonsense_reasoning'] = expanded_maqa_commonsense_df


#data processing
# 3
# 18.0 -> 18 ROUND

import numpy as np

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
        #print(f"  '{dataset_name}' 'answer' column rounded.")



#cleansin
#4 all in format ['string]


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
    sample_df = all_datasets['maqa_MAQA_commonsense_reasoning'].head(2)
    for index, row in sample_df.iterrows():
        answer_value = row['answer']
        #print(f"  Question: {row['question'][:50]}...")
        #print(f"  Answer: {answer_value}, Type: {type(answer_value)}, Item Type: {type(answer_value[0]) if answer_value else 'N/A'}")

if 'strategyqa_dev' in all_datasets:
    #print("\n--- strategyqa_dev sample ---")
    sample_df = all_datasets['strategyqa_dev'].head(2)
    for index, row in sample_df.iterrows():
        answer_value = row['answer']
        #print(f"  Question: {row['question'][:50]}...")
        #print(f"  Answer: {answer_value}, Type: {type(answer_value)}, Item Type: {type(answer_value[0]) if answer_value else 'N/A'}")


#################################
# END OF CLEANSING
################################


# SRC/RUN_DATASETS



RESULTS_PATH = ""
BATCH_SIZE = 1




def run_model_on_dataset():
    """
    This function loops over all datasets, their categories and subcategories
    """

    # TODO: Somehow loop over dataset so the questions can be processed at least in
    #  batches in 8

    datasets = []
    results_df = pd.DataFrame()

    model = Model()

    for dataset in datasets:
        for categorie in dataset:
            for subcategorie in categorie:
                # TODO: Put categories, subcategories and questins into lists
                dataframe_to_append = model.run_batch_and_compute_confidence(
                    dataset_name="", 
                    categories = [], 
                    subcategories = [],
                    questions = [],
                    right_answers = []
                    )
                results_df.concat([results_df, dataframe_to_append])
    
    # TODO: save results_dfto RESULTS_PATH


if __name__ == "__main__":
    run_model_on_dataset()
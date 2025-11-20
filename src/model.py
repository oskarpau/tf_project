from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import os
import time
import psutil
import math
import pandas as pd

from config import COLUMN_SCHEMA
from confidence_methods import *


def _bytes_to_mib(value: int) -> float:
    """Convert a raw byte count to mebibytes for human-friendly reporting."""
    return value / (1024 ** 2)

def _print_memory_snapshot(label: str) -> None:
    """Print host and device memory usage for the current process."""
    process = psutil.Process(os.getpid())
    rss_mib = _bytes_to_mib(process.memory_info().rss)
    print(f"[Memory] {label}: CPU RSS {rss_mib:.2f} MiB")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for device_idx in range(torch.cuda.device_count()):
            allocated = _bytes_to_mib(torch.cuda.memory_allocated(device_idx))
            reserved = _bytes_to_mib(torch.cuda.memory_reserved(device_idx))
            total = _bytes_to_mib(torch.cuda.get_device_properties(device_idx).total_memory)
            print(
                f"           GPU {device_idx}: allocated {allocated:.2f} MiB | "
                f"reserved {reserved:.2f} MiB | total {total:.2f} MiB"
            )

class Model:
    ########################################################
    # START OF INIT
    ########################################################
    def __init__(self, running_in_colab=False):
        self.model = None
        self.model_name = "Qwen/Qwen3-VL-4B-Instruct"
        self.running_in_colab = running_in_colab
        self.processor = None

        _print_memory_snapshot("Start")
        init_start = time.perf_counter()
        if not self.running_in_colab:
            # Configure int8 quantization
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            # USE THIS LOCALLY AND ON LINUX
            # Flash Attention only works in Linux and not in Colab
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation="flash_attention_2"
            )

        else:
            # USE THIS WHEN RUNNING IN COLAB
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        _print_memory_snapshot("Post-init")
        init_time = time.perf_counter() - init_start
        print(f"[Init] Loaded {self.model_name} in {init_time:.2f} seconds")
    ########################################################
    # END OF INIT
    ########################################################

    ########################################################
    # START OF PREPROCESSING PROMPT
    ########################################################
    def preprocess_prompt(
            self,
            quest_answ_confid_model_answ: list[dict],
            use_verbal_confidence: bool = False,
            second_try: bool = False
        ) -> tuple[list, list]:

        # Wrap each plain-text prompt in the chat schema that the processor expects.
        batched_messages = []
        active_indexes = []
        for idx, qac_model_answer in enumerate(quest_answ_confid_model_answ):
            question      =  qac_model_answer["question"]
            confidence    =  qac_model_answer["confidence"]
            model_answer  =  qac_model_answer["model answer"]
            question_type =  qac_model_answer["question type"]
            wrong_on_first_try = qac_model_answer["wrong on first try"]

            if second_try and not wrong_on_first_try:
                # Only re-ask questions that were incorrect initially
                continue

            prompt = ""
            prompt_start = None
            if not second_try:  
                if not use_verbal_confidence:
                    if question_type == "multi_str":
                        prompt_start = ( 
                        "Instruction : Given a question that has multiple answers,"
                        "answer the question, "
                        "following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer"
                        "2. When providing an answer, use the format ||ANSWERS||"
                        "where ANSWERS are the answers to the given question."
                        "3. Separate each answer of ANSWERS with a comma and"
                        "a space."
                        "Use the following format for the final answer:"
                        "||ANSWER1, ANSWER2, ANSWER3||"
                        "Now, please answer this question."
                        "Question: " 
                        )
                    else:
                        prompt_start = ( 
                        "Instruction : Given a question that has only one correct answer as True or False,"
                        "answer the question, "
                        "following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer"
                        "2. When providing an answer, use the format ||ANSWER||"
                        "where ANSWER is the answer to the given question."
                        "3. Answer only with ||True|| or ||False||."
                        "Use the following format for the final answer:"
                        "||ANSWER||"
                        "Now, please answer this question."
                        "Question: " 
                        )                      

                else:
                    if question_type == "multi_str":
                        prompt_start = ( 
                        "Instruction : Given a question that has multiple answers,"
                        "answer the question and then provide the confidence in this"
                        "answer, which indicates how likely you think your answer"
                        "is true, following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer and"
                        "confidence."
                        "2. When providing an answer, use the format ||ANSWERS||"
                        "where ANSWERS are the answers to the given question."
                        "3. Separate each answer of ANSWERS with a comma and"
                        "a space."
                        "4. The confidence should be a numerical number in the"
                        "range of 0-100."
                        "Use the following format for the final answer and"
                        "confidence:"
                        "||ANSWER1, ANSWER2, ANSWER3||, CONFIDENCE"
                        "Now, please answer this question."
                        "Question: "
                        )
                    else:
                        prompt_start = ( 
                        "Instruction : Given a question that has only one correct answer as True or False,"
                        "answer the question and then provide the confidence in this"
                        "answer, which indicates how likely you think your answer"
                        "is true, following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer and"
                        "confidence."
                        "2. When providing an answer, use the format ||ANSWER||"
                        "where ANSWER is the answer to the given question."
                        "3. Answer only with ||True||, CONFIDENCE or ||False||, CONFIDENCE."
                        "4. The confidence should be a numerical number in the"
                        "range of 0-100."
                        "Use the following format for the final answer and"
                        "confidence:"
                        "||ANSWER||, CONFIDENCE"
                        "Now, please answer this question."
                        "Question: "
                        )                       

            elif second_try and wrong_on_first_try: # We only create the prompts for questions that were answered wrongly
                if not use_verbal_confidence:
                    if question_type == "multi_str":
                        prompt_start = ( 
                        "Instruction : Given a question that has multiple answers,"
                        "answer the question, "
                        "following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer"
                        "2. When providing an answer, use the format ||ANSWERS||"
                        "where ANSWERS are the answers to the given question."
                        "3. Separate each answer of ANSWERS with a comma and"
                        "a space."
                        "You were asked the same question previously and"
                        f"you answer was ||{model_answer}||, with a confidence of {confidence}."
                        "Use the following format for the final answer:"
                        "||ANSWER1, ANSWER2, ANSWER3||"
                        "Now, please answer this question."
                        "Question: " 
                        )
                    else:
                        prompt_start = ( 
                        "Instruction : Given a question that has only one correct answer as True or False,"
                        "answer the question, "
                        "following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer"
                        "2. When providing an answer, use the format ||ANSWER||"
                        "where ANSWER is the answer to the given question."
                        "3. Answer only with ||True|| or ||False||."
                        "Use the following format for the final answer:"
                        "||ANSWER||"
                        "You were asked the same question previously and"
                        f"you answer was ||{model_answer}||, with a confidence of {confidence}."
                        "Use the following format for the final answer:"
                        "||ANSWER||"
                        "Now, please answer this question."
                        "Question: " 
                        )                        


                else:
                    if question_type == "multi_str":
                        prompt_start = ( 
                        "Instruction : Given a question that has multiple answers,"
                        "answer the question and then provide the confidence in this"
                        "answer, which indicates how likely you think your answer"
                        "is true, following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer and"
                        "confidence."
                        "2. When providing an answer, use the format ||ANSWERS||"
                        "where ANSWERS are the answers to the given question."
                        "3. Separate each answer of ANSWERS with a comma and"
                        "a space."
                        "4. The confidence should be a numerical number in the"
                        "range of 0-100."
                        "You were asked the same question previously and"
                        f"you answer was ||{model_answer}||, with a confidence of {confidence}"
                        "Use the following format for the final answer and"
                        "confidence:"
                        "||ANSWER1, ANSWER2, ANSWER3||, CONFIDENCE"
                        "Now, please answer this question."
                        "Question: "
                        )
                    else:
                        prompt_start = ( 
                        "Instruction : Given a question that has only one correct answer as True or False,"
                        "answer the question and then provide the confidence in this"
                        "answer, which indicates how likely you think your answer"
                        "is true, following the instructions below:"
                        "1. Keep your response as brief as possible without"
                        "any explanation, and then provide your answer and"
                        "confidence."
                        "2. When providing an answer, use the format ||ANSWER||"
                        "where ANSWER is the answer to the given question."
                        "3. Answer only with ||True||, CONFIDENCE or ||False||, CONFIDENCE."
                        "4. The confidence should be a numerical number in the"
                        "range of 0-100."
                        "You were asked the same question previously and"
                        f"you answer was ||{model_answer}||, with a confidence of {confidence}"
                        "Use the following format for the final answer and"
                        "confidence:"
                        "||ANSWER||, CONFIDENCE"
                        "Now, please answer this question."
                        "Question: "
                        )                        

                
            if prompt_start is None:
                # Question type not supported for the current configuration
                continue

            prompt += prompt_start + question

            # Each element in ``batched_messages`` is a full chat exchange adhering to
            # the schema expected by the processor.
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            batched_messages.append(conversation)
            active_indexes.append(idx)

        return batched_messages, quest_answ_confid_model_answ, active_indexes

    ########################################################
    # END OF PREPROCESSING PROMPT
    ########################################################

    ########################################################
    # START OF RUNNING MODEL AND GENERATING TOKENS
    ########################################################

    def run_model(
        self,
        batched_prompts: list,
    ) -> tuple[list, list, list]:
        # Apply the chat template to produce padded token tensors ready for the model.
        inputs = self.processor.apply_chat_template(
            batched_prompts,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            padding_side='left',
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)

        # Record the actual number of prompt tokens for each batch element so we can
        # strip them from the generated sequence afterwards.
        input_lengths = [int(length) for length in inputs.attention_mask.sum(dim=1).tolist()]

        # Run inference with token scores
        output = self.model.generate(
            **inputs,
            max_new_tokens=128,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True
        )

        # Slice away the prompt tokens to keep only the generated continuation per sample.
        generated_ids_trimmed = [
            out_ids[input_length:] for input_length, out_ids in zip(input_lengths, output.sequences)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output, output_text, generated_ids_trimmed



    ########################################################
    # END OF RUNNING MODEL AND GENERATING TOKENS
    ########################################################


    ########################################################
    # START OF CALCULATING CONFIDENCE AND FILTERING CORRECT ANSWERS
    ########################################################

    def calculate_conf_and_filter_correct(
            self,
            df_out,
            quest_answ_confid_model_answ,
            output,
            output_text,
            generated_ids_trimmed,
            verbal = False,
            second_try = False,
            wrong_answer_indexes = None,
            active_indexes = None
        ) -> tuple[pd.DataFrame, list]:

        if wrong_answer_indexes is None:
            wrong_answer_indexes = []
        if active_indexes is None:
            active_indexes = list(range(len(output_text)))

        # Determine if answer was correct
        for i, text in enumerate(output_text):
            question_idx = active_indexes[i]
            question_entry = quest_answ_confid_model_answ[question_idx]
            # Extract answer string between ||...||
            answer_str = text.lower().split("||")[1]
            # Split answers, strip whitespace
            model_answer = [a.strip() for a in answer_str.split(",")]
            correct_answer = True
            for right in question_entry["right answer"]:
                if right.lower().strip() not in model_answer:
                    correct_answer = False
                    if not second_try:
                        wrong_answer_indexes.append(question_idx)
                    break
            
            # Now we add this to entry in the DataFrame
            if not second_try:
                df_out.loc[question_idx, "first_try_correct"] = correct_answer
                df_out.loc[question_idx, "first_try_answer"] = model_answer
                question_entry["wrong on first try"] = not correct_answer
                question_entry["model answer"] = answer_str
                
            else:
                df_out.loc[question_idx, "second_try_correct"] = correct_answer
                df_out.loc[question_idx, "second_try_answer"] = model_answer
                question_entry["model second answer"] = answer_str
            

        # Calculate confidence and add to DataFrame
        confidence_for_second_try = []
        if not verbal:
            confidence_max_prob = estimate_confidence_max_prob(output, generated_ids_trimmed, processor=self.processor)
            confidence_entropy = estimate_confidence_entropy(output, generated_ids_trimmed, processor=self.processor)
            confidence_margin = estimate_confidence_margin(output, generated_ids_trimmed, processor=self.processor)

            for batch_pos, question_idx in enumerate(active_indexes):
                if not second_try:
                    df_out.loc[question_idx, "first_try_max_prob"] = round(confidence_max_prob[batch_pos], 2)
                    df_out.loc[question_idx, "first_try_entropy"] = round(confidence_entropy[batch_pos], 2)
                    df_out.loc[question_idx, "first_try_margin"] = round(confidence_margin[batch_pos], 2)
                else:
                    df_out.loc[question_idx, "second_try_max_prob"] = round(confidence_max_prob[batch_pos], 2)
                    df_out.loc[question_idx, "second_try_entropy"] = round(confidence_entropy[batch_pos], 2)
                    df_out.loc[question_idx, "second_try_margin"] = round(confidence_margin[batch_pos], 2)

            confidence_for_second_try = [round(conf, 2) for conf in confidence_entropy]
        else:
            confidence_verbal = verbal_confidence(output_text)
            for batch_pos, question_idx in enumerate(active_indexes):
                if not second_try:
                    df_out.loc[question_idx, "first_try_verbal"] = confidence_verbal[batch_pos]
                else:
                    df_out.loc[question_idx, "second_try_verbal"] = confidence_verbal[batch_pos]

            confidence_for_second_try = [round(conf, 2) for conf in confidence_verbal]
        
        if not second_try and wrong_answer_indexes:
            wrong_idx_set = set(wrong_answer_indexes)
            for batch_pos, question_idx in enumerate(active_indexes):
                if question_idx in wrong_idx_set:
                    quest_answ_confid_model_answ[question_idx]["confidence"] = confidence_for_second_try[batch_pos]
        elif second_try:
            for batch_pos, question_idx in enumerate(active_indexes):
                quest_answ_confid_model_answ[question_idx]["model second confidence"] = confidence_for_second_try[batch_pos]
            
        return df_out, quest_answ_confid_model_answ, wrong_answer_indexes

    ########################################################
    # END OF CALCULATING CONFIDENCE AND FILTERING CORRECT ANSWERS
    ########################################################

    ########################################################
    # START OF MAIN FUNCTION FOR PREPROCESSING INPUT AND CALCULATING CONFIDENCE
    ########################################################

    def run_batch_and_compute_confidence(
            self,
            dataset_name: str, 
            categories: list, 
            subcategories: list,   
            questions: list,
            right_answers: list,
            question_types: list # true/false, multi_str
        ) -> pd.DataFrame:

        """
        Processes a list of text questions, computes confidence based on 
        verbalized confidence and average values of
        maximum token propabilities, entropies ad margins.

        Runs all prompts at maximum of four times:
        1. For calculating token probabilities
        2. For calculating verbalized probabilities
        3. For calculating token probabilities of wrongly answered question
        4. For calculating verbalized probabilities of wrongly answered question

        For question answered correctly on the first try, the second try columns
        are filled with NaNs.

        Args:
            dataset_name (str): Name of the dataset
            categorie (list[str]): 
            subcategorie (list[str]):    
            questions (list[str]):
            right_answers (list[str]):
            question_types (list[str]):

        Returns:
            pd.DataFrame: DataFrame with all the results file's columns,
            ready to be appended to the current running DataFrame.
        """
        df_out = pd.DataFrame({
            col: pd.Series(dtype=dtype)
            for col, dtype in COLUMN_SCHEMA.items()
        })

        quest_answ_confid_model_answ = []

        wrong_answer_indexes = []

        # Create a list for holding information about each question in dict structure
        for i, qa in enumerate(questions):
            entry = dict()
            entry["question"] = questions[i]
            entry["question type"] = question_types[i]
            right_answer = right_answers[i]
            if isinstance(right_answer, str):
                normalized_right_answers = [right_answer]
            else:
                normalized_right_answers = list(right_answer)
            entry["right answer"] = normalized_right_answers
            entry["confidence"] = []
            entry["model answer"] = []
            entry["wrong on first try"] = False
            entry["model second answer"] = []
            entry["model second confidence"] = []
            quest_answ_confid_model_answ.append(entry)

        for i, entry in enumerate(quest_answ_confid_model_answ):
            # Add entry to DataFrame
            df_out.loc[len(df_out)] = {
                "dataset": dataset_name,
                "categorie": categories[i],
                "subcategorie": subcategories[i],
                "question": quest_answ_confid_model_answ[i]["question"],
                "question_type": quest_answ_confid_model_answ[i]["question type"],
                "right_answer": quest_answ_confid_model_answ[i]["right answer"],
                "first_try_answer": None,
                "first_try_max_prob": None,  
                "first_try_entropy": None,   
                "first_try_margin": None,    
                "first_try_verbal": None,    
                "first_try_correct": None,
                "second_try_answer": None,
                "second_try_max_prob": None, 
                "second_try_entropy": None,  
                "second_try_margin": None,   
                "second_try_verbal": None,   
                "second_try_correct": None   
            }          


        # 1. Calculate token probabilities for first try
        batched_prompts, quest_answ_confid_model_answ, active_indexes =  self.preprocess_prompt(quest_answ_confid_model_answ)
        output, output_text, generated_ids_trimmed = self.run_model(batched_prompts)
        df_out, quest_answ_confid_model_answ, wrong_answer_indexes = self.calculate_conf_and_filter_correct(
            df_out,
            quest_answ_confid_model_answ,
            output, 
            output_text, 
            generated_ids_trimmed,
            wrong_answer_indexes=wrong_answer_indexes,
            active_indexes=active_indexes
        )
        
        # 2. Calculate token probabilities of wrongly answered question
        if wrong_answer_indexes != []:
            batched_prompts, quest_answ_confid_model_answ, active_indexes =  self.preprocess_prompt(
                quest_answ_confid_model_answ, 
                second_try=True
            )
            output, output_text, generated_ids_trimmed = self.run_model(batched_prompts)
            df_out, quest_answ_confid_model_answ, wrong_answer_indexes = self.calculate_conf_and_filter_correct(
                df_out,
                quest_answ_confid_model_answ,
                output, 
                output_text, 
                generated_ids_trimmed,
                wrong_answer_indexes=wrong_answer_indexes,
                second_try=True,
                active_indexes=active_indexes
            )

        wrong_answer_indexes = []

        # 3. Calculate verbalized probabilities for first try
        batched_prompts, quest_answ_confid_model_answ, active_indexes =  self.preprocess_prompt(
            quest_answ_confid_model_answ, 
            use_verbal_confidence=True
        )
        output, output_text, generated_ids_trimmed = self.run_model(batched_prompts)
        df_out, quest_answ_confid_model_answ, wrong_answer_indexes = self.calculate_conf_and_filter_correct(
            df_out,
            quest_answ_confid_model_answ,
            output, 
            output_text, 
            generated_ids_trimmed,
            wrong_answer_indexes=wrong_answer_indexes,
            verbal=True,
            active_indexes=active_indexes
        )

        # 4. Calculate verbalized probabilities of wrongly answered question
        if wrong_answer_indexes != []:
            batched_prompts, quest_answ_confid_model_answ, active_indexes =  self.preprocess_prompt(
                quest_answ_confid_model_answ, 
                use_verbal_confidence=True,
                second_try=True
            )
            output, output_text, generated_ids_trimmed = self.run_model(batched_prompts)
            df_out, quest_answ_confid_model_answ, wrong_answer_indexes = self.calculate_conf_and_filter_correct(
                df_out,
                quest_answ_confid_model_answ,
                output, 
                output_text, 
                generated_ids_trimmed,
                wrong_answer_indexes=wrong_answer_indexes,
                second_try=True,
                verbal=True,
                active_indexes=active_indexes
            )
        return df_out

    ########################################################
    # END OF MAIN FUNCTION FOR PREPROCESSING INPUT AND CALCULATING CONFIDENCE
    ########################################################
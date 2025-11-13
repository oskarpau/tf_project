from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import os
import time
import psutil
import math
import pandas as pd

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
    def __init__(self, running_in_colab=False):
        self.model = None
        self.model_name = "Qwen/Qwen3-VL-4B-Instruct"
        self.running_in_colab = running_in_colab

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
        processor = AutoProcessor.from_pretrained(self.model_name)

        _print_memory_snapshot("Post-init")
        init_time = time.perf_counter() - init_start
        print(f"[Init] Loaded {self.model_name} in {init_time:.2f} seconds")

    def batch_messages(
            messages: list, 
            use_verbal_confidence: bool = False
        ) -> list:

        # Wrap each plain-text prompt in the chat schema that the processor expects.
        batched_messages = []
        for prompt_text in messages:
            # Each element in ``batched_messages`` is a full chat exchange adhering to
            # the schema expected by the processor.
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

            if use_verbal_confidence:
                conversation[0]["content"].append({"type": "text", "text": "Numerical confidence: [to be filled by model]"})

            batched_messages.append(conversation)

        return batched_messages

    def run_batch_and_compute_confidence(
            dataset_name: str, 
            categories: list, 
            subcategories: list,   
            questions: list
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

        Returns:
            pd.DataFrame: DataFrame with all the results file's columns,
            ready to be appended to the current running DataFrame.
        """

        # 1. Calculate token probabilities for first try
        

        # 2. Calculate verbalized probabilities for first try

        # 3. Calculate token probabilities of wrongly answered question

        # 4. Calculate verbalized probabilities of wrongly answered question

        pass
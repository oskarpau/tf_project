"""Batch chat inference demo with confidence heuristics and memory telemetry.

This script loads the multimodal Qwen 3 VL model with an int8 quantized
configuration, runs multiple textual prompts in a single batch, and reports
per-sample confidence scores along with GPU memory statistics.  It is intended
as an exploratory tool for understanding how to:

* Prepare multi-sample chat conversations with the Hugging Face chat template
    interface.
* Request token-level logits from ``model.generate`` to compute custom
    confidence heuristics (max probability, entropy, margin).
* Track execution timing, throughput, and GPU memory usage for profiling.
"""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import os
import time
import psutil
import math

# Set this according to current environment
RUNNING_IN_COLAB = False

USE_VERBAL_CONFIDENCE = False


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

def estimate_confidence_max_prob(output, generated_ids_trimmed, debug=False):
    """Estimate confidence from token probabilities for each sample in the batch."""
    batch_size = output.sequences.shape[0]
    probs_per_sample = [[] for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(output.logits):
        probs_step = torch.nn.functional.softmax(step_scores, dim=-1)
        max_probs, max_indices = torch.max(probs_step, dim=-1)

        for sample_idx in range(batch_size):
            # ``generate`` may emit fewer tokens for some samples; skip when a
            # particular timestep falls outside the generated continuation.
            if step_idx >= len(generated_ids_trimmed[sample_idx]):
                continue

            token_id = max_indices[sample_idx].item()
            token_str = processor.tokenizer.decode([token_id])
            if debug:
                print(
                    f"Sample {sample_idx}, Step {step_idx}: Most probable token {token_str!r}, "
                    f"Prob {max_probs[sample_idx].item():.4f}"
                )
            probs_per_sample[sample_idx].append(max_probs[sample_idx].item())

    avg_probs = []
    for sample_probs in probs_per_sample:
        if not sample_probs:
            avg_probs.append(0.0)
        else:
            avg_probs.append(sum(sample_probs) / len(sample_probs))

    return avg_probs

def estimate_confidence_entropy(output, generated_ids_trimmed, debug=False):
    """Compute the average entropy of the top tokens for each generated sample."""
    batch_size = output.sequences.shape[0]
    entropies_per_sample = [[] for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(output.logits):
        probs_step = torch.nn.functional.softmax(step_scores, dim=-1)
        topk_values, topk_indices = torch.topk(probs_step, 5, dim=-1)

        for sample_idx in range(batch_size):
            # Guard against variable-length continuations, as above.
            if step_idx >= len(generated_ids_trimmed[sample_idx]):
                continue

            entropy = 0.0
            for rank, (token_id, prob) in enumerate(
                zip(topk_indices[sample_idx].tolist(), topk_values[sample_idx].tolist()), 1
            ):
                if prob > 0:
                    contribution = -prob * math.log(prob + 1e-12)
                    entropy += contribution
                else:
                    contribution = 0.0

                if debug:
                    token_str = processor.tokenizer.decode([token_id])
                    print(
                        f"Sample {sample_idx}, Step {step_idx}, Top {rank}: Token {token_id} ('{token_str}'), Prob {prob:.4f}, "
                        f"Contribution {contribution:.4f}"
                    )

            entropies_per_sample[sample_idx].append(entropy)

    avg_entropies = []
    for sample_entropies in entropies_per_sample:
        if not sample_entropies:
            avg_entropies.append(0.0)
        else:
            avg_entropies.append(sum(sample_entropies) / len(sample_entropies))

    return avg_entropies

def estimate_confidence_margin(output, generated_ids_trimmed, debug=False):
    """Measure how far apart the top-2 probabilities are across generated tokens."""
    batch_size = output.sequences.shape[0]
    margins_per_sample = [[] for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(output.logits):
        probs_step = torch.nn.functional.softmax(step_scores, dim=-1)
        topk_values, topk_indices = torch.topk(probs_step, 2, dim=-1)

        for sample_idx in range(batch_size):
            # Skip samples that already finished generating.
            if step_idx >= len(generated_ids_trimmed[sample_idx]):
                continue

            margin = topk_values[sample_idx][0].item() - topk_values[sample_idx][1].item()
            if debug:
                token1_str = processor.tokenizer.decode([topk_indices[sample_idx][0].item()])
                token2_str = processor.tokenizer.decode([topk_indices[sample_idx][1].item()])
                print(
                    f"Sample {sample_idx}, Step {step_idx}: Top1 '{token1_str}' Prob {topk_values[sample_idx][0].item():.4f} - "
                    f"Top2 '{token2_str}' Prob {topk_values[sample_idx][1].item():.4f} = Margin {margin:.4f}"
                )
            margins_per_sample[sample_idx].append(margin)

    avg_margins = []
    for sample_margins in margins_per_sample:
        if not sample_margins:
            avg_margins.append(0.0)
        else:
            avg_margins.append(sum(sample_margins) / len(sample_margins))

    return avg_margins

# --- Script start ---
#
# The sections below follow a rough order of operations: telemetry helpers,
# model loading, prompt preparation, generation, metric computation, and
# reporting.  These headings make it easier to jump to the part that is most
# relevant for experimentation.
_print_memory_snapshot("Start")

if torch.cuda.is_available():
    # Reset peak counters so the summary at the end reflects only this run.
    for device_idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_idx}")
        torch.cuda.reset_peak_memory_stats(device)

# --- Model configuration --------------------------------------------------
model_name = "Qwen/Qwen3-VL-4B-Instruct"

# Configure int8 quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

init_start = time.perf_counter()

model = None

if RUNNING_IN_COLAB:
    # USE THIS WHEN RUNNING IN COLAB
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )

else:
    # USE THIS LOCALLY AND ON LINUX
    # Flash Attention only works in Linux and not in Colab
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )


processor = AutoProcessor.from_pretrained(model_name)

_print_memory_snapshot("Post-init")
init_time = time.perf_counter() - init_start
print(f"[Init] Loaded {model_name} in {init_time:.2f} seconds")

start_time = time.perf_counter()

#             {"type": "text", "text": "Keep your answer very short. I need you to act like a humanoid robot. There are 3 possible location for coffee beans. Next to the coffee machine, in the living room or under the bed. Which place would you start looking at?"},
#             {"type": "text", "text": "I need you to act like a humanoid robot. There are 3 possible location for coffee beans. Next to the coffee machine, in the living room or under the bed. Which place would you start looking at? Also make a plan."},


# --- Prompt definition ----------------------------------------------------
prompts = [
    (
        "Keep your answer very short. I need you to act like a humanoid robot. "
        "There are 3 possible location for coffee beans. Next to the coffee machine, "
        "in the living room or under the bed. Which place would you start looking at?"
    ),
    (
        "Keep your answer very short. I need you to act like a humanoid robot. "
        "There are 3 possible location for coffee beans. Next to the coffee machine, "
        "in the shop or under the bed. Which place would you start looking at?"
    ),
]

# Wrap each plain-text prompt in the chat schema that the processor expects.
batched_messages = []
for prompt_text in prompts:
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

    if USE_VERBAL_CONFIDENCE:
        conversation[0]["content"].append({"type": "text", "text": "Numerical confidence: [to be filled by model]"})

    batched_messages.append(conversation)

# Apply the chat template to produce padded token tensors ready for the model.
inputs = processor.apply_chat_template(
    batched_messages,
    tokenize=True,
    add_generation_prompt=True,
    padding=True,
    padding_side='left',
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Record the actual number of prompt tokens for each batch element so we can
# strip them from the generated sequence afterwards.
input_lengths = [int(length) for length in inputs.attention_mask.sum(dim=1).tolist()]

# Run inference with token scores
output = model.generate(
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


# # For each generated token, print the top 5 most probable tokens and their scores
# for step_idx, step_scores in enumerate(output.logits):
#     scores_tensor = step_scores[0]  # shape: (vocab_size,)
#     probs = torch.nn.functional.softmax(scores_tensor, dim=-1)
#     topk = torch.topk(probs, 5)
#     print(f"Step {step_idx}: Top 5 tokens:")
#     for rank, (token_id, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), 1):
#         token_str = processor.tokenizer.decode([token_id])
#         score = scores_tensor[token_id].item()
#         print(f"  {rank}. Token {token_id} ('{token_str}'): Score {score:.4f}, Prob {prob:.4f}")

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

elapsed = time.perf_counter() - start_time
# Per-sample token counts and throughput are useful for understanding how
# generation varies across inputs in a shared batch.
tokens_generated = [len(ids) for ids in generated_ids_trimmed]
tokens_per_second = [
    (num_tokens / elapsed) if elapsed > 0 else float("inf")
    for num_tokens in tokens_generated
]

# --- Compute confidence ---
# Each heuristic below traverses the stored logits and produces one scalar per
# sample, which we later surface alongside the textual output.
start_time_conf = time.perf_counter()
confidence_max_prob = estimate_confidence_max_prob(output, generated_ids_trimmed)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on max probabilities took {elapsed_conf:.4f}")

start_time_conf = time.perf_counter()
confidence_entropy = estimate_confidence_entropy(output, generated_ids_trimmed)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on entropy took {elapsed_conf:.4f}")

start_time_conf = time.perf_counter()
confidence_margin = estimate_confidence_margin(output, generated_ids_trimmed)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on margins took {elapsed_conf:.4f}")

# --- Print results ---
# Present outputs and metrics in a consistent per-sample format so downstream
# parsing (or manual inspection) is straightforward.
for idx, text in enumerate(output_text):
    print(f"[Output][Sample {idx}] {text}")

for idx, conf in enumerate(confidence_max_prob):
    print(f"[Confidence][Sample {idx}] Avg max probability: {conf * 100:.2f}%")

for idx, entropy in enumerate(confidence_entropy):
    print(f"[Confidence][Sample {idx}] Avg entropy: {entropy * 100:.2f}")

for idx, margin in enumerate(confidence_margin):
    print(f"[Confidence][Sample {idx}] Avg margin: {margin * 100:.2f}")

print(f"[Timing] Batch took {elapsed:.2f} seconds")
for idx, (num_tokens, tps) in enumerate(zip(tokens_generated, tokens_per_second)):
    print(f"           Sample {idx}: Tokens {num_tokens} | {tps:.2f} tok/s")

# Print total tokens per second for the batch
total_tokens = sum(tokens_generated)
total_tps = total_tokens / elapsed if elapsed > 0 else float("inf")
print(f"[Timing] Total tokens: {total_tokens} | Total tok/s: {total_tps:.2f}")

if torch.cuda.is_available():
    # Peak stats capture the maximum observed footprint since the last reset at
    # the top of the script, covering model load, inference, and confidence work.
    print("[Memory][Peak] Maximum GPU usage during run:")
    for device_idx in range(torch.cuda.device_count()):
        device = torch.device(f"cuda:{device_idx}")
        peak_allocated = _bytes_to_mib(torch.cuda.max_memory_allocated(device))
        peak_reserved = _bytes_to_mib(torch.cuda.max_memory_reserved(device))
        print(
            f"           GPU {device_idx}: peak allocated {peak_allocated:.2f} MiB | "
            f"peak reserved {peak_reserved:.2f} MiB"
        )
else:
    print("[Memory][Peak] GPU unavailable; no VRAM usage recorded.")

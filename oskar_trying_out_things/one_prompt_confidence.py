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
    return value / (1024 ** 2)

def _print_memory_snapshot(label: str) -> None:
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

def estimate_confidence_max_prob(output, input_ids, debug=False):
    """Estimate confidence from token probabilities."""
    probs = []
    # Skip the prompt part of the sequence
    generated_token_ids = output.sequences[0][input_ids.shape[1]:]

    # For each generated step, extract probability of the most probable token
    for step_scores in output.logits:
        probs_step = torch.nn.functional.softmax(step_scores[0], dim=-1)
        max_prob, max_idx = torch.max(probs_step, dim=-1)
        token_str = processor.tokenizer.decode([max_idx.item()])
        if debug:
            print(f"Step: Most probable token: {token_str!r}, Prob: {max_prob.item():.4f}")
        probs.append(max_prob.item())

    if not probs:
        return 0.0

    # Average probability of most probable tokens
    avg_prob = sum(probs) / len(probs)
    return avg_prob

def estimate_confidence_entropy(output, input_ids, debug=False):
    entropies = []
    for step_idx, step_scores in enumerate(output.logits):
        scores_tensor = step_scores[0]  # shape: (vocab_size,)
        probs = torch.nn.functional.softmax(scores_tensor, dim=-1)
        topk = torch.topk(probs, 5)
        entropy = 0.0
        for i, (token_id, prob) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
            if prob > 0:
                entropy += prob * math.log(prob + 1e-12)
            if debug:
                token_str = processor.tokenizer.decode([token_id])
                print(f"Step {step_idx}, Top {i+1}: Token {token_id} ('{token_str}'), Prob {prob:.4f}, Contribution {-prob * math.log(prob + 1e-12):.4f}")
        entropy = -entropy  # Make positive
        entropies.append(entropy)
    if not entropies:
        return 0.0
    avg_entropy = sum(entropies) / len(entropies)
    return avg_entropy

def estimate_confidence_margin(output, input_ids, debug=False):
    margins = []
    for step_idx, step_scores in enumerate(output.logits):
        scores_tensor = step_scores[0]  # shape: (vocab_size,)
        probs = torch.nn.functional.softmax(scores_tensor, dim=-1)
        topk = torch.topk(probs, 2)
        margin = topk.values[0].item() - topk.values[1].item()
        if debug:
            token1_str = processor.tokenizer.decode([topk.indices[0].item()])
            token2_str = processor.tokenizer.decode([topk.indices[1].item()])
            print(f"Step {step_idx}: Top1 '{token1_str}' Prob {topk.values[0].item():.4f} - Top2 '{token2_str}' Prob {topk.values[1].item():.4f} = Margin {margin:.4f}")
        margins.append(margin)
    if not margins:
        return 0.0
    avg_margin = sum(margins) / len(margins)
    return avg_margin

# --- Script start ---
_print_memory_snapshot("Start")

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



messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Keep your answer very short. I need you to act like a humanoid robot. There are 3 possible location for coffee beans. Next to the coffee machine, in the living room or under the bed. Which place would you start looking at?"},
        ],
    }
]

confidence_field = "Numerical confidence: [to be filled by model]"
if USE_VERBAL_CONFIDENCE:
    messages[0]["content"].append({"type": "text", "text": confidence_field})

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Run inference with token scores
output = model.generate(
    **inputs,
    max_new_tokens=128,
    return_dict_in_generate=True,
    output_scores=True,
    output_logits=True
)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, output.sequences)
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
tokens_generated = len(generated_ids_trimmed[0])
tokens_per_second = tokens_generated / elapsed if elapsed > 0 else float("inf")

# --- Compute confidence ---
start_time_conf = time.perf_counter()
confidence_max_prob = estimate_confidence_max_prob(output, inputs.input_ids)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on max probabilities took {elapsed_conf:.4f}")

start_time_conf = time.perf_counter()
confidence_entropy = estimate_confidence_entropy(output, inputs.input_ids)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on entropy took {elapsed_conf:.4f}")

start_time_conf = time.perf_counter()
confidence_margin = estimate_confidence_margin(output, inputs.input_ids)
elapsed_conf = time.perf_counter() - start_time_conf
print(f"Confidence estimation based on margins took {elapsed_conf:.4f}")

# --- Print results ---
print(output_text)
print(f"[Confidence] Estimated model confidence based on average max probabilities: {confidence_max_prob * 100:.2f}%")
print(f"[Confidence] Estimated model confidence based on average entropy: {confidence_entropy * 100:.2f}")
print(f"[Confidence] Estimated model confidence based on average margin: {confidence_margin * 100:.2f}")
print(f"[Timing] Prompt 1 took {elapsed:.2f} seconds | Tokens: {tokens_generated} | {tokens_per_second:.2f} tok/s")

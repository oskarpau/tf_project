from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
import os
import time
import psutil
import math
import numpy as np


def estimate_confidence_max_prob(
        output: torch.Tensor, 
        generated_ids_trimmed, 
        processor,
        debug=False
    ) -> list:
    """Estimate confidence from token probabilities for each sample in the batch.

    Args:
        output: ``GenerateDecoderOnlyOutput`` (or similar) returned by
            ``model.generate`` with ``output_logits=True``.  ``output.logits`` is
            expected to be an iterable of tensors shaped ``(batch, vocab_size)``.
        generated_ids_trimmed: Sequence of per-sample tensors containing only the
            newly generated token IDs (prompt portion removed). The length of each
            item determines how many timesteps to consider for that sample.
        processor: Processor for encoding/decoding tokens
        debug: When ``True``, prints per-step token and probability details.

    Returns:
        list[float]: Average maximum probability per generated token for each
        batch element. Empty generations yield ``0.0``.
    """
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
            avg_probs.append(sum(sample_probs) / len(sample_probs) * 100)

    return avg_probs

def estimate_confidence_entropy(output, generated_ids_trimmed, processor, debug=False):
    """Compute the average entropy of the top tokens for each generated sample.

    Args:
        output: Same structure as in :func:`estimate_confidence_max_prob`; logits
            for each generation step must be accessible via ``output.logits``.
        generated_ids_trimmed: Sequence of generated token ID tensors per sample,
            used to cap the number of timesteps processed for each batch item.
        processor: Processor for encoding/decoding tokens
        debug: Enables verbose logging of per-token probabilities and entropy
            contributions when set to ``True``.

    Returns:
        list[float]: Average entropy (in bits i.e. using log2) measured over the top-5 token
        probabilities for each batch element. Samples with no generated tokens
        produce ``0.0``.
    """
    topk = 5
    max_entropy = math.log(topk, 2)
    batch_size = output.sequences.shape[0]
    entropies_per_sample = [[] for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(output.logits):
        probs_step = torch.nn.functional.softmax(step_scores, dim=-1)

        # Calculate entropies for only 5 most probable tokens
        topk_values, topk_indices = torch.topk(probs_step, topk, dim=-1)

        for sample_idx in range(batch_size):
            # Guard against variable-length continuations, as above.
            if step_idx >= len(generated_ids_trimmed[sample_idx]):
                continue

            entropy = 0.0
            for rank, (token_id, prob) in enumerate(
                zip(topk_indices[sample_idx].tolist(), topk_values[sample_idx].tolist()), 1
            ):
                if prob > 0:
                    contribution = -prob * math.log(prob + 1e-12, 2)
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
            # Normalize and append entropy, also clip for any floating precision related issues
            avg_entropies.append(np.clip(1 - (sum(sample_entropies) / len(sample_entropies)) / max_entropy, 0, 1) * 100)
            
    return avg_entropies

def estimate_confidence_margin(output, generated_ids_trimmed, processor, debug=False):
    """Measure how far apart the top-2 probabilities are across generated tokens.

    Args:
        output: Generation output bundle providing step-wise logits (see
            :func:`estimate_confidence_max_prob`).
        generated_ids_trimmed: Sequence of per-sample generated token tensors used
            to determine how many logit steps belong to each sample.
        processor: Processor for encoding/decoding tokens
        debug: If ``True``, logs the top-2 tokens and their probabilities for each
            processed step.

    Returns:
        list[float]: Average probability margin between the top-1 and top-2 tokens
        for each sample. When a sample yields no new tokens, the margin defaults
        to ``0.0``.
    """
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


def verbal_confidence(output_text):
    """Gets verbal confidence from outputted text

    Args:
        output_text: list[str]
    Returns:
        list[float]: Confidences
    """
    
    confidences = []
    for text in output_text:
        confidences.append(float(text.split(",")[-1]))
    return confidences
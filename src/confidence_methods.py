import torch
import math
import numpy as np
import re


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _try_parse_confidence_from_segment(segment: str) -> tuple[float | None, str]:
    """Try to extract a single confidence value from a candidate segment.

    Returns (value, "") on success, or (None, reason) on failure.
    """
    if segment is None:
        return None, "segment is None"

    segment_str = str(segment)
    numbers = _NUMBER_RE.findall(segment_str)
    if not numbers:
        return None, "no numeric tokens found"

    parsed: list[float] = []
    for token in numbers:
        try:
            parsed.append(float(token))
        except ValueError:
            # Should be rare given the regex, but keep this safe.
            continue

    if not parsed:
        return None, f"numeric tokens found but none could be parsed as float: {numbers!r}"

    in_0_100 = [v for v in parsed if 0.0 <= v <= 100.0]
    in_0_1 = [v for v in parsed if 0.0 <= v <= 1.0]

    if in_0_100:
        # If multiple values exist, use the last one (most common pattern: "..., CONF1||CONF2").
        return float(in_0_100[-1]), ""

    if in_0_1 and not any(v > 1.0 for v in parsed):
        # If everything looks like probabilities, scale to 0-100.
        return float(in_0_1[-1] * 100.0), ""

    return None, f"numeric candidates found {parsed!r} but none in [0, 100] (or all-in-[0,1] prob form)"


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
    
    confidences: list[float] = []
    for text in output_text:
        raw = text
        try:
            if not isinstance(text, str):
                raise TypeError(f"expected str, got {type(text).__name__}")
            stripped = text.strip()
            if not stripped:
                raise ValueError("empty output text")

            # Preferred format per prompt: "||ANSWER(S)||, CONFIDENCE".
            segments_to_try: list[tuple[str, str]] = []

            # 1) Keyword-based (most explicit)
            segments_to_try.append(("keyword 'confidence'", stripped))

            # 2) Portion after the answer delimiters, if present
            parts = stripped.split("||")
            if len(parts) >= 3:
                after_delims = "||".join(parts[2:]).strip()
                segments_to_try.append(("after '||...||'", after_delims))

            # 3) Portion after the last comma
            if "," in stripped:
                segments_to_try.append(("after last comma", stripped.split(",")[-1].strip()))

            # 4) Full text as a last resort
            segments_to_try.append(("full text", stripped))

            chosen: float | None = None
            failure_reasons: list[str] = []

            # Special-case: if "confidence: X" appears anywhere, try to parse from the tail of that match first.
            m = re.search(r"\bconfidence\b\s*[:=]\s*(.*)$", stripped, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1)
                value, reason = _try_parse_confidence_from_segment(candidate)
                if value is not None:
                    chosen = value
                else:
                    failure_reasons.append(f"keyword confidence match failed: {reason}; segment={candidate!r}")

            if chosen is None:
                for label, segment in segments_to_try:
                    value, reason = _try_parse_confidence_from_segment(segment)
                    if value is not None:
                        chosen = value
                        break
                    failure_reasons.append(f"{label} failed: {reason}; segment={segment!r}")

            if chosen is None:
                # Hard failure: print raw text and all reasons, then return NaN.
                print("[verbal_confidence] Could not extract confidence. Returning NaN.")
                print(f"[verbal_confidence] Raw text: {raw!r}")
                print("[verbal_confidence] Reasons:")
                for r in failure_reasons:
                    print(f"  - {r}")
                confidences.append(float("nan"))
                continue

            confidences.append(float(chosen))
        except Exception as e:
            # Safety net: never crash the pipeline due to parsing; emit diagnostics.
            print("[verbal_confidence] Exception while extracting confidence. Returning NaN.")
            print(f"[verbal_confidence] Raw text: {raw!r}")
            print(f"[verbal_confidence] Exception: {type(e).__name__}: {e}")
            confidences.append(float("nan"))

    return confidences
"""
Speculative decoding inference for Qwen2.5-VL with Kangaroo adapter.

This replaces model.generate() with a custom draft-verify loop:
1. Prefill: Run full model on all input tokens (text + visual) normally
2. Draft: Run early layers + adapter to generate candidate tokens
3. Verify: Run remaining layers to check draft tokens
4. Accept tokens until first mismatch (lossless for greedy decoding)

Adapted from Kangaroo's inference_kangaroo.py for Qwen2.5-VL.
"""

import time

import torch
from typing import Optional, Tuple, List
from transformers.cache_utils import DynamicCache


def _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens):
    """Build comprehensive timing and acceptance statistics."""
    decode_time = total_time - prefill_time
    avg_accept = sum(accept_length_list) / len(accept_length_list) if accept_length_list else 0
    tokens_per_second = num_new_tokens / total_time if total_time > 0 else 0
    decode_tokens_per_second = num_new_tokens / decode_time if decode_time > 0 else 0
    return {
        # Acceptance metrics
        'accept_lengths': accept_length_list,
        'avg_accept_length': avg_accept,
        'total_rounds': len(accept_length_list),
        'total_tokens': num_new_tokens,
        # Timing metrics (seconds)
        'total_time': total_time,
        'prefill_time': prefill_time,
        'decode_time': decode_time,
        'draft_times': draft_times,
        'verify_times': verify_times,
        'avg_draft_time': sum(draft_times) / len(draft_times) if draft_times else 0,
        'avg_verify_time': sum(verify_times) / len(verify_times) if verify_times else 0,
        # Speed metrics
        'tokens_per_second': tokens_per_second,
        'decode_tokens_per_second': decode_tokens_per_second,
    }


@torch.no_grad()
def kangaroo_speculative_generate(
    model,  # KangarooQwenModel
    inputs,  # dict with input_ids, attention_mask, pixel_values, etc.
    processor,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
    do_sample: bool = False,
    past_key_values=None,  # existing KV cache from previous turns (streaming)
    debug_verify: bool = False,  # Run full-model verification alongside split draft-verify
):
    """
    Speculative decoding generation for Qwen2.5-VL.

    Args:
        model: KangarooQwenModel instance
        inputs: Processor output dict (input_ids, attention_mask, pixel_values, etc.)
        processor: Qwen2.5-VL processor (for tokenizer info)
        max_new_tokens: Maximum new tokens to generate
        early_exit_layer: Which layer to exit early (default 2)
        speculative_steps: Max draft tokens per round (default 6)
        threshold: Confidence threshold for early stopping draft (default 0.6)
        do_sample: Whether to sample (only greedy supported for now)
        past_key_values: DynamicCache from previous turns for streaming

    Returns:
        output_ids: Generated token IDs (input + generated)
        past_key_values: Updated KV cache
        stats: Dict with timing and acceptance metrics
    """
    assert not do_sample, "Only greedy decoding is supported for speculative decoding"

    # Timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    base_model = model.base_model
    adapter_model = model.adapter_model
    head_model = model.head_model
    device = inputs['input_ids'].device

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    token_eos = tokenizer.eos_token_id
    if isinstance(token_eos, list):
        token_eos_set = set(token_eos)
        token_eos = token_eos[0]
    else:
        token_eos_set = {token_eos}

    input_ids = inputs['input_ids']
    batch_size, context_length = input_ids.shape
    assert batch_size == 1, "Speculative decoding only supports batch_size=1"

    max_length = context_length + max_new_tokens

    # Allocate output buffer
    global_tokens = torch.full((batch_size, max_length), token_eos, dtype=torch.long, device=device)
    global_tokens[:, :context_length] = input_ids

    accept_length_list = [1]
    start_index = context_length

    # ========== STEP 0: Prefill with full model ==========
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_start = time.perf_counter()
    # Build forward kwargs for the full model
    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs.get('attention_mask'),
        'use_cache': True,
        'output_hidden_states': True,
        'return_dict': True,
        'past_key_values': past_key_values,
        # Vision inputs
        'pixel_values': inputs.get('pixel_values'),
        'pixel_values_videos': inputs.get('pixel_values_videos'),
        'image_grid_thw': inputs.get('image_grid_thw'),
        'video_grid_thw': inputs.get('video_grid_thw'),
        'second_per_grid_ts': inputs.get('second_per_grid_ts'),
        # Token dropping (disabled during speculative decoding)
        'drop_method': 'none',
        'drop_threshold': 1.0,
        'drop_absolute': True,
    }
    # Remove None values
    forward_kwargs = {k: v for k, v in forward_kwargs.items() if v is not None}

    output = base_model.model(**forward_kwargs)

    # Store KV cache in the early exit wrapper
    base_model.past_key_values = output.past_key_values

    # Get logits and first generated token
    hidden_state = output.hidden_states[-1]  # After norm (last hidden state from model output)
    logits = head_model(hidden_state)
    first_token = torch.argmax(logits[:, -1, :], dim=-1)
    global_tokens[:, start_index] = first_token.item()

    if debug_verify:
        # Compare our first token with the model's own logits
        model_logits = output.logits
        model_first_token = torch.argmax(model_logits[:, -1, :], dim=-1)
        print(f"[DEBUG] Prefill first token: ours={first_token.item()}, model={model_first_token.item()}, "
              f"match={first_token.item() == model_first_token.item()}")
        # Check hidden_states[-1] == last_hidden_state
        lhs_diff = (output.hidden_states[-1][:, -1, :] - output.hidden_states[-1][:, -1, :]).abs().max().item()
        print(f"[DEBUG] hidden_states[-1] self-diff: {lhs_diff}")

    # Get early exit hidden state for adapter initialization
    hidden_state_early = output.hidden_states[early_exit_layer]

    # Initialize adapter KV cache
    _, adapter_past_key_values = adapter_model.forward_early_stop(
        inputs_embeds=hidden_state_early,
        use_cache=True,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start

    # Per-round timing
    draft_times = []
    verify_times = []

    # Check if first token is EOS
    if first_token.item() in token_eos_set:
        output_ids = global_tokens[:, :start_index + 1]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_time = time.perf_counter() - t_start
        stats = _build_stats([1], prefill_time, [], [], total_time, 1)
        return output_ids, base_model.past_key_values, stats

    # ========== STEP 1-4: Draft-Verify Loop ==========
    max_infer_steps = min(max_length, start_index + max_new_tokens)
    stop = False

    while start_index < max_infer_steps - 1 - speculative_steps:
        start_index_copy = start_index
        end_index = start_index + 1

        # ---- STEP 1: Draft token generation with early layers + adapter ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_start = time.perf_counter()
        exited_hidden_states = None

        for step in range(1 + speculative_steps):
            in_token = global_tokens[:, end_index - 1:end_index]

            # Check if we need to fill missing adapter KV cache entry
            adapter_cache_len = adapter_past_key_values[0][0].shape[2] if adapter_past_key_values else 0

            if adapter_cache_len < end_index - 1:
                # All draft tokens were accepted last round - adapter is missing
                # the KV entry for the last accepted token
                hidden_state_early_last = exited_hidden_states[:, -1:, :] if exited_hidden_states is not None else None
            else:
                hidden_state_early_last = None

            # Run early layers (draft)
            hidden_state_early = base_model.forward_draft_or_large_model(
                in_tokens_small=in_token,
            )

            if step == 0:
                exited_hidden_states = None

            exited_hidden_states = hidden_state_early if exited_hidden_states is None \
                else torch.cat([exited_hidden_states, hidden_state_early], dim=1)

            # Prepend missing entry if needed
            adapter_input = hidden_state_early
            if hidden_state_early_last is not None:
                adapter_input = torch.cat([hidden_state_early_last, hidden_state_early], dim=1)

            # Check early exit condition
            if step == speculative_steps or (step > 0 and predict_score < threshold):
                break

            # Run adapter to predict next token
            hidden_state, adapter_past_key_values = adapter_model.forward_early_stop(
                inputs_embeds=adapter_input,
                past_key_values=adapter_past_key_values,
                use_cache=True,
            )

            predict_logits = head_model(hidden_state[:, -1:, :]).float()
            predicted_token = torch.argmax(predict_logits[:, -1, :], dim=-1)
            global_tokens[:, end_index] = predicted_token

            end_index += 1
            predict_score = predict_logits.softmax(dim=-1).max().item()

        # ---- STEP 2: Verify with remaining layers ----
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_draft_end = time.perf_counter()
        draft_times.append(t_draft_end - t_draft_start)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t_verify_start = time.perf_counter()
        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        assert verify_cache_len == start_index, \
            f"Verify cache mismatch: {verify_cache_len} != {start_index}"
        assert exited_hidden_states.shape[1] == end_index - start_index, \
            f"Shape mismatch: {exited_hidden_states.shape[1]} != {end_index - start_index}"

        hidden_state_, hidden_state_normed = base_model.forward_draft_or_large_model(
            in_features_large=exited_hidden_states,
        )

        logits = head_model(hidden_state_normed).float()
        output_tokens = torch.argmax(logits, dim=-1)

        if debug_verify and len(accept_length_list) <= 5:
            round_num = len(accept_length_list)
            print(f"[DEBUG] Round {round_num}: start_index={start_index_copy}, end_index={end_index}, "
                  f"output_length={end_index - start_index_copy}, "
                  f"draft_L0_cache={base_model._get_layer_cache_length(0)}, "
                  f"verify_L{early_exit_layer}_cache={base_model._get_layer_cache_length(early_exit_layer)}, "
                  f"_seen_tokens={base_model.past_key_values._seen_tokens}, "
                  f"verify_token='{tokenizer.decode([output_tokens[0, 0].item()])}'({output_tokens[0, 0].item()}), "
                  f"logits_top3={torch.topk(logits[0, 0], 3).indices.tolist()}")

        # ---- STEP 3: Accept verified tokens ----
        output_length = end_index - start_index
        for i in range(output_length):
            is_last = (i == output_length - 1)
            is_eos = (output_tokens[0, i].item() in token_eos_set)
            is_mismatch = (not is_last and output_tokens[0, i] != global_tokens[0, start_index + 1 + i])

            if is_last or is_eos or is_mismatch:
                global_tokens[0, start_index + 1 + i] = output_tokens[0, i]
                start_index = start_index + 1 + i
                if is_eos:
                    stop = True
                break

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        verify_times.append(time.perf_counter() - t_verify_start)

        accept_length_list.append(start_index - start_index_copy)

        # ---- STEP 4: Trim KV caches to accepted position ----
        # Trim draft layers cache (layers 0 to exit_layer-1)
        draft_cache_len = base_model._get_layer_cache_length(0)
        if draft_cache_len > start_index:
            base_model.trim_draft_layers_cache(start_index)

        # Trim verify layers cache (layers exit_layer to end)
        verify_cache_len = base_model._get_layer_cache_length(early_exit_layer)
        if verify_cache_len > start_index:
            base_model.trim_verify_layers_cache(start_index)

        # Trim adapter cache
        if adapter_past_key_values and adapter_past_key_values[0][0].shape[2] > start_index:
            adapter_past_key_values = [
                (k[:, :, :start_index, :], v[:, :, :start_index, :])
                for k, v in adapter_past_key_values
            ]

        # Update seen_tokens counter
        base_model.past_key_values._seen_tokens = start_index

        if stop:
            break

    # Trim output
    output_ids = global_tokens[:, :start_index + 1]
    num_new_tokens = start_index + 1 - context_length
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    total_time = time.perf_counter() - t_start
    stats = _build_stats(accept_length_list, prefill_time, draft_times, verify_times, total_time, num_new_tokens)
    return output_ids, base_model.past_key_values, stats


def speculative_generate_for_streaming(
    model,  # KangarooQwenModel
    inputs,  # Processor output
    processor,
    past_key_values=None,
    max_new_tokens: int = 512,
    early_exit_layer: int = 2,
    speculative_steps: int = 6,
    threshold: float = 0.6,
):
    """
    Wrapper for speculative decoding in the streaming proactive_eval setting.
    Returns the generated text and updated KV cache.
    """
    output_ids, past_key_values, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=max_new_tokens,
        early_exit_layer=early_exit_layer,
        speculative_steps=speculative_steps,
        threshold=threshold,
        do_sample=False,
        past_key_values=past_key_values,
    )

    # Extract only the new tokens (exclude input)
    input_length = inputs['input_ids'].shape[1]
    new_token_ids = output_ids[:, input_length:]

    tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    reply_text = tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)[0]

    return reply_text, past_key_values, stats

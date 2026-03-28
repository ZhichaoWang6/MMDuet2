"""
Example inference script for Kangaroo self-speculative decoding.
Runs both speculative and autoregressive decoding, compares outputs and speed.

Usage:
    python inference_example.py \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter_path ./adapter_checkpoints/adapter_epoch_0 \
        --prompt "Hello, what can you do?"
"""

import argparse
import time
import torch
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import kangaroo_speculative_generate, autoregressive_generate_direct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--adapter_path', type=str, default=None)
    parser.add_argument('--exit_layer', type=int, default=2)
    parser.add_argument('--speculative_steps', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=0.6)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--prompt', type=str, default='Hello, what can you do?')
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = KangarooQwenModel(
        base_model_path=args.model_path,
        adapter_model_path=args.adapter_path,
        early_exit_layer=args.exit_layer,
        dtype=torch.bfloat16,
    )
    device = model.device
    processor = AutoProcessor.from_pretrained(args.model_path)

    # Build input
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)

    # ========== 1. Run speculative decoding ==========
    print(f"\nGenerating with speculative decoding (exit_layer={args.exit_layer}, "
          f"steps={args.speculative_steps}, threshold={args.threshold})...")

    output_ids, _, stats = kangaroo_speculative_generate(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        early_exit_layer=args.exit_layer,
        speculative_steps=args.speculative_steps,
        threshold=args.threshold,
    )

    # Decode speculative output
    spec_new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    spec_reply = processor.batch_decode(spec_new_tokens, skip_special_tokens=True)[0]
    spec_num_tokens = spec_new_tokens.shape[1]

    # ========== 2. Run autoregressive baseline (direct forward, same path as speculative) ==========
    print("Generating with autoregressive decoding (direct forward baseline)...")

    # Reset model state for clean AR run
    model.base_model.past_key_values = None
    model.reset_status()
    model.base_model.model.rope_deltas = None  # force recompute

    ar_reply, _, ar_stats = autoregressive_generate_direct(
        model=model,
        inputs=inputs,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
    )
    ar_num_tokens = ar_stats['total_tokens']
    ar_time = ar_stats['total_time']

    # ========== 3. Compare and print results ==========
    output_match = (spec_reply == ar_reply)
    length_match = (spec_num_tokens == ar_num_tokens)
    speedup = ar_time / stats['total_time'] if stats['total_time'] > 0 else 0

    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print("=" * 60)

    print(f"\n[Speculative Decoding]")
    print(f"  Reply:  {spec_reply}")
    print(f"  Tokens: {spec_num_tokens}")
    print(f"  Time:   {stats['total_time']:.3f}s (prefill={stats['prefill_time']:.3f}s, decode={stats['decode_time']:.3f}s)")
    print(f"  Tok/s:  {stats['tokens_per_second']:.1f} (decode: {stats['decode_tokens_per_second']:.1f})")
    print(f"  Avg accept length: {stats['avg_accept_length']:.2f}")
    print(f"  Total rounds:      {stats['total_rounds']}")
    print(f"  Avg draft time:    {stats['avg_draft_time']*1000:.1f}ms")
    print(f"  Avg verify time:   {stats['avg_verify_time']*1000:.1f}ms")
    print(f"  Accept lengths:    {stats['accept_lengths']}")

    print(f"\n[Autoregressive Baseline (direct forward)]")
    print(f"  Reply:  {ar_reply}")
    print(f"  Tokens: {ar_num_tokens}")
    print(f"  Time:   {ar_time:.3f}s (prefill={ar_stats['prefill_time']:.3f}s, decode={ar_stats['decode_time']:.3f}s)")
    print(f"  Tok/s:  {ar_stats['tokens_per_second']:.1f} (decode: {ar_stats['decode_tokens_per_second']:.1f})")

    print(f"\n[Comparison]")
    print(f"  Speedup:       {speedup:.2f}x")
    tag = "MATCH" if output_match else "MISMATCH"
    print(f"  Output text:   {tag}")
    tag = "MATCH" if length_match else "MISMATCH"
    print(f"  Output length: {tag} (spec={spec_num_tokens}, ar={ar_num_tokens})")
    if not output_match:
        print(f"  WARNING: Outputs differ! Greedy decoding should produce identical results.")
        print(f"    Spec: {repr(spec_reply[:300])}")
        print(f"    AR:   {repr(ar_reply[:300])}")
    print("=" * 60)


if __name__ == '__main__':
    main()

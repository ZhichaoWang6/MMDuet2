"""
Example inference script for Kangaroo self-speculative decoding.

Usage:
    python inference_example.py \
        --model_path Qwen/Qwen2.5-VL-3B-Instruct \
        --adapter_path ./adapter_checkpoints/adapter_epoch_0 \
        --prompt "Hello, what can you do?"
"""

import argparse
import torch
from transformers import AutoProcessor

from kangaroo_model import KangarooQwenModel
from inference_kangaroo import kangaroo_speculative_generate


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

    # Run speculative decoding
    print(f"Generating with speculative decoding (exit_layer={args.exit_layer}, "
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

    # Decode output
    new_tokens = output_ids[:, inputs['input_ids'].shape[1]:]
    reply = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]

    # Print results
    print("\n" + "=" * 60)
    print(f"Prompt: {args.prompt}")
    print(f"Reply: {reply}")
    print("=" * 60)
    print(f"Metrics:")
    print(f"  Total tokens:          {stats['total_tokens']}")
    print(f"  Total time:            {stats['total_time']:.3f}s")
    print(f"  Prefill time:          {stats['prefill_time']:.3f}s")
    print(f"  Decode time:           {stats['decode_time']:.3f}s")
    print(f"  Tokens/sec (total):    {stats['tokens_per_second']:.1f}")
    print(f"  Tokens/sec (decode):   {stats['decode_tokens_per_second']:.1f}")
    print(f"  Avg accept length:     {stats['avg_accept_length']:.2f}")
    print(f"  Total rounds:          {stats['total_rounds']}")
    print(f"  Avg draft time:        {stats['avg_draft_time']*1000:.1f}ms")
    print(f"  Avg verify time:       {stats['avg_verify_time']*1000:.1f}ms")
    print(f"  Accept lengths:        {stats['accept_lengths']}")


if __name__ == '__main__':
    main()

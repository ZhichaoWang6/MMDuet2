"""
Minimal test to verify that splitting model layers produces identical results.

Tests:
1. Full model forward → logits vs split (early layers → verify layers) → logits
2. Single token decode: full model vs split forward
"""

import torch
from transformers import AutoProcessor
from transformers.cache_utils import DynamicCache

from kangaroo_model import KangarooQwenModel


def test_layer_split(model_path='Qwen/Qwen2.5-VL-3B-Instruct', exit_layer=2):
    print(f"Loading model from {model_path}, exit_layer={exit_layer}...")
    model = KangarooQwenModel(
        base_model_path=model_path,
        adapter_model_path=None,
        early_exit_layer=exit_layer,
        dtype=torch.bfloat16,
    )
    device = model.device
    processor = AutoProcessor.from_pretrained(model_path)

    base_model = model.base_model  # EarlyExitQwen2_5_VLForConditionalGeneration
    full_model = base_model.model  # Qwen2_5_VLForConditionalGeneration
    qwen_model = full_model.model  # Qwen2_5_VLModel
    lm_head = model.head_model

    # Build input
    prompt = "Hello, what can you do?"
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)

    print(f"\nInput shape: {inputs['input_ids'].shape}")

    # ===== TEST 1: Prefill - full forward vs split =====
    print("\n" + "=" * 60)
    print("TEST 1: Prefill - compare full forward with hidden_states split")
    print("=" * 60)

    with torch.no_grad():
        output = full_model(
            **{k: v for k, v in inputs.items() if v is not None},
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            drop_method='none', drop_threshold=1.0, drop_absolute=True,
        )

    # Method A: full model logits (from CausalLM wrapper)
    full_logits = output.logits[:, -1, :]
    full_token = torch.argmax(full_logits, dim=-1)

    # Method B: hidden_states[-1] (after norm) → lm_head
    hs_last = output.hidden_states[-1][:, -1:, :]
    manual_logits = lm_head(hs_last).squeeze(1)
    manual_token = torch.argmax(manual_logits, dim=-1)

    # Method C: hidden_states[exit_layer] → verify layers → norm → lm_head
    hs_early = output.hidden_states[exit_layer][:, -1:, :]  # 1 token at last position
    # Run through remaining layers manually
    past_kv = output.past_key_values
    context_len = inputs['input_ids'].shape[1]
    cache_pos = torch.tensor([context_len - 1], device=device)
    rope_deltas = full_model.rope_deltas
    if rope_deltas is not None:
        delta = (cache_pos[0] + rope_deltas).to(device)
    else:
        delta = cache_pos[0]
    pos_ids = torch.tensor([[0]], device=device) + delta
    pos_ids = pos_ids.unsqueeze(0).expand(3, -1, -1)
    pe = qwen_model.rotary_emb(hs_early, pos_ids)
    attn_mask = torch.ones((1, context_len), dtype=torch.bool, device=device)
    causal_mask = qwen_model._update_causal_mask(attn_mask, hs_early, cache_pos, past_kv, False)

    h = hs_early
    for layer in qwen_model.layers[exit_layer:]:
        layer_out = layer(h, attention_mask=causal_mask, position_ids=pos_ids,
                          past_key_value=past_kv, output_attentions=False,
                          use_cache=False, cache_position=cache_pos,
                          position_embeddings=pe)
        h = layer_out[0]
    h_normed = qwen_model.norm(h)
    split_logits = lm_head(h_normed).squeeze(1)
    split_token = torch.argmax(split_logits, dim=-1)

    logit_diff_AB = (full_logits.float() - manual_logits.float()).abs().max().item()
    logit_diff_AC = (full_logits.float() - split_logits.float()).abs().max().item()
    logit_diff_BC = (manual_logits.float() - split_logits.float()).abs().max().item()

    print(f"  Full model token:    {full_token.item()} = '{processor.decode([full_token.item()])}'")
    print(f"  HS[-1]+lm_head token:{manual_token.item()} = '{processor.decode([manual_token.item()])}'")
    print(f"  Split layers token:  {split_token.item()} = '{processor.decode([split_token.item()])}'")
    print(f"  Logit max diff (full vs hs[-1]):  {logit_diff_AB}")
    print(f"  Logit max diff (full vs split):   {logit_diff_AC}")
    print(f"  Logit max diff (hs[-1] vs split): {logit_diff_BC}")
    print(f"  Token A==B: {full_token.item() == manual_token.item()}, "
          f"A==C: {full_token.item() == split_token.item()}, "
          f"B==C: {manual_token.item() == split_token.item()}")

    # ===== TEST 2: Decode step - full model generate 1 token vs earlyexit split =====
    print("\n" + "=" * 60)
    print("TEST 2: Decode - compare full model next token vs earlyexit split")
    print("=" * 60)

    # Path A: Use generate() for 1 token
    full_model.rope_deltas = None  # reset for clean generate
    with torch.no_grad():
        gen_output = full_model.generate(
            **{k: v for k, v in inputs.items() if v is not None},
            max_new_tokens=5,
            return_dict_in_generate=True,
            do_sample=False,
            drop_method='none', drop_threshold=1.0, drop_absolute=True,
        )
    ar_tokens = gen_output.sequences[0, inputs['input_ids'].shape[1]:].tolist()
    ar_text = processor.decode(ar_tokens, skip_special_tokens=True)
    print(f"  AR generate (5 tokens): {ar_tokens} = '{ar_text}'")

    # Path B: Use earlyexit split for the same tokens
    # Re-do prefill with a fresh cache
    base_model.past_key_values = None
    with torch.no_grad():
        output2 = full_model(
            **{k: v for k, v in inputs.items() if v is not None},
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
            drop_method='none', drop_threshold=1.0, drop_absolute=True,
        )
    base_model.past_key_values = output2.past_key_values

    first_token_B = torch.argmax(lm_head(output2.hidden_states[-1][:, -1:, :]), dim=-1).item()
    print(f"  Split prefill first token: {first_token_B} = '{processor.decode([first_token_B])}'")
    print(f"  AR first token:            {ar_tokens[0]} = '{processor.decode([ar_tokens[0]])}'")
    print(f"  First token match: {first_token_B == ar_tokens[0]}")

    # Now decode token by token using earlyexit
    split_tokens = [first_token_B]
    for step in range(4):
        in_token = torch.tensor([[split_tokens[-1]]], device=device)
        # Draft: layers 0 to exit_layer-1
        draft_h = base_model.forward_draft_or_large_model(in_tokens_small=in_token)
        # Verify: layers exit_layer to end
        _, verify_h_normed = base_model.forward_draft_or_large_model(in_features_large=draft_h)
        token_logits = lm_head(verify_h_normed[:, -1:, :]).float()
        next_token = torch.argmax(token_logits, dim=-1).item()
        split_tokens.append(next_token)
        print(f"  Step {step+1}: split_token={next_token}('{processor.decode([next_token])}'), "
              f"ar_token={ar_tokens[step+1] if step+1 < len(ar_tokens) else 'N/A'}"
              f"('{processor.decode([ar_tokens[step+1]])}' if step+1 < len(ar_tokens) else ''), "
              f"match={next_token == ar_tokens[step+1] if step+1 < len(ar_tokens) else 'N/A'}")

    split_text = processor.decode(split_tokens, skip_special_tokens=True)
    print(f"\n  AR tokens:    {ar_tokens}")
    print(f"  Split tokens: {split_tokens}")
    print(f"  AR text:    '{ar_text}'")
    print(f"  Split text: '{split_text}'")
    print(f"  Match: {ar_tokens[:len(split_tokens)] == split_tokens}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--exit_layer', type=int, default=2)
    args = parser.parse_args()
    test_layer_split(args.model_path, args.exit_layer)

import torch
import os
import copy
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchview import draw_graph
from dataclasses import dataclass
from torchinfo import summary




class TraceManager:
    def __init__(self, model, tokenizer):
        self.attention_weights = []   # [gen_step][layer] -> tensor[num_heads, seq, seq]
        self.attention_output = []    # [gen_step][layer] -> tensor[seq, hidden]
        self.mlp_output = []          # [gen_step][layer] -> tensor[seq, hidden]
        self.hidden_states = []        # [gen_step][layer] -> tensor[seq, hidden]

        # self.current_attn_weights = []
        self.current_attn_output = []
        self.current_mlp_output = []

        self.hooks = []
        self.model = model
        self.tokenizer = tokenizer

        self.current_prompt = None
        self.token_ids = []
        self.register_hooks(model)

    def get_attn_weights(self, layer_idx, gen_step=-1):
        return self.attention_weights[gen_step][layer_idx]
    
    def get_mlp_output(self, layer_idx, gen_step=-1):
        return self.mlp_output[gen_step][layer_idx]
    
    def get_attn_output(self, layer_idx, gen_step=-1):
        return self.attention_output[gen_step][layer_idx]
    
    def get_hidden_state(self, layer_idx, gen_step=-1):
        return self.hidden_states[gen_step][layer_idx]

    # ---- 保存attention output ----
    def _make_attn_output_hook(self, layer_id):
        def hook(module, inp, out):
            # out shape: [B, T, hidden]
            self.current_attn_output.append(out.detach().cpu())
        return hook

    # ---- 保存mlp output ----
    def _make_mlp_output_hook(self, layer_id):
        def hook(module, inp, out):
            self.current_mlp_output.append(out.detach().cpu())
        return hook

    def register_hooks(self, model):
        self.clean_all_hooks()
        for i, layer in enumerate(model.model.layers):

            # self_attn.o_proj 输出 = attention residual delta
            self.hooks.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self._make_attn_output_hook(i)
                )
            )

            # mlp.down_proj 输出 = mlp residual delta
            self.hooks.append(
                layer.mlp.down_proj.register_forward_hook(
                    self._make_mlp_output_hook(i)
                )
            )

    def reset_all(self):
        self.attention_weights.clear()
        self.attention_output.clear()
        self.mlp_output.clear()
        self.hidden_states.clear()

        self.current_attn_output.clear()
        self.current_mlp_output.clear()

    def clean_all_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def set_prompt(self, prompt):
        self.current_prompt = prompt
        self.token_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze(0).tolist()
        print(f"Prompt set to: '{prompt}' with token IDs: {self.token_ids}")
        self.reset_all()

    def step(self):
        if self.current_prompt is None:
            raise ValueError("Prompt not set. Call set_prompt() before step().")
        
        inputs = torch.tensor([self.token_ids], dtype=torch.long).to(self.model.device)
        attention_mask = torch.ones_like(inputs).to(self.model.device)
        self.current_attn_output.clear()
        self.current_mlp_output.clear()
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True
            )

        next_token_id = outputs.logits[:, -1, :].argmax(dim=-1)
        self.token_ids.append(next_token_id.item())
        next_token = self.tokenizer.decode(
            next_token_id,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # attentions/hidden_states are returned as tuples indexed by layer.
        current_attn_weights = [
            layer_attn.detach().cpu().squeeze(0)
            for layer_attn in outputs.attentions
        ]
        current_hidden_states = [
            layer_hidden.detach().cpu().squeeze(0)
            for layer_hidden in outputs.hidden_states
        ]

        current_attn_output = [
            layer_output.squeeze(0)
            for layer_output in self.current_attn_output
        ]
        current_mlp_output = [
            layer_output.squeeze(0)
            for layer_output in self.current_mlp_output
        ]

        self.attention_weights.append(current_attn_weights)
        self.attention_output.append(current_attn_output)
        self.mlp_output.append(current_mlp_output)
        self.hidden_states.append(current_hidden_states)

        self.current_prompt += next_token

        return {
            "token_id": next_token_id.detach().cpu(),
            "token": next_token,
            "prompt": self.current_prompt,
        }





# ================================
# 1. 加载模型
# ================================
if __name__ == "__main__":
    from lm_head import LMHead
    model_path = Path(__file__).parent.parent.parent / "models" / "qwen_2_5_1_5b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    head = LMHead(model, apply_final_norm=True)
    # summary(model, input_data=torch.Tensor([[1,2,3]]).to(model.device), depth=5)

    mgr = TraceManager(model, tokenizer)
    prompt = "Once upon a time"
    mgr.set_prompt(prompt)

    print('original: ', end='')
    for token_id in mgr.token_ids:
        token = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(token.ljust(10), end='|')

    step_info = mgr.step()
    print(f"Generated token: {step_info['token']} (ID: {step_info['token_id'].item()})")

    saved_hidden_states = mgr.hidden_states[-1]
    for hidden_idx, hidden_states in enumerate(saved_hidden_states):
        is_embedding = hidden_idx == 0
        is_final_norm_output = hidden_idx == len(saved_hidden_states) - 1
        if is_embedding:
            label = "Embed   "
        elif is_final_norm_output:
            label = "Final   "
        else:
            label = f"Layer {hidden_idx - 1:<2}"

        logits = head.decode(hidden_states, apply_final_norm=not is_final_norm_output)
        tokens = logits.argmax(dim=-1)
        print(f"\n{label}: ", end='')
        decoded_tokens = [tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False) for token_id in tokens]
        for token in decoded_tokens:
            print(token.ljust(10), end='|')


    


        



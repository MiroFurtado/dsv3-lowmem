import os
import json
import time
from argparse import ArgumentParser
from typing import List, Union, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from datetime import datetime
import pytz

# -----------
# MODEL CODE
# -----------

class ModelArgs:
    def __init__(
        self,
        max_seq_len: int = 2048,
        max_batch_size: int = 1,
        # ... add any other fields needed ...
        **kwargs
    ):
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        # store anything extra
        for k, v in kwargs.items():
            setattr(self, k, v)

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        self.scale = 1.0

    def forward(self, x: torch.Tensor):
        # naive linear
        return torch.nn.functional.linear(x, self.weight * self.scale, self.bias)

class Transformer(torch.nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Minimal stand-in for demonstration; replace with real architecture
        self.args = args
        self.linear = Linear(768, 50257)  # example
        self.max_seq_len = args.max_seq_len

    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # Minimal forward
        # tokens shape: [batch_size, current_length]
        x = torch.randn(tokens.shape[0], tokens.shape[1], 768, device=tokens.device)
        logits = self.linear(x)  # shape: [batch_size, current_length, vocab_size]
        # Return the last token's logits (like a standard causal model)
        return logits[:, -1, :]  # shape: [batch_size, vocab_size]

# -----------
# HELPER FUNCTIONS
# -----------

def my_load_model(model: torch.nn.Module, filename: Union[str, os.PathLike]):
    """
    Loads a safetensors weight file into the model.
    """
    filename = str(filename)
    import safetensors.torch
    print(f"loading {filename}")
    sd = safetensors.torch.load_file(filename, device="cpu")

    # Transfer to GPU
    total = torch.tensor(0.0, device='cpu')
    for k in list(sd.keys()):
        if '.experts.' not in k:
            sd[k] = sd[k].to('cuda')
        # Force load
        total += sd[k].view(-1)[0].float().cpu()

    # Load into model
    model.load_state_dict(sd, strict=False, assign=True)

# Simple temperature-based sampler
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

# For consistent timestamp prints
pacific_tz = pytz.timezone('America/Los_Angeles')
def stamp():
    t = datetime.now(pacific_tz)
    return t.strftime("%I:%M:%S %p")

# -----------
# GLOBALS
# -----------

# We’ll store the loaded model/tokenizer here for FastAPI
model_instance = None
tokenizer_instance = None
args_instance = None

# Basic printing wrappers
realprint = print
def print0(*args, rank=0, **kwargs):
    """
    Only print if rank=0
    """
    if rank == 0:
        realprint(f"{stamp()} [gpu_{rank}]", *args, **kwargs)

# -----------
# MODEL INITIALIZATION
# -----------

def initialize_model(ckpt_path: str, config: str):
    """
    Loads model + tokenizer into global variables if not already done.
    """
    global model_instance, tokenizer_instance, args_instance

    if model_instance is not None:
        # Already initialized
        return

    # Dist info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Basic setup
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)

    # Read config
    with open(config) as f:
        args_instance = ModelArgs(**json.load(f))

    # Build model
    with torch.device("cuda"):
        model_instance = Transformer(args_instance)

    # Load weights (sharded by rank)
    weight_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    my_load_model(model_instance, weight_path)

    # Fix linear scales (example usage)
    for module in model_instance.modules():
        if isinstance(module, Linear):
            module.weight.scale = module.scale

    # Load tokenizer
    tokenizer_instance = AutoTokenizer.from_pretrained(ckpt_path)

    # Optional quick test to confirm everything works
    # (Comment out if you don't want the extra overhead)
    _ = tokenizer_instance.decode(
        generate(
            model_instance,
            [tokenizer_instance.encode("Test prompt.")],
            max_new_tokens=2,
            eos_id=-1,
            temperature=0.1
        )[0],
        skip_special_tokens=True
    )
    print0("Model & tokenizer initialized successfully!", rank=rank)

# -----------
# GENERATION
# -----------

@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0,
    tokenizer=None
) -> List[List[int]]:
    """
    Generates new tokens from given prompts.
    """
    rank = int(os.getenv("RANK", "0"))
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))

    # Setup tokens array
    tokens = torch.full(
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda"
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    started_at = time.time()
    num_tokens_generated = 0
    prev_pos = 0

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)

        # If there's a prompt token at cur_pos, preserve it
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)

        tokens[:, cur_pos] = next_token

        # Optional debug print for rank=0
        if tokenizer and rank == 0:
            out_str = tokenizer.decode(tokens[0, cur_pos:cur_pos+1].tolist(), skip_special_tokens=True)
            realprint(out_str, end='', flush=True)

        num_tokens_generated += 1
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos

        if finished.all():
            break

    elapsed = time.time() - started_at
    print0(f"\nGenerated {num_tokens_generated} tokens in {elapsed:.2f}s "
           f"({num_tokens_generated / elapsed:.1f} tok/sec)\n", rank=rank)

    # Gather completions
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens

# -----------
# FASTAPI
# -----------

app = FastAPI()

class ChatCompletionRequest(BaseModel):
    messages: List[dict]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    """
    Chat endpoint for generation.
    """
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not initialized.")

    # Construct the prompt
    prompt_tokens = tokenizer_instance.apply_chat_template(
        request.messages, add_generation_prompt=True
    )

    # Generate
    completion_tokens = generate(
        model_instance,
        [prompt_tokens],
        request.max_tokens,
        tokenizer_instance.eos_token_id,
        request.temperature,
        tokenizer=tokenizer_instance
    )

    # Decode
    completion = tokenizer_instance.decode(completion_tokens[0], skip_special_tokens=True)

    return {
        "id": "cmpl-12345",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completion_tokens[0]),
            "total_tokens": len(prompt_tokens) + len(completion_tokens[0])
        }
    }

# -----------
# MAIN
# -----------

def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    api: bool = False,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main entry for local or distributed usage.
    """
    # Dist info
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Initialize + load model into globals
    initialize_model(ckpt_path, config)

    # If user wants an API, typically only start on rank=0
    if api and rank == 0:
        print0("Starting API server at http://0.0.0.0:8000", rank=rank)
        uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    # If user wants interactive mode
    if interactive:
        messages = []
        counter = -1
        while True:
            counter += 1

            def user_input():
                if counter == 0:
                    return "WARM ME UP. TELL ME A LONG STORY. LET'S GET WARM."
                return input(">>> ")

            # If distributed, broadcast the prompt from rank 0 to others
            if world_size == 1:
                prompt = user_input()
            else:
                if rank == 0:
                    prompt = user_input()
                    dist.broadcast_object_list([prompt], src=0)
                else:
                    holder = [None]
                    dist.broadcast_object_list(holder, src=0)
                    prompt = holder[0]

            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            messages.append({"role": "user", "content": prompt})

            # Build chat prompt
            prompt_tokens = tokenizer_instance.apply_chat_template(messages, add_generation_prompt=True)

            # Generate
            completion_tokens = generate(
                model_instance,
                [prompt_tokens],
                max_new_tokens,
                tokenizer_instance.eos_token_id,
                temperature,
                tokenizer=tokenizer_instance
            )
            completion = tokenizer_instance.decode(completion_tokens[0], skip_special_tokens=True)

            print0("\n" + completion + "\n", rank=rank)
            messages.append({"role": "assistant", "content": completion})

            # Clear messages for next round (comment out if you want multi-turn history)
            messages.clear()

    # If user wants file-based prompts
    elif input_file:
        with open(input_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

        # Generate for each prompt
        prompt_tokens_list = [
            tokenizer_instance.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True
            ) for p in prompts
        ]
        outs = generate(
            model_instance,
            prompt_tokens_list,
            max_new_tokens,
            tokenizer_instance.eos_token_id,
            temperature
        )
        for prompt, tokens_out in zip(prompts, outs):
            completion = tokenizer_instance.decode(tokens_out, skip_special_tokens=True)
            print0(f"Prompt: {prompt}\nCompletion: {completion}\n", rank=rank)

    # Done
    print0("Exiting main.", rank=rank)


def run_main():
    """
    Entry point when launched via 'python' or 'torchrun'.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--api", action="store_true", help="Start as an API server")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    # We require at least one mode
    assert args.input_file or args.interactive or args.api, (
        "Must specify at least one mode: --input-file, --interactive, or --api"
    )

    # Possibly init the process group for multi-GPU
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")

    try:
        # Pass all args with names so ordering is correct
        main(
            ckpt_path=args.ckpt_path,
            config=args.config,
            input_file=args.input_file,
            interactive=args.interactive,
            api=args.api,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    finally:
        # Destroy process group if needed
        if world_size > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    run_main()

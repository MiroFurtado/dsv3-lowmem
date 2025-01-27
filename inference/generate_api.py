import os
import json
import time
from argparse import ArgumentParser
from typing import List, Union, Optional

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# ------------------------------
# FASTAPI IMPORTS
# ------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ------------------------------
# YOUR IMPORTS / DEFINITIONS
# ------------------------------
from model import Linear, Transformer, ModelArgs

realprint = print
print0 = realprint
torch.set_num_threads(32)

rank = int(os.getenv("RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))

# We will store global references to the model & tokenizer
model = None
tokenizer = None

# ------------------------------
# SAMPLING & GENERATION (unchanged)
# ------------------------------
def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)

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
    Generates new tokens based on the given prompt tokens using the specified model.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full(
        (len(prompt_tokens), total_len), -1, dtype=torch.long, device="cuda"
    )
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    finished = torch.tensor([False] * len(prompt_tokens), device="cuda")
    prompt_mask = tokens != -1

    started_at = time.time()
    numdid = 0
    prev_pos = 0

    for cur_pos in range(min(prompt_lens), total_len):
        logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # If it's still part of the original prompt, keep it
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token

        # Print incremental token if rank=0
        if tokenizer and rank == 0:
            string = tokenizer.decode(tokens[0, cur_pos:cur_pos+1].tolist(), skip_special_tokens=True)
            realprint(string, flush=True, end='')

        numdid += 1
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break

    elapsed = time.time() - started_at
    print0(f"\nDid {numdid} tokens in {elapsed:.2f} seconds "
           f"({numdid / elapsed:.1f} tok/sec)\n")

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i]:prompt_lens[i]+max_new_tokens]
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens

# ------------------------------
# UTILITY & LOADING (unchanged)
# ------------------------------
def my_load_model(model: torch.nn.Module, filename: Union[str, os.PathLike]):
    import safetensors.torch
    filename = str(filename)
    print0(f"loading {filename}")
    sd = safetensors.torch.load_file(filename, device="cpu")
    total = torch.tensor(0., device='cpu')
    for k in list(sd.keys()):
        if '.experts.' not in k:
            sd[k] = sd[k].to('cuda')
        total += sd[k].view(-1)[0].float().cpu()
    model.load_state_dict(sd, strict=False, assign=True)

from datetime import datetime
import pytz
pacific_tz = pytz.timezone('America/Los_Angeles')
def stamp():
    t = datetime.now(pacific_tz)
    return t.strftime("%I:%M:%S %p")

def print(*args, **kwargs):
    """
    Override built-in print so we prefix [gpu_rank].
    """
    realprint(f"{stamp()} [gpu_{rank}]", *args, **kwargs)

# Only prints if rank==0
def print0(*args, **kwargs):
    if rank == 0:
        realprint(f"{stamp()} [gpu_{rank}]", *args, **kwargs)

# ------------------------------
# FASTAPI SETUP
# ------------------------------
app = FastAPI()

class ChatCompletionRequest(BaseModel):
    messages: List[dict]                  # a list of {"role":..., "content":...}
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0

@app.post("/v1/chat/completions")
def chat_completion(request: ChatCompletionRequest):
    """
    A minimalistic OpenAI-like chat endpoint:
      POST /v1/chat/completions
      {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 100,
        "temperature": 0.7
      }
    Returns JSON with the model's reply.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # Build a single chat prompt using your tokenizer's "apply_chat_template()"
    prompt_tokens = tokenizer.apply_chat_template(request.messages, add_generation_prompt=True)

    # Generate
    completion_tokens = generate(
        model,
        [prompt_tokens],
        request.max_tokens,
        tokenizer.eos_token_id,
        request.temperature,
        tokenizer=tokenizer
    )

    # Decode
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)

    # Return an OpenAI-like schema
    return {
        "id": "chatcmpl-12345",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "my-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": completion
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": len(completion_tokens[0]),
            "total_tokens": len(prompt_tokens) + len(completion_tokens[0])
        }
    }

# ------------------------------
# MAIN FUNCTION
# ------------------------------
def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    api: bool = False,
) -> None:
    """
    Main function to load the model, then do:
    - interactive mode
    - or batch mode if input_file is given
    - or run API if --api is given
    """
    global model, tokenizer

    if world_size > 1:
        dist.init_process_group("nccl")

    # GPU settings
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(965)

    # Load config
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print0(args)

    # Build model
    print0("making model")
    with torch.device("cuda"):
        model = Transformer(args)
    print0("loading model")

    # Load weights
    weight_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    my_load_model(model, weight_path)

    # Fix linear scales, if relevant
    for module in model.modules():
        if isinstance(module, Linear):
            module.weight.scale = module.scale

    print0("firing up tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

    # Just a quick test decode (optional)
    _ = tokenizer.decode(generate(model, [tokenizer.encode("DeepSeek")], 2, -1, 1.0)[0])

    # If user wants an API server, start it on rank=0
    if api:
        if rank == 0:
            print0("Starting API server at http://0.0.0.0:8000")
            uvicorn.run(app, host="0.0.0.0", port=8000)
        return

    # Otherwise, do your existing interactive or batch mode
    if interactive:
        messages = []
        counter = -1
        while True:
            counter += 1
            def inpoot():
                if counter == 0:
                    return "WARM ME UP. TELL ME A LONG STORY. LET'S GET WARM."
                return input(">>> ")

            if world_size == 1:
                prompt = inpoot()
            else:
                if rank == 0:
                    prompt = inpoot()
                    objects = [prompt]
                    dist.broadcast_object_list(objects, 0)
                else:
                    objects = [None]
                    dist.broadcast_object_list(objects, 0)
                    prompt = objects[0]

            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue

            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature, tokenizer=tokenizer)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print0('\n' + completion + '\n')
            messages.append({"role": "assistant", "content": completion})

            messages.clear()  # comment out for multi-turn history
    else:
        # batch mode
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines() if line.strip()]
        assert len(prompts) <= args.max_batch_size
        prompt_tokens_list = [
            tokenizer.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True)
            for p in prompts
        ]
        completion_tokens_list = generate(
            model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, temperature
        )
        completions = tokenizer.batch_decode(completion_tokens_list, skip_special_tokens=True)
        for prompt, completion in zip(prompts, completions):
            print0(f"Prompt: {prompt}")
            print0(f"Completion: {completion}\n")

    if world_size > 1:
        dist.destroy_process_group()

# ------------------------------
# CLI ENTRY POINT
# ------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--api", action="store_true",
        help="Run a FastAPI server at 0.0.0.0:8000 instead of interactive/file modes.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    # Must specify at least one mode
    if not (args.input_file or args.interactive or args.api):
        raise ValueError("Must specify --input-file, --interactive, or --api.")

    main(
        ckpt_path=args.ckpt_path,
        config=args.config,
        input_file=args.input_file,
        interactive=args.interactive,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        api=args.api,
    )
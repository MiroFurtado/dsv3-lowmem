import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp, these_mps):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.

    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    state_dicts = [{} for _ in range(mp)]

    files = glob(os.path.join(hf_ckpt_path, "*.safetensors"))
    print(f"Found {len(files)} safetensors files to process")
    
    for file_path in tqdm(files, desc="Processing files"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for name in tqdm(keys, desc=f"Processing {os.path.basename(file_path)}", leave=False):
                if "model.layers.61" in name:
                    continue
                param: torch.Tensor = f.get_tensor(name)
                if name.startswith("model."):
                    name = name[len("model."):]
                name = name.replace("self_attn", "attn")
                name = name.replace("mlp", "ffn")
                name = name.replace("weight_scale_inv", "scale")
                name = name.replace("e_score_correction_bias", "bias")
                key = name.split(".")[-2]
                assert key in mapping
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                for i in these_mps:
                    new_param = param
                    if "experts" in name and "shared_experts" not in name:
                        idx = int(name.split(".")[-3])
                        if idx < i * n_local_experts or idx >= (i + 1) * n_local_experts:
                            continue
                    elif dim is not None:
                        assert param.size(dim) % mp == 0
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    state_dicts[i][name] = new_param

    os.makedirs(save_path, exist_ok=True)

    for i in these_mps:
        p = os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
        if os.path.exists(p):
            print(f"{p=}: already exists, skipping")
            continue
        print(f"Saving shard {i} to {p}")
        with tqdm(total=len(state_dicts[i]), desc=f"Saving shard {i}") as pbar:
            save_file(state_dicts[i], p)
            pbar.update(len(state_dicts[i]))
        print(f"Successfully saved shard {i}")

    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    # parser.add_argument("--which-mp", type=int, required=True)
    parser.add_argument("--which-mp", type=int, required=True, help="Which model parallel shard to generate (0-based index)")
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
assert 0 <= args.which_mp < args.model_parallel, f"which-mp must be between 0 and {args.model_parallel-1}"

    print(f"Generating shard {args.which_mp} of {args.model_parallel}")
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel, [args.which_mps])

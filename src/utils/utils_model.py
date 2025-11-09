import torch
from collections import OrderedDict

def _load_state_dict_from_ckpt(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    # Handle both {'state_dict': ...} and plain dict formats
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], (dict, OrderedDict)):
        return state["state_dict"]
    elif isinstance(state, (dict, OrderedDict)):
        return state
    else:
        raise ValueError("Checkpoint format not recognized: expected dict or {'state_dict': dict}.")

def compare_ckpt_vs_model(ckpt_path, model, topk=50, show_examples=True):
    """
    Compare checkpoint and model state dicts without modifying or filtering anything.

    Produces and prints four sets:
      - only_in_ckpt:    keys that exist only in the checkpoint
      - only_in_model:   keys that exist only in the model
      - shape_mismatch:  keys that exist in both but have different tensor shapes
      - dtype_mismatch:  keys that exist in both, have the same shape, but different data types
    """
    ckpt_sd = _load_state_dict_from_ckpt(ckpt_path)
    model_sd = model.state_dict()

    ckpt_keys  = list(ckpt_sd.keys())
    model_keys = list(model_sd.keys())

    ckpt_key_set  = set(ckpt_keys)
    model_key_set = set(model_keys)

    only_in_ckpt  = sorted(list(ckpt_key_set - model_key_set))
    only_in_model = sorted(list(model_key_set - ckpt_key_set))

    shape_mismatch = []
    dtype_mismatch = []

    shared = ckpt_key_set & model_key_set
    for k in sorted(shared):
        v_ckpt  = ckpt_sd[k]
        v_model = model_sd[k]
        # Support both tensors and scalars (e.g., 0-D items like temperature)
        shape_ckpt  = getattr(v_ckpt, "shape", None)
        shape_model = getattr(v_model, "shape", None)
        if (shape_ckpt is not None) and (shape_model is not None):
            if tuple(shape_ckpt) != tuple(shape_model):
                shape_mismatch.append((k, tuple(shape_ckpt), tuple(shape_model)))
                continue
        # If shape matches, check dtype
        dtype_ckpt  = getattr(v_ckpt, "dtype", None)
        dtype_model = getattr(v_model, "dtype", None)
        if (dtype_ckpt is not None) and (dtype_model is not None):
            if dtype_ckpt != dtype_model:
                dtype_mismatch.append((k, str(dtype_ckpt), str(dtype_model)))

    # —— Summary print —— #
    print("=== CKPT vs MODEL DIFF REPORT ===")
    print(f"ckpt path: {ckpt_path}")
    print(f"total ckpt keys : {len(ckpt_keys)}")
    print(f"total model keys: {len(model_keys)}\n")

    print(f"only_in_ckpt   : {len(only_in_ckpt)}")
    print(f"only_in_model  : {len(only_in_model)}")
    print(f"shape_mismatch : {len(shape_mismatch)}")
    print(f"dtype_mismatch : {len(dtype_mismatch)}\n")

    if show_examples:
        def _head(lst, n=topk):
            return lst[:n]

        if only_in_ckpt:
            print(f"-- only_in_ckpt (showing up to {topk}) --")
            for k in _head(only_in_ckpt):
                print("  ", k)
            if len(only_in_ckpt) > topk: print("  ...")
            print()

        if only_in_model:
            print(f"-- only_in_model (showing up to {topk}) --")
            for k in _head(only_in_model):
                print("  ", k)
            if len(only_in_model) > topk: print("  ...")
            print()

        if shape_mismatch:
            print(f"-- shape_mismatch (showing up to {topk}) --")
            for k, s_ckpt, s_model in _head(shape_mismatch):
                print(f"  {k}: {s_ckpt}  vs  {s_model}")
            if len(shape_mismatch) > topk: print("  ...")
            print()

        if dtype_mismatch:
            print(f"-- dtype_mismatch (showing up to {topk}) --")
            for k, d_ckpt, d_model in _head(dtype_mismatch):
                print(f"  {k}: {d_ckpt}  vs  {d_model}")
            if len(dtype_mismatch) > topk: print("  ...")
            print()

    # Also return the results for programmatic use or saving
    return {
        "only_in_ckpt": only_in_ckpt,
        "only_in_model": only_in_model,
        "shape_mismatch": shape_mismatch,
        "dtype_mismatch": dtype_mismatch,
    }


def safe_load_ctclip(model, ckpt_path, verbose=True):
    """
    1️⃣ Compare the checkpoint and model
    2️⃣ Automatically remove redundant 'position_ids' keys found only in the checkpoint
    3️⃣ Load safely with strict=False
    """
    # ---- Step 1: Comparison ----
    report = compare_ckpt_vs_model(ckpt_path, model, topk=10, show_examples=False)
    if verbose:
        print(f"Diff summary: { {k: len(v) for k,v in report.items()} }")

    # ---- Step 2: Load state dict ----
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get("state_dict", state)
    sd = OrderedDict(sd)

    # Remove redundant keys (only 'position_ids')
    drop_keys = [k for k in sd.keys() if "position_ids" in k]
    for k in drop_keys:
        sd.pop(k)
        if verbose:
            print(f"[drop] {k}")

    # ---- Step 3: Safe loading ----
    msg = model.load_state_dict(sd, strict=False)
    if verbose:
        print("Load complete:")
        print("  missing_keys:", len(msg.missing_keys))
        print("  unexpected_keys:", len(msg.unexpected_keys))
    return msg

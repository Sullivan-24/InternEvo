"""
utils package
"""
from typing import Dict, Tuple
from z3 import If
from .z3_config import *
# def parse_microbatch_key(key: str) -> Tuple[bool, int, int]:
#     "parse microbatch key"
#     is_forward = key.startswith("f")
#     mid, pid = key.split("_")[1:]

#     return is_forward, int(pid), int(mid)

def parse_microbatch_key(key: str) -> Tuple[bool, int, int]:
    "parse microbatch key"
    cwi, stream_idx, k, mid, pid = key.split("_")

    return cwi, int(stream_idx), k, int(pid), int(mid)

# def _replace_mid_in_key(key: str, new_mid: int) -> str:
#     is_forward, pid, _ = parse_microbatch_key(key)

#     return f"{'f' if is_forward else 'b'}_{new_mid}_{pid}"
def _replace_mid_in_key(key: str, new_mid: int) -> str:
    cwi, stream_idx, k, pid, _ = parse_microbatch_key(key)

    return f"{cwi}_{stream_idx}_{k}_{new_mid}_{pid}"

def resort_microbatch_index(num_microbatches: int, model_res: Dict[str, int]) -> dict:
    "resort microbatch index"
    # max_value = -1
    # for k in model_res:
    #     print(k, model_res[k], type(model_res[k]))
    #     max_value = max_value if max_value > model_res[k] else model_res[k]
    max_value = max(model_res.values())
    only_forward_starts = {mb: max_value for mb in range(num_microbatches)}

    for key, value in model_res.items():
        # if key.startswith("b_"):
        #     continue
        if not key[6:].startswith("f_"):
            continue
        _, _, _, _, mid = parse_microbatch_key(key)

        if only_forward_starts[mid] > value:
            only_forward_starts[mid] = value

    sorted_forward_starts = sorted(only_forward_starts.items(), key=lambda x: x[1])
    sorted_indexes = {
        pair[0]: new_idx for new_idx, pair in enumerate(sorted_forward_starts)
    }

    res = {
        _replace_mid_in_key(key, sorted_indexes[parse_microbatch_key(key)[-1]]): value
        for key, value in model_res.items()
    }

    return res

def print_to_file(filename: str, content: str) -> None:
    if filename:
        with open(filename, "a") as f:
            f.write(content)
            f.flush()
    print(content, end="")
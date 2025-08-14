import json
from glob import glob
from copy import deepcopy
import itertools
from typing import Callable
import sys

schema = {
    "$schema": "http://json-schema.org/schema#",
    "type": "object",
    "properties": {
        "schemaVersion": {"type": "integer"},
        "deviceProperties": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "totalGlobalMem": {"type": "integer"},
                    "computeMajor": {"type": "integer"},
                    "computeMinor": {"type": "integer"},
                    "maxThreadsPerBlock": {"type": "integer"},
                    "maxThreadsPerMultiprocessor": {"type": "integer"},
                    "regsPerBlock": {"type": "integer"},
                    "regsPerMultiprocessor": {"type": "integer"},
                    "warpSize": {"type": "integer"},
                    "sharedMemPerBlock": {"type": "integer"},
                    "sharedMemPerMultiprocessor": {"type": "integer"},
                    "numSms": {"type": "integer"},
                    "sharedMemPerBlockOptin": {"type": "integer"},
                },
                "required": [
                    "computeMajor",
                    "computeMinor",
                    "id",
                    "maxThreadsPerBlock",
                    "maxThreadsPerMultiprocessor",
                    "name",
                    "numSms",
                    "regsPerBlock",
                    "regsPerMultiprocessor",
                    "sharedMemPerBlock",
                    "sharedMemPerBlockOptin",
                    "sharedMemPerMultiprocessor",
                    "totalGlobalMem",
                    "warpSize",
                ],
            },
        },
        "distributedInfo": {
            "type": "object",
            "properties": {
                "backend": {"type": "string"},
                "rank": {"type": "integer"},
                "world_size": {"type": "integer"},
            },
            "required": ["backend", "rank", "world_size"],
        },
        "traceEvents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "ph": {"type": "string"},
                    "cat": {"type": "string"},
                    "name": {"type": "string"},
                    "pid": {"type": ["integer", "string"]},
                    "tid": {"type": ["integer", "string"]},
                    "ts": {"type": "integer"},
                    "dur": {"type": "integer"},
                    "args": {
                        "type": "object",
                        "properties": {
                            "External id": {"type": "integer"},
                            "Ev Idx": {"type": "integer"},
                            "Fwd thread id": {"type": "integer"},
                            "Sequence number": {"type": "integer"},
                            "queued": {"type": "integer"},
                            "device": {"type": "integer"},
                            "context": {"type": "integer"},
                            "stream": {"type": "integer"},
                            "correlation": {"type": "integer"},
                            "registers per thread": {"type": "integer"},
                            "shared memory": {"type": "integer"},
                            "blocks per SM": {"type": "number"},
                            "warps per SM": {"type": "number"},
                            "grid": {"type": "array", "items": {"type": "integer"}},
                            "block": {"type": "array", "items": {"type": "integer"}},
                            "est. achieved occupancy %": {"type": "integer"},
                            "cbid": {"type": "integer"},
                            "bytes": {"type": "integer"},
                            "memory bandwidth (GB/s)": {"type": ["number", "string"]},
                            "name": {"type": "string"},
                            "labels": {"type": "string"},
                            "sort_index": {"type": "integer"},
                            "Op count": {"type": "integer"},
                        },
                    },
                    "id": {"type": "integer"},
                    "bp": {"type": "string"},
                    "s": {"type": "string"},
                },
                "required": ["name", "ph", "pid", "tid", "ts"],
            },
        },
        "traceName": {"type": "string"},
    },
    "required": [
        "deviceProperties",
        "distributedInfo",
        "schemaVersion",
        "traceEvents",
        "traceName",
    ],
}

# trace = json.load(open('/home/ubuntu/tsinghua/zly/mini_trace.json'))
# df = pd.DataFrame(data=trace['traceEvents'], columns=['ph', 'cat', 'name', 'pid', 'tid', 'ts', 'dur'])
# print(df)
# with torch.profiler.record_function(f"SCH-forward_step-{scheduled_node.minibatch}-0"):
def get_annotations(
    events, prefix="SCH-", pid=None, rename: Callable[[str], str] = None
):
    ret = []
    for e in events:
        if e.get("cat", None) == "user_annotation" and e["name"].startswith(prefix):
            # SCH-RECV_FORWARD-0-True
            v = deepcopy(e)
            # v['cname']='black' # TODO
            if rename is not None:
                v["name"] = rename(v["name"])
            ret.append(v)
            if pid is not None:
                v["pid"] = 0
                v["tid"] = pid
            # print(v)
    return ret


def strip_name(s: str):
    types = {
        "RECV_FORWARD": "recv_F",
        "RECV_BACKWARD": "recv_B",
        "SEND_FORWARD": "send_F",
        "SEND_BACKWARD": "send_B",
        "F": "F",
        "B": "B",
        "W": "W",
    }
    op = s.split("-")[1]
    return "-".join([types.get(op, op)] + s.split("-")[2:])


def main(path_prefix, rank_start, rank_end, rank_step):
    merged_trace = None
    for rank in range(rank_start, rank_end, rank_step):
        fns = glob(path_prefix + f"rank{rank}_dp0_wp0_tp0/*.json")
        assert len(fns) == 1
        fn = fns[0]
        trace = json.load(open(fn))
        events = get_annotations(trace["traceEvents"], pid=rank, rename=strip_name)
        if merged_trace is None:
            merged_trace = trace
            merged_trace["traceEvents"] = []
        merged_trace["traceEvents"] += events
        print(f'Imported {len(merged_trace["traceEvents"])} events from {fn}')
    json.dump(merged_trace, open("out.json", "w"))
    print("Generated merged trace ./out.json")


if __name__ == "__main__":
    start, end, step = 0, 7, 1
    # if len(sys.argv) > 1:
    #     path_prefix = sys.argv[1]
    #     # f"/data0/tsinghua/huangkz/logs/trace_ite_4-0801-073644-rank_{rank}/*.pt.trace.json"
    # else:
    #     print(
    #         f"Usage: python {sys.argv[0]} <path_prefix> [rank_start={start}] [end={end}] [step={step}]"
    #     )
    #     print(
    #         f"Example: python /data0/tsinghua/huangkz/logs/trace_ite_4-0801-073644-rank_"
    #     )
    #     exit()
    # if len(sys.argv) > 2:
    #     start = int(sys.argv[2])
    # if len(sys.argv) > 3:
    #     end = int(sys.argv[3])
    # if len(sys.argv) > 4:
    #     step = int(sys.argv[4])
    path_prefix = "/cpfs01/user/matenghui/RUN/14b_llama2_train/08-13-19.00.15/traces/"
    main(path_prefix, start, end, step)

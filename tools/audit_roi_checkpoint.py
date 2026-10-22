"""CPU structural audit of a newly saved ROI DCP and its HF export."""
import json
from pathlib import Path
import pickle
import struct


def audit(checkpoint, step, roi_sha):
    import torch
    cp = Path(checkpoint)
    result = dict(checkpoint=str(cp), step=step, dcp={}, ranks=[], errors=[])
    for name in ("model", "optimizer"):
        folder = cp / name
        with (folder / ".metadata").open("rb") as stream:
            metadata = pickle.load(stream)
        extents = {}
        for item in metadata.storage_data.values():
            extents[item.relative_path] = max(extents.get(item.relative_path, 0), item.offset + item.length)
        missing = [name for name, end in extents.items()
                   if not (folder / name).is_file() or (folder / name).stat().st_size < end]
        result["dcp"][name] = dict(records=len(metadata.storage_data), files=len(extents), missing=missing)
        result["errors"].extend(missing)
    paths = sorted((cp / "extra_state").glob("extra_state_rank_*.pt"))
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        rng = state.get("rng_state", {})
        valid = (state.get("global_step") == step
                 and state.get("lr_scheduler", {}).get("last_epoch") == step
                 and bool(state.get("train_dataloader"))
                 and {"torch", "cuda", "numpy", "python"}.issubset(rng)
                 and rng.get("cuda") is not None)
        result["ranks"].append(dict(file=path.name, valid=valid))
        if not valid:
            result["errors"].append(path.name)
    if len(paths) != 8:
        result["errors"].append("expected_eight_ranks")
    hf = cp / "hf_ckpt"
    weight_map = json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"]
    keys = set()
    for filename in sorted(set(weight_map.values())):
        path = hf / filename
        with path.open("rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(length))
        for key, value in header.items():
            if key == "__metadata__":
                continue
            keys.add(key)
            start, end = value["data_offsets"]
            if not 0 <= start <= end <= path.stat().st_size - length - 8 or weight_map.get(key) != filename:
                result["errors"].append("invalid_tensor:" + key)
    if keys != set(weight_map):
        result["errors"].append("hf_index_mismatch")
    if json.loads((hf / "config.json").read_text()).get("roi_config_sha256") != roi_sha:
        result["errors"].append("roi_identity_mismatch")
    result.update(hf_tensors=len(keys), hf_shards=len(set(weight_map.values())),
                  complete=not result["errors"])
    return result

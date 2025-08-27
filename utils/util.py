from easydict import EasyDict
from pathlib import Path

def get_sampling_dir(config: EasyDict) -> Path:
    parts = [
        config.data,
        config.solver,
        config.algorithm_type,
        config.skip_type,
        f"FS{config.flow_shift}",
        f"NFE{config.NFE}",
        f"CFG{config.CFG}",
        f"ORDER{config.order}",
    ]
    name = config.model + "".join(f"({p})" for p in parts)
    return Path(config.save_root) / name

from pathlib import Path
import re
from tensorboard.backend.event_processing import event_accumulator as ea

def get_pt(pt_dir: str, label: str):
    if label == 'latest':
        return get_latest_pt(pt_dir)

    acc = ea.EventAccumulator(pt_dir, size_guidance={ea.SCALARS: 0}); acc.Reload()
    vals = {e.step: e.value for e in acc.Scalars(label)}  # step -> scalar

    files = {int(m.group(1)): str(p)
             for p in Path(pt_dir).glob("step_*.pt")
             for m in [re.search(r"step_(\d+)\.pt$", p.name)] if m}

    cands = [(vals[s], s, f) for s, f in files.items() if s in vals]
    return min(cands)[2] if cands else None    

def get_latest_pt(pt_dir: str):
    files = {
        int(m.group(1)): str(p)
        for p in Path(pt_dir).glob("step_*.pt")
        for m in [re.search(r"step_(\d+)\.pt$", p.name)]
        if m
    }
    return files[max(files)] if files else None
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

def _collect_ckpts(pt_dir: str):
    return {
        int(m.group(1)): str(p)
        for p in Path(pt_dir).glob("step_*.pt")
        for m in [re.search(r"step_(\d+)\.pt$", p.name)]
        if m
    }

def _choose_by_step(files: dict[int, str], target: int) -> str | None:
    if not files:
        return None
    if target in files:
        return files[target]
    lower_eq = [s for s in files if s <= target]
    if lower_eq:
        return files[max(lower_eq)]
    # fallback: absolute closest
    closest = min(files, key=lambda s: abs(s - target))
    return files[closest]

def get_latest_pt(pt_dir):
    return get_pt(pt_dir, label='latest')    

def get_pt(pt_dir: str, label: str | None = None, step: int | None = None) -> str | None:
    files = _collect_ckpts(pt_dir)
    if not files:
        return None

    # 1) step이 주어지면 최우선 적용
    if step is not None:
        return _choose_by_step(files, int(step))

    # 2) latest 처리
    if label == 'latest':
        return files[max(files)]  # get_latest_pt와 동일

    # 3) label이 숫자 문자열이면 step처럼 동작(편의)
    if isinstance(label, str) and label.isdigit():
        return _choose_by_step(files, int(label))

    # 4) 텐서보드 스칼라 태그로 최소값 step 선택
    if not label:  # label이 None/빈문자라면 선택 불가
        return None

    acc = ea.EventAccumulator(pt_dir, size_guidance={ea.SCALARS: 0})
    acc.Reload()
    try:
        events = acc.Scalars(label)  # 예: 'valid/fid'
    except KeyError:
        return None

    vals = {e.step: e.value for e in events}  # step -> scalar
    cands = [(vals[s], s, f) for s, f in files.items() if s in vals]
    return min(cands)[2] if cands else None

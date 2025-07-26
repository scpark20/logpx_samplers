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
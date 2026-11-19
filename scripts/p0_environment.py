"""Phase 0: record the machine we run on. Writes results/p0_environment.json."""
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import ROOT, SEED, folder_size_bytes, seed_everything, write_result  # noqa: E402


def main():
    seed_everything()
    import torch

    gpu = torch.cuda.is_available()
    disk = shutil.disk_usage(ROOT)
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_logical_cores": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": gpu,
        "gpu_name": torch.cuda.get_device_name(0) if gpu else None,
        "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if gpu else None,
        "disk_free_gb": round(disk.free / 1e9, 1),
        "venv_size_gb": round(folder_size_bytes(ROOT / ".venv") / 1e9, 2),
        "storage_budget_gb": 10.0,
        "seed": SEED,
    }
    write_result("p0_environment", info)


if __name__ == "__main__":
    main()

"""Run real CUDA parity gates; absent hardware is a failure, not a passing skip."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import time
import xml.etree.ElementTree as ET


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "cuda-parity.json"
    report = {
        "started_at_unix": time(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "phases": [],
        "status": "preflight",
        "source_sha256": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in (
                "hdbscan_gpu/*.py", "tests/*.py", "scripts/*.py", "scripts/*.sbatch",
                "pyproject.toml", "uv.lock",
            )
            for path in sorted(root.glob(pattern))
        },
    }
    write_report(report_path, report)
    import cupy as cp
    import numpy as np
    import sklearn

    if np.__version__ != "2.3.5" or sklearn.__version__ != "1.9.1":
        raise RuntimeError("CUDA parity requires NumPy 2.3.5 and scikit-learn 1.9.1")
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError("allocate exactly one visible CUDA GPU for this bounded job")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    probe = cp.arange(8, dtype=cp.float64)
    if float(cp.sum(probe).item()) != 28.0:
        raise RuntimeError("CUDA float64 preflight failed")
    report.update({
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "cupy": cp.__version__,
        "pytest": importlib.metadata.version("pytest"),
        "gpu": name.decode() if isinstance(name, bytes) else name,
        "compute_capability": [properties["major"], properties["minor"]],
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
        "status": "running",
    })
    write_report(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key != "source_sha256"}), flush=True)
    cp.show_config()
    sys.stdout.flush()

    for precision in ("float64", "float32"):
        token = f"cuda-{precision}"
        junit = output / f"{token}.xml"
        selection = f"{token} or cpu_oracle or float64_oracle" if precision == "float64" else token
        command = [
            sys.executable, "-m", "pytest", "-q", "-rA", "-p", "no:cacheprovider",
            "tests", "-k", selection, f"--junitxml={junit}",
        ]
        start = time()
        result = subprocess.run(command, cwd=root, check=False)
        phase = {
            "precision": precision,
            "returncode": result.returncode,
            "seconds": time() - start,
            "junit": junit.name,
        }
        if junit.exists():
            cases = ET.parse(junit).getroot().findall(".//testcase")
            cuda_cases = [case for case in cases if token in case.get("name", "")]
            phase.update({
                "selected_tests": len(cases),
                "cuda_tests": len(cuda_cases),
                "cuda_skipped": sum(case.find("skipped") is not None for case in cuda_cases),
                "cuda_failed": sum(
                    case.find("failure") is not None or case.find("error") is not None
                    for case in cuda_cases
                ),
            })
        passed = (
            result.returncode == 0
            and phase.get("cuda_tests", 0) > 0
            and phase.get("cuda_skipped", 0) == 0
            and phase.get("cuda_failed", 0) == 0
        )
        phase["passed"] = passed
        report["phases"].append(phase)
        report["status"] = "running" if passed else "failed"
        write_report(report_path, report)
        print(json.dumps(phase), flush=True)
        if not passed:
            return 1
    report.update({"status": "passed", "finished_at_unix": time()})
    write_report(report_path, report)
    print(f"CUDA parity passed; evidence: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

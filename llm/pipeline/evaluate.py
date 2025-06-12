import subprocess
from pathlib import Path

EVAL_SCRIPT = Path(__file__).resolve().parent.parent / "InstructKGC" / "ie2instruction" / "eval_func.py"


def evaluate(result_path: Path, task: str, sort_by: str = ""):
    cmd = [
        "python", str(EVAL_SCRIPT),
        "--path1", str(result_path),
        "--task", task,
    ]
    if sort_by:
        cmd.extend(["--sort_by", sort_by])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    evaluate(Path("kg_output.json"), "RE")

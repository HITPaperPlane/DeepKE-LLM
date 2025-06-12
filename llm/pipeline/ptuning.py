import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "InstructKGC" / "src" / "finetuning_chatglm_pt.py"


def run_ptuning(train_path: Path, output_dir: Path, model_dir: Path = Path("/model")):
    cmd = [
        "deepspeed", "--include", "localhost:0", str(SCRIPT),
        "--train_path", str(train_path),
        "--model_dir", str(model_dir),
        "--num_train_epochs", "3",
        "--train_batch_size", "2",
        "--gradient_accumulation_steps", "1",
        "--output_dir", str(output_dir),
        "--log_steps", "10",
        "--max_len", "768",
        "--max_src_len", "450",
        "--pre_seq_len", "16",
        "--prefix_projection", "true",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    run_ptuning(Path("dataset/InstructIE/train_zh_aug.json"), Path("ptuning_out"))

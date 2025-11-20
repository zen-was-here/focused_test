#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path
import os
import shutil

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
PYTHON_VERSION = "3.11"  # target Python version

def ensure_venv():
    """Create a venv using Python 3.11 if it doesn't exist."""
    if not VENV_DIR.exists():
        # Use pyenv to find Python 3.11
        python3_11 = shutil.which("python3.11")
        if python3_11 is None:
            raise RuntimeError("Python 3.11 not found. Install via pyenv or system package manager.")
        subprocess.run([python3_11, "-m", "venv", str(VENV_DIR)], check=True)

def venv_python():
    """Return path to the python executable in the venv."""
    return VENV_DIR / "bin" / "python"

def run_subprocess(args, **kwargs):
    """Run a subprocess using project root as cwd."""
    return subprocess.run(args, cwd=PROJECT_ROOT, check=True, **kwargs)

def setup():
    ensure_venv()
    run_subprocess([str(venv_python()), "-m", "knowledge_base.setup_kb"])

def run():
    ensure_venv()
    run_subprocess([str(venv_python()), "main.py"])

def eval_agent():
    ensure_venv()
    run_subprocess([str(venv_python()), "-m", "eval.eval"])

def clean():
    # Remove Chroma database
    chroma_db = PROJECT_ROOT / "chroma_db"
    if chroma_db.exists() and chroma_db.is_dir():
        shutil.rmtree(chroma_db)

    # Remove eval results
    eval_file = PROJECT_ROOT / "eval_results.json"
    if eval_file.exists():
        eval_file.unlink()

    # Remove __pycache__ directories and .pyc files
    for root, dirs, files in os.walk(PROJECT_ROOT, topdown=False):
        for name in dirs:
            if name == "__pycache__":
                shutil.rmtree(Path(root) / name)
        for file in files:
            if file.endswith(".pyc"):
                os.remove(Path(root) / file)

def help_message():
    print("""Travel Booking Agent - Python Script Commands

  setup      - Initialize knowledge base
  run        - Run the CLI agent
  eval       - Run evaluation suite
  clean      - Clean generated files
  help       - Show this help message

Local Models:
  Set MODEL=llama3.2 in .env for Ollama
""")

def main():
    parser = argparse.ArgumentParser(description="Travel Booking Agent commands")
    parser.add_argument("command", help="Command to run", choices=["setup", "run", "eval", "clean", "help"])
    args = parser.parse_args()

    commands = {
        "setup": setup,
        "run": run,
        "eval": eval_agent,
        "clean": clean,
        "help": help_message
    }

    commands[args.command]()

if __name__ == "__main__":
    main()

import subprocess
import sys

scripts = [
    "01_build_manifest.py",
    "02_filter_art.py",
    "03_rule_preclassify.py",
    "04_ollama_classify.py",
    "05_merge_and_review.py",
]

for script in scripts:
    print(f"\n===== Running {script} =====")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        raise SystemExit(f"[FAILED] {script}")
print("\n[ALL DONE]")

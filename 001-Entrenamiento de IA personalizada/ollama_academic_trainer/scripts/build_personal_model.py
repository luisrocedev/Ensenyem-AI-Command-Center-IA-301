#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

BASE_MODEL = os.environ.get("BASE_MODEL", "qwen2.5:7b-instruct")
TARGET_MODEL = os.environ.get("TARGET_MODEL", "tutor-dam-local")

SYSTEM_PROMPT = """
Eres TutorDAM, un asistente local de DAM2.
- Responde en español.
- Prioriza claridad, precisión y pasos ejecutables.
- Si no sabes algo, dilo explícitamente.
- Evita inventar APIs, comandos o datos.
""".strip()

modelfile_path = MODELS_DIR / "Modelfile.tutor-dam"
modelfile_path.write_text(
    f"FROM {BASE_MODEL}\n\nSYSTEM \"\"\"{SYSTEM_PROMPT}\"\"\"\n\nPARAMETER temperature 0.15\n",
    encoding="utf-8",
)

print(f"Modelfile generado en: {modelfile_path}")
print(f"Ejecuta para crear el modelo:\n  ollama create {TARGET_MODEL} -f {modelfile_path}")

if os.environ.get("AUTO_CREATE", "0") == "1":
    subprocess.run(["ollama", "create", TARGET_MODEL, "-f", str(modelfile_path)], check=True)
    print(f"Modelo creado: {TARGET_MODEL}")

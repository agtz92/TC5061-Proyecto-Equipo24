"""Chequeos de seguridad del proyecto MiBici en un solo comando.

    python security/run_checks.py              # todo: secretos, bandit y pip-audit
    python security/run_checks.py --skip-audit # sin pip-audit (no requiere internet)
    python security/run_checks.py --installed  # audita el entorno activo en vez de requirements.txt

Pasos:
1. Higiene del repositorio (pytest tests/test_security.py): .gitignore, .env fuera
   de Git, sin credenciales escritas en archivos versionados y sin URIs fijas.
2. Bandit: análisis estático del código (mibici/, security/ y scripts de la raíz).
   Falla con cualquier hallazgo de severidad media o alta.
3. pip-audit: vulnerabilidades conocidas (PyPI/OSV) en las dependencias fijadas en
   requirements.txt. Falla si aparece una que no esté justificada en
   security/riesgos_aceptados.toml.

Las salidas se guardan en reports/security/ como evidencia. Código de salida 0 solo
si los tres pasos pasan.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import subprocess  # nosec B404
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "security"
ACCEPTED_FILE = ROOT / "security" / "riesgos_aceptados.toml"
BANDIT_TARGETS = ["mibici", "security", "guardarmodelos.py", "verificarmodelos.py"]


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Ejecuta una herramienta con el mismo intérprete (argumentos fijos, sin shell)."""
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)  # nosec B603


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def check_repo_hygiene() -> bool:
    banner("1/3  Higiene del repositorio: secretos, .gitignore y variables de entorno")
    res = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_security.py"])
    print(res.stdout.strip() or res.stderr.strip())
    (OUT_DIR / "repo_hygiene.txt").write_text(res.stdout + res.stderr, encoding="utf-8")
    return res.returncode == 0


def check_bandit() -> bool:
    banner("2/3  Bandit: análisis estático del código")
    base = [sys.executable, "-m", "bandit", "-c", "pyproject.toml", "-r", *BANDIT_TARGETS]
    res = run([*base, "-f", "txt"])
    run([*base, "-f", "json", "-o", str(OUT_DIR / "bandit.json")])
    text = res.stdout
    (OUT_DIR / "bandit.txt").write_text(text, encoding="utf-8")
    start = text.find("Test results:")
    print(text[start:].strip() if start >= 0 else (text + res.stderr).strip())
    # Bandit regresa 1 con cualquier hallazgo; se decide aquí: falla solo con media o alta.
    try:
        data = json.loads((OUT_DIR / "bandit.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("No se pudo leer el resultado de bandit:", res.stderr.strip())
        return False
    serious = [r for r in data["results"] if r["issue_severity"] in ("MEDIUM", "HIGH")]
    return not serious and not data.get("errors")


def load_accepted() -> dict[tuple[str, str], dict]:
    data = tomllib.loads(ACCEPTED_FILE.read_text(encoding="utf-8"))
    accepted = {}
    for item in data.get("aceptado", []):
        accepted[(item["paquete"].lower(), item["version"])] = item
    return accepted


def check_pip_audit(installed: bool) -> bool:
    banner("3/3  pip-audit: vulnerabilidades conocidas en dependencias")
    json_out = OUT_DIR / "pip_audit.json"
    cmd = [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", "--desc", "off",
           "-f", "json", "-o", str(json_out)]
    if installed:
        source = "entorno instalado"
        res = run(cmd)
    else:
        # El módulo propio (-e .) no está en PyPI: se audita el resto de requirements.txt.
        lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        reqs = [line for line in lines if not line.strip().startswith("-e")]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("\n".join(reqs) + "\n")
        source = "requirements.txt"
        res = run([*cmd, "-r", tmp.name])
        Path(tmp.name).unlink(missing_ok=True)
    if not json_out.exists():
        print(res.stdout, res.stderr)
        return False

    deps = json.loads(json_out.read_text(encoding="utf-8"))["dependencies"]
    accepted = load_accepted()
    new, known = [], []
    for dep in deps:
        vulns = dep.get("vulns", [])
        if not vulns:
            continue
        ids = sorted({v["id"] for v in vulns})
        key = (dep["name"].lower(), dep.get("version", ""))
        (known if key in accepted else new).append((dep["name"], dep.get("version", ""), ids, vulns))

    lines = [f"pip-audit sobre {source}: {len(deps)} paquetes revisados ({date.today()})", ""]
    if new:
        lines.append("VULNERABILIDADES NUEVAS (no aceptadas) - actualizar o justificar:")
        for name, ver, ids, vulns in new:
            fixes = sorted({f for v in vulns for f in v.get("fix_versions", [])})
            lines.append(f"  {name}=={ver}: {len(ids)} aviso(s) {', '.join(ids)}; corregido en {', '.join(fixes) or '-'}")
    else:
        lines.append("Sin vulnerabilidades nuevas: todas las dependencias pasan o tienen riesgo aceptado.")
    if known:
        lines.append("")
        lines.append("Riesgos aceptados y documentados en security/riesgos_aceptados.toml:")
        for name, ver, ids, _ in known:
            item = accepted[(name.lower(), ver)]
            lines.append(f"  {name}=={ver}: {len(ids)} aviso(s); revisar antes de {item.get('revisar', '-')}")
    skipped = [d["name"] for d in deps if "skip_reason" in d]
    if skipped:
        lines.append("")
        lines.append(f"Omitidos (no están en PyPI): {', '.join(skipped)}")
    summary = "\n".join(lines)
    print(summary)
    (OUT_DIR / "pip_audit.txt").write_text(summary + "\n", encoding="utf-8")

    expired = [k for k in known if str(accepted[(k[0].lower(), k[1])].get("revisar", "9999")) < str(date.today())]
    if expired:
        print("\nAceptaciones vencidas, revisar:", ", ".join(f"{n}=={v}" for n, v, *_ in expired))
    return not new and not expired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-audit", action="store_true", help="omite pip-audit (requiere internet)")
    parser.add_argument("--installed", action="store_true", help="audita el entorno activo")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {"Higiene del repositorio": check_repo_hygiene(), "Bandit": check_bandit()}
    if not args.skip_audit:
        results["pip-audit"] = check_pip_audit(args.installed)

    banner("Resumen")
    for name, ok in results.items():
        print(f"  {'OK   ' if ok else 'FALLA'}  {name}")
    print(f"\nEvidencia en {OUT_DIR.relative_to(ROOT)}/")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())

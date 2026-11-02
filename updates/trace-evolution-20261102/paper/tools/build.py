"""Compile only: never overwrite evidence tables with historical results."""

import os
from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "build"
    output.mkdir(exist_ok=True)
    env = dict(os.environ, TEXINPUTS=".:..:", BIBINPUTS=".:..:", BSTINPUTS=".:..:")
    latex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error"]
    commands = [latex + ["../main.tex"], ["bibtex", "main"],
                latex + ["../main.tex"], latex + ["../main.tex"],
                latex + ["../supplement.tex"], ["bibtex", "supplement"],
                latex + ["../supplement.tex"], latex + ["../supplement.tex"]]
    with (output / "build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=output, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ("main.log", "supplement.log"):
        text = (output / name).read_text(errors="replace")
        if any(token in text for token in ("Overfull", "Float too large", "undefined references", "undefined citations", "multiply defined")):
            raise RuntimeError(f"Inspect unresolved layout/reference diagnostics: {output / name}")
    print("Compiled main.pdf and supplement.pdf; no overfull or unresolved cross-references.")


if __name__ == "__main__":
    main()

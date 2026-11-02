"""Render every current PDF page and verify publication diagnostics/fonts."""

import hashlib
import json
from pathlib import Path
import re
import subprocess

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "build" / "publication_qa"
    output.mkdir(exist_ok=True)
    report = {}
    for name in ("main", "supplement"):
        pdf = root / "build" / f"{name}.pdf"
        log = pdf.with_suffix(".log").read_text(errors="replace")
        for term in ("Overfull", "Float too large", "undefined references",
                     "undefined citations", "multiply defined"):
            if term in log:
                raise RuntimeError(f"{name}: unresolved {term}")
        fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
        if "Type 3" in fonts or "CID" in fonts:
            raise RuntimeError(f"{name}: unsupported publication font")
        for row in fonts.splitlines()[2:]:
            if not re.search(r"\byes\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$", row):
                raise RuntimeError(f"{name}: unembedded or malformed font row {row}")
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
        count = int(re.search(r"Pages:\s+(\d+)", info).group(1))
        subprocess.run(["pdftoppm", "-png", "-r", "100", str(pdf), str(output / name)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        pages = sorted(output.glob(f"{name}-*.png"))
        if len(pages) != count:
            raise RuntimeError("Rendered page inventory differs")
        contacts = []
        for start in range(0, count, 4):
            contact = Image.new("RGB", (1260, 1680), "#b9bfc3")
            draw = ImageDraw.Draw(contact)
            for offset, path in enumerate(pages[start:start + 4]):
                with Image.open(path) as raw:
                    raw = raw.convert("RGB")
                    raw.thumbnail((610, 800))
                    x, y = (offset % 2) * 630 + 10, (offset // 2) * 840 + 25
                    contact.paste(raw, (x, y))
                    draw.text((x, y - 18), f"{name} page {start + offset + 1}", fill="black")
            path = output / f"{name}_contact_{start // 4 + 1}.png"
            contact.save(path)
            contacts.append(str(path.relative_to(root)))
        report[name] = dict(pages=count, sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                            fonts=fonts, contacts=contacts,
                            visual_inspection="Required separately by author; rendering alone is not visual approval")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

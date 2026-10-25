#!/usr/bin/env python3
"""Publish completed CPU analysis only; GPU plans stay explicitly pending."""
import hashlib
import json
from pathlib import Path
import shutil

import export_results as ex

SOURCE=Path('/home/wwww/hzl/results/rebuttal_evidence_extensions_20261025_2012')
DEST=Path(__file__).resolve().parents[1]/'updates/mechanism-audit-20261025'


def main():
    if DEST.exists():raise FileExistsError(DEST)
    if json.loads((SOURCE/'offline/provenance.json').read_text())['status']!='completed':
        raise ValueError('CPU evidence is incomplete')
    files=list((SOURCE/'offline').glob('*'))+[SOURCE/'README.md',SOURCE/'manifest.json',SOURCE/'manifest_seal.json']
    manifest=[]
    for source in files:
        if not source.is_file():continue
        payload=source.read_bytes()
        if ex.has_secret(payload):raise ValueError('Sensitive content; not uploading: '+str(source))
        target=DEST/source.relative_to(SOURCE);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        digest=hashlib.sha256(payload).hexdigest()
        if ex.digest(source)!=digest or ex.digest(target)!=digest:raise ValueError('Source changed during export')
        manifest.append(dict(path=str(target.relative_to(DEST)),bytes=len(payload),sha256=digest))
    (DEST/'verification.json').write_text(json.dumps(dict(verified=True,source=str(SOURCE),
        included='completed CPU analysis; pending GPU manifest explicitly not a result',
        gpu_results_included=False,files=manifest),indent=2)+'\n')


if __name__=='__main__':main()

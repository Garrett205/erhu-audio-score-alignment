"""Data-free snapshot integrity and privacy checks."""
from pathlib import Path
import ast,hashlib,json,re

def main():
    root=Path(__file__).resolve().parent
    prohibited={'.wav','.flac','.mp3','.m4a','.musicxml','.mxl','.xml','.pt','.pth','.ckpt','.png','.docx','.pdf'}
    forbidden_dirs={'__pycache__','node_modules','.venv','checkpoints'}
    failures=[];count=0
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if '.git' in relative.parts or relative.parts[0] in {'work','inputs'}:continue
        if path.is_dir():
            if path.name in forbidden_dirs:failures.append(str(relative))
            continue
        count+=1
        if path.suffix.lower() in prohibited or path.stat().st_size>20*1024*1024:failures.append(str(relative))
        text=path.read_text(encoding='utf-8-sig')
        if re.search(r'\b[A-Za-z]:[\\/]|/(?:Users|home)/',text):failures.append(f'absolute path: {relative}')
        if re.search(r'(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY)',text):failures.append(f'secret pattern: {relative}')
        if path.suffix=='.py':compile(text,str(relative),'exec')
    constants=json.loads((root/'configs/frozen_constants.json').read_text(encoding='utf-8'))['onset.py']
    expected={'START_GATE_CUT_HZ':1800,'START_GATE_PERI_TH':.01,'START_GATE_OK_FR':40,'START_GATE_BACK_FR':40,'END_GATE_CUT_HZ':1800,'END_GATE_PERI_TH':.01,'END_GATE_OK_FR':30,'END_GATE_BACK_FR':-28}
    source_constants={}
    for n in ast.parse((root/'src/pipeline/onset.py').read_text(encoding='utf-8')).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Name) and t.id in expected:source_constants[t.id]=ast.literal_eval(n.value)
    assert source_constants==expected
    assert all(constants[k]==v for k,v in expected.items())
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    assert not failures,failures
    print(f'PASS: {count} files; source syntax, hashes, gate constants, prohibited artifacts, absolute paths and secret patterns')

if __name__=='__main__':main()

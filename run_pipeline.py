"""Portable I/O launcher around the frozen paper scripts."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Authorized WAV')
    parser.add_argument('--score', type=Path, required=True, help='Authorized MusicXML')
    parser.add_argument('--output', type=Path, required=True, help='New work directory')
    parser.add_argument('--pitch-only', action='store_true')
    args = parser.parse_args()
    audio, score = args.input.resolve(), args.score.resolve()
    if audio.suffix.lower() != '.wav' or not audio.is_file():
        parser.error('--input must be an existing WAV')
    if not score.is_file() or score.suffix.lower() not in {'.xml', '.musicxml'}:
        parser.error('--score must be an existing MusicXML file')
    work = args.output.resolve()
    work.mkdir(parents=True, exist_ok=False)
    # Copy authorized inputs into an isolated work area; stages can write only there.
    shutil.copyfile(audio, work / 'input.wav')
    shutil.copyfile(score, work / 'input.musicxml')
    env = os.environ.copy()
    env.update(ERHU_WORK_DIR=str(work), MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1',
               PYTHONIOENCODING='utf-8', MPLCONFIGDIR=str(work / '.mpl'))
    pipe = Path(__file__).resolve().parent / 'src/pipeline'
    stages = ['PMSDB.py', 'pitch.py']
    if not args.pitch_only:
        stages += ['mxml.py', 'onset.py', 'os.py', 'dp1.py', 'dp2.py', 'dp3.py']
    for stage in stages:
        with (work / (stage + '.log')).open('w', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-B', str(pipe / stage)], cwd=work,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if not args.pitch_only:
        subprocess.run([sys.executable, '-B', str(pipe / 'run_mel.py'),
                        '--input', str(work / 'input.wav'), '--work', str(work)],
                       cwd=work, env=env, check=True)
    print(work)


if __name__ == '__main__':
    main()

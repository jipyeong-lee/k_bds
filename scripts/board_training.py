#!/usr/bin/env python3
"""board_training.py — 판의 「학습 모니터」 탭 자료를 만든다. 생성물: design/training.json · design/training.html.
KISTI 에서 실행 중인 GRPO job 의 로그를 읽어 step 별 지표를 JSON 으로 쓴다. 읽기 전용(ssh BatchMode, 별칭 kbds266).
사용: python3 scripts/board_training.py [--interval 60] [--once]   화면은 30초마다 training.json 을 다시 읽는다."""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'design')
ALIAS = 'kbds266'
# 원격(/bin/sh 로 들어와 bash -s 로 받는다). 실행 중 job 마다 logs/*_<id>.log 에서 지표 dict 줄을 읽고 JSON 한 줄로 낸다
REMOTE = r'''python3 - <<'PY'
import ast, glob, json, os, re, subprocess
KEYS = {'rewards/FormatThink/mean': 'fmt', 'completions/clipped_ratio': 'clip', 'rewards/AccuracyMix/mean': 'acc',
        'completions/mean_length': 'len', 'step_time': 'time', 'memory(GiB)': 'mem', 'reward': 'reward', 'entropy/mean': 'ent'}
os.chdir(os.path.expanduser('~/kbds_project'))
q = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-t', 'R', '-o', '%i|%j|%M|%L|%N'], capture_output=True, text=True).stdout
out = []
for line in q.splitlines():
    jid, name, elapsed, left, node = (line.split('|') + [''] * 5)[:5]
    logs = glob.glob('logs/*_%s.log' % jid)
    if not (jid.isdigit() and logs): continue
    rows, last = [], None
    for l in open(logs[0], errors='ignore'):
        m = re.search(r"\{'loss'.*\}", l)
        if not m: continue
        try: d = ast.literal_eval(m.group(0))
        except Exception: continue
        step = int(d['global_step/max_steps'].split('/')[0]); total = int(d['global_step/max_steps'].split('/')[1])
        rows.append([step] + [float(d[k]) for k in KEYS]); last = d
    if not rows: continue
    cks = glob.glob('work/checkpoints/*/v*/checkpoint-*')  # 가장 최근에 저장된 체크포인트가 든 폴더 = 지금 학습의 출력
    ck = sorted(int(p.rsplit('-', 1)[1]) for p in cks if os.path.dirname(p) == os.path.dirname(max(cks, key=os.path.getmtime))) if cks else []
    out.append({'id': jid, 'name': name, 'elapsed': elapsed, 'left': left, 'node': node, 'total': total, 'cols': ['step'] + list(KEYS.values()),
                'rows': rows, 'checkpoints': ck, 'swift_remaining': last.get('remaining_time'), 'swift_elapsed': last.get('elapsed_time'), 'log': logs[0]})
print(json.dumps(out))
PY
'''


def collect():
    try:
        r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', ALIAS, 'bash', '-s'], input=REMOTE,
                           capture_output=True, text=True, errors='replace', timeout=60)
    except Exception as e:
        return {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:160]}', 'jobs': []}
    lines = [l for l in r.stdout.splitlines() if l.startswith('[')]
    if not lines:
        return {'ok': False, 'error': 'ssh 연결 없음 — OTP 로그인이 필요하다' if r.returncode else '실행 중인 학습 로그가 없다', 'jobs': []}
    return {'ok': True, 'error': None, 'jobs': json.loads(lines[-1])}


def write(snap):
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, 'training.json.tmp')  # 임시 파일 → rename: 화면이 반쯤 쓴 파일을 읽지 않는다
    with open(tmp, 'w', encoding='utf-8') as f: json.dump(snap, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(OUT, 'training.json'))
    shutil.copyfile(os.path.join(ROOT, 'scripts', 'board_training.html'), os.path.join(OUT, 'training.html'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--interval', type=int, default=60); ap.add_argument('--once', action='store_true')
    a = ap.parse_args()
    while True:
        snap = {**collect(), 'updated': time.time(), 'updated_text': time.strftime('%Y-%m-%d %H:%M:%S'), 'interval': a.interval}
        write(snap)
        print(f'[{snap["updated_text"]}] ' + (f'jobs={len(snap["jobs"])} steps=' + ','.join(str(j['rows'][-1][0]) for j in snap['jobs']) if snap['ok'] else snap['error']), flush=True)
        if a.once: return
        time.sleep(a.interval)


if __name__ == '__main__':
    try: main()
    except KeyboardInterrupt: sys.exit(0)

"""TMS4 다운로드 자료 QC 플래그 부여 및 지점별 분리 저장.

플래그: 0 정상, 1 다운로드(현장 방문) 중, 3 접촉 이상 후 복구된 구간(보정값),
        4 접촉 이상 후 회복 없음, 8 스파이크, 9 결측(NaN)
우선순위: 9 > 1 > 8 > 4 > 3 > 0
시간: 원자료 UTC, 현장 기록지 현지시각(UTC+2)
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

IN_DIR = Path('/root/.claude/uploads/44a75c9f-5d9e-5247-a7be-759905774f5b')
OUT_DIR = Path('/tmp/claude-0/-home-claude/44a75c9f-5d9e-5247-a7be-759905774f5b/scratchpad/tms/out')
LOCAL_OFFSET = pd.Timedelta(hours=2)          # 기록지 시각 = UTC + 2
COLS = ['idx', 'time', 'T1', 'T2', 'T3', 'raw', 'vwc', 'err']
DROP_THR = 1.0                                # 15분 사이 VWC 하강 기준 (%p)
SPIKE_THR = 1.0                               # 스파이크 기준 (%p)
PRIORITY = [9, 1, 8, 4, 3]                    # 앞쪽이 우선

# 센서별 설치 구간 (UTC, 자료에서 확인). (지점, 시작, 끝). 끝=None이면 파일 끝까지.
DEPLOY = {
    'A_S5':   [('A',  '2026-07-13 10:15', '2026-07-20 10:45'),
               ('A',  '2026-07-20 11:00', None)],
    'A_L4':   [('A',  '2026-07-13 10:15', '2026-07-20 11:00'),
               ('A',  '2026-07-20 11:15', None)],
    'A_S1':   [('C5', '2026-07-11 14:15', '2026-08-01 15:00'),
               ('A',  '2026-08-01 15:15', None)],
    'A_L10':  [('C5', '2026-07-11 14:15', '2026-08-01 15:00'),
               ('A',  '2026-08-01 15:15', None)],
    'B1_S3':  [('B1', '2026-07-12 13:45', None)],
    'B1_L12': [('B1', '2026-07-12 14:00', None)],
    'B1_L9':  [('B1', '2026-07-26 17:30', None)],
    'B2_S11': [('B2', '2026-07-12 10:30', None)],
    'B2_L1':  [('B2', '2026-07-12 10:30', None)],
}

# 설치 구간 안에서 센서가 땅 밖에 있던 시각 (UTC) -> 9
OUT_OF_GROUND = {
    'A_L4':  [('2026-07-20 11:00', '2026-07-20 11:00')],
    'A_S1':  [('2026-08-01 13:45', '2026-08-01 15:00')],
    'A_L10': [('2026-08-01 13:45', '2026-08-01 15:00')],
}

# TMS 다운로드 방문 (현지시각, 도착-출발). 기록지에서 TMS·mushroom·S/L 관련 항목만.
VISITS = {
    'A':  [('07-15 11:20', '07-15 11:40'), ('07-19 20:20', '07-19 21:00'),
           ('07-23 19:33', '07-23 19:43'), ('07-28 14:03', '07-28 15:13'),
           ('08-01 16:44', '08-01 17:11'), ('08-04 17:35', '08-04 17:45'),
           ('08-07 17:17', '08-07 17:24')],
    'B1': [('07-15 18:37', '07-15 19:00'), ('07-19 17:43', '07-19 17:53'),
           ('07-23 18:56', '07-23 19:07'), ('07-28 13:38', '07-28 13:50'),
           ('08-01 17:26', '08-01 17:35'), ('08-04 17:54', '08-04 18:02'),
           ('08-07 17:52', '08-07 17:58')],
    'B2': [('07-15 12:45', '07-15 13:25'), ('07-19 16:25', '07-19 16:36'),
           ('07-23 17:53', '07-23 17:59'), ('07-28 15:50', '07-28 15:58'),
           ('08-01 16:03', '08-01 16:10'), ('08-04 15:46', '08-04 15:53'),
           ('08-07 15:45', '08-07 15:52')],
    'C5': [('07-15 13:55', '07-15 14:05'), ('07-19 15:10', '07-19 15:20'),
           ('07-23 17:30', '07-23 17:35'), ('07-28 16:22', '07-28 16:25'),
           ('08-01 15:15', '08-01 15:44')],
}

# 접촉 이상 구간 (UTC). recover=None이면 회복 없음 -> 4, 있으면 복원 -> 3.
# drop: 처음 떨어진 기록, recover: 회복 직후 첫 기록
CONTACT = {
    'B2_S11': [('2026-07-15 11:15', '2026-07-30 15:45'),   # 7/30 흙 채워 복구
               ('2026-08-01 14:15', None)],                # 삼각대 철거 후 회복 없음
}


def read_tms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', header=None, names=COLS, decimal='.')
    df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M')
    return (df.drop_duplicates('time', keep='last')
              .set_index('time').sort_index())


def to_utc(local: str, year: int = 2026) -> pd.Timestamp:
    return pd.Timestamp(f'{year}-{local}') - LOCAL_OFFSET


def detect_drops(v: pd.Series) -> pd.Series:
    """15분 하강폭이 DROP_THR 이상인 시각(검토용)."""
    dv = v.diff()
    return dv[dv <= -DROP_THR]


def spike_mask(v: pd.Series) -> pd.Series:
    """한 기록만 튀었다가 바로 돌아오는 값."""
    d_prev, d_next = v.diff(), v.diff(-1)
    return ((d_prev.abs() >= SPIKE_THR) & (d_next.abs() >= SPIKE_THR)
            & (np.sign(d_prev) == np.sign(d_next)))


def restore(v: pd.Series, drop: pd.Timestamp, recover: pd.Timestamp) -> pd.Series:
    """떨어진 구간: anomaly(구간 양끝 직선 제거) + (직전값~회복직후값 직선)."""
    seg = v.loc[drop:recover - pd.Timedelta(seconds=1)]
    x = np.linspace(0.0, 1.0, len(seg))
    seg_line = seg.iloc[0] + (seg.iloc[-1] - seg.iloc[0]) * x
    pre = v.loc[:drop].iloc[-2]          # 떨어지기 직전 값
    post = v.loc[recover]                # 회복 직후 값
    target = pre + (post - pre) * x
    return pd.Series(seg.to_numpy() - seg_line + target, index=seg.index)


def process(key: str, df: pd.DataFrame) -> list[tuple[str, pd.DataFrame, list[str]]]:
    v = df['vwc'].astype(float)
    t = df.index
    flags = {f: pd.Series(False, index=t) for f in PRIORITY}
    vwc_qc = v.copy()
    notes = []

    for a, b in OUT_OF_GROUND.get(key, []):
        flags[9] |= (t >= a) & (t <= b)
    flags[9] |= v.isna() | (v <= 0)

    flags[8] |= spike_mask(v)

    for drop, rec in CONTACT.get(key, []):
        drop = pd.Timestamp(drop)
        if rec is None:
            flags[4] |= t >= drop
            notes.append(f'flag4 {drop:%m-%d %H:%M} UTC ~ 끝')
        else:
            rec = pd.Timestamp(rec)
            fixed = restore(v, drop, rec)
            vwc_qc.loc[fixed.index] = fixed
            flags[3] |= (t >= drop) & (t < rec)
            notes.append(f'flag3 {drop:%m-%d %H:%M} ~ {rec:%m-%d %H:%M} UTC '
                         f'(보정량 {(fixed - v.loc[fixed.index]).iloc[0]:+.1f} -> '
                         f'{(fixed - v.loc[fixed.index]).iloc[-1]:+.1f} %p)')

    results = []
    for site, start, end in DEPLOY[key]:
        in_dep = (t >= pd.Timestamp(start)) & ((t <= pd.Timestamp(end)) if end else True)
        fl1 = pd.Series(False, index=t)
        for a, b in VISITS.get(site, []):
            fl1 |= (t >= to_utc(a)) & (t <= to_utc(b))
        flags_site = {**flags, 1: flags[1] | fl1}

        flag = pd.Series(0, index=t, dtype='int8')
        for f in reversed(PRIORITY):
            flag[flags_site[f]] = f

        out = df.loc[in_dep, ['T1', 'T2', 'T3', 'raw', 'vwc']].rename(columns={'vwc': 'vwc_raw'})
        out['vwc_qc'] = vwc_qc[in_dep].where(flag[in_dep] != 9)
        out['flag'] = flag[in_dep]
        out.insert(0, 'time_local', out.index + LOCAL_OFFSET)
        out.index.name = 'time_utc'

        sensor = key.split('_')[1]
        name = f'{site}_{sensor}_{out.index[0]:%Y%m%d%H%M}_{out.index[-1]:%Y%m%d%H%M}'
        drops = detect_drops(v[in_dep & (flag == 0)])
        site_notes = notes + [f'검토 필요 하강 {ts:%m-%d %H:%M} UTC {dv:+.1f}'
                              for ts, dv in drops.items()]
        results.append((name, out, site_notes))
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob('*_2026*.csv')):
        key = re.sub(r'^[0-9a-f]+-', '', path.stem).rsplit('_', 1)[0]
        if key not in DEPLOY:
            print(f'[건너뜀] {path.name}: DEPLOY 설정 없음')
            continue
        df = read_tms(path)
        n_pre = (df.index < pd.Timestamp(DEPLOY[key][0][1])).sum()
        for name, out, notes in process(key, df):
            out.to_csv(OUT_DIR / f'{name}.csv', float_format='%.2f')
            counts = out['flag'].value_counts().sort_index()
            print(f'{name}: {len(out)}행, ' +
                  ', '.join(f'flag{k}={v}' for k, v in counts.items()))
            for n in notes:
                print('   ', n)
        print(f'    (설치 전 {n_pre}행은 출력에서 제외)')


if __name__ == '__main__':
    main()

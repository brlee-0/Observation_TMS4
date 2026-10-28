"""Check the output of tms4_flag.py before the work is reported (tms4-download-check skill).

Usage: python check_result.py <TMS CSV folder> <output folder> [--max-missing 0.05]

ERROR = the output breaks a rule of the skill. Fix it and run again.
WARN  = the output may be right, but the case must be listed under "확인 필요" in the report.
Exit code 1 if any ERROR, otherwise 0. The report is also written to <output>/check_report.txt.

Checks
  E1  processing_log.txt exists
  E2  output file names: <site>_<sensor>_<YYYYMMDDHHMM>_<YYYYMMDDHHMM>.csv
  E3  columns: time_utc, time_local, T1, T2, T3, raw, vwc_raw, vwc_qc, flag
  E4  time_utc unique, every 15 min without gaps, first/last = file name
  E5  time_local = time_utc + LOCAL_OFFSET
  E6  flag only 0, 1, 3, 4, 8, 9
  E7  vwc_raw missing or <= 0  ->  flag 9
  E8  flag 9  ->  vwc_qc empty
  E9  flag 0/4/8, and 1 outside a restored segment  ->  vwc_qc = vwc_raw
  E10 flag 3  ->  vwc_qc present
  E11 raw values unchanged: T1, T2, T3, vwc_raw equal the input files at the same time
  E12 vwc_qc of every row not flagged 9 inside the physical range (0-60 %)
  E13 every input sensor is in DEPLOY (no guessed sensor-site mapping)
  E14 every input sensor in DEPLOY has output; deployments of one sensor do not overlap
  E15 every input file name is readable (<site>_<sensor>_YYYYMMDD.csv)
  W1  share of flag 9 above --max-missing
  W2  missing run longer than 4 h
  W3  error column of the input not 0
  W4  REVIEW lines in processing_log.txt (each must appear in the report)
  W5  T1/T2/T3 outside the physical range (no temperature flag exists)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tms4_flag as tf  # noqa: E402  (same rules and settings as the processing script)

COLUMNS = ['time_utc', 'time_local', 'T1', 'T2', 'T3', 'raw', 'vwc_raw', 'vwc_qc', 'flag']
FLAGS = {0, 1, 3, 4, 8, 9}
NAME = re.compile(r'^(?P<site>[A-Z]\d?)_(?P<sensor>[SL]\d{1,2})_(?P<t0>\d{12})_(?P<t1>\d{12})\.csv$')
VWC_RANGE = (0.0, 60.0)       # %, Lolly volumetric water content; above 60 is not soil water
T_RANGE = (-40.0, 50.0)       # degC, Svalbard/Boseong summer-autumn; outside = sensor fault
TOL = 0.006                   # output is written with 2 decimals
LONG_GAP = 16                 # records (4 h)


class Report:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.n_err = 0
        self.n_warn = 0

    def err(self, code: str, where: str, msg: str) -> None:
        self.n_err += 1
        self.lines.append(f'ERROR {code} {where}: {msg}')

    def warn(self, code: str, where: str, msg: str) -> None:
        self.n_warn += 1
        self.lines.append(f'WARN  {code} {where}: {msg}')


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start index, length) of consecutive True values."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def show_times(t: pd.DatetimeIndex, idx: np.ndarray, k: int = 3) -> str:
    s = ', '.join(f'{x:%m-%d %H:%M}' for x in t[idx][:k])
    return s + (f' ... ({len(idx)} rows)' if len(idx) > k else '')


def check_file(path: Path, inputs: dict[str, pd.DataFrame], rep: Report,
               args: argparse.Namespace) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    m = NAME.match(path.name)
    if not m:
        rep.err('E2', path.name, 'file name is not <site>_<sensor>_<start>_<end>.csv')
        return None
    sensor = m['sensor']
    t0 = pd.to_datetime(m['t0'], format='%Y%m%d%H%M')
    t1 = pd.to_datetime(m['t1'], format='%Y%m%d%H%M')
    df = pd.read_csv(path)
    if list(df.columns) != COLUMNS:
        rep.err('E3', path.name, f'columns {list(df.columns)}')
        return sensor, t0, t1

    t = pd.DatetimeIndex(pd.to_datetime(df['time_utc']))
    tl = pd.DatetimeIndex(pd.to_datetime(df['time_local']))
    if t.has_duplicates:
        rep.err('E4', path.name, f'{int(t.duplicated().sum())} duplicated times')
    step = np.diff(t) if len(t) > 1 else np.array([], 'timedelta64[ns]')
    if (step != tf.STEP).any():
        bad = np.where(step != tf.STEP)[0]
        rep.err('E4', path.name, f'not 15-min continuous after {show_times(t, bad)}')
    if len(t) and (t[0] != t0 or t[-1] != t1):
        rep.err('E4', path.name, f'data {t[0]:%Y%m%d%H%M}-{t[-1]:%Y%m%d%H%M} != file name')
    if not (tl - t == tf.LOCAL_OFFSET).all():
        rep.err('E5', path.name, f'time_local is not UTC{tf.LOCAL_OFFSET / pd.Timedelta("1h"):+g}h')

    f = df['flag'].to_numpy()
    v = df['vwc_raw'].to_numpy(float)
    q = df['vwc_qc'].to_numpy(float)
    if not set(np.unique(f)) <= FLAGS:
        rep.err('E6', path.name, f'unknown flags {sorted(set(np.unique(f)) - FLAGS)}')
    i = np.where((np.isnan(v) | (v <= 0)) & (f != 9))[0]
    if len(i):
        rep.err('E7', path.name, f'missing/zero VWC not flagged 9 at {show_times(t, i)}')
    i = np.where((f == 9) & ~np.isnan(q))[0]
    if len(i):
        rep.err('E8', path.name, f'flag 9 with vwc_qc at {show_times(t, i)}')
    # restored segments from CONTACT; flag 1 (visit) rows inside them keep the restored value
    restored = np.zeros(len(t), bool)
    for drop, rec, *_ in tf.CONTACT.get(sensor, []):
        if rec is not None:
            restored |= np.asarray((t >= pd.Timestamp(drop)) & (t < pd.Timestamp(rec)))
    keep = np.isin(f, (0, 4, 8)) | ((f == 1) & ~restored)
    i = np.where(keep & ~(np.abs(q - v) <= TOL))[0]
    if len(i):
        rep.err('E9', path.name, f'vwc_qc != vwc_raw outside flag 3 at {show_times(t, i)}')
    i = np.where((f == 3) & np.isnan(q))[0]
    if len(i):
        rep.err('E10', path.name, f'flag 3 without restored value at {show_times(t, i)}')

    # E11 raw values unchanged
    src = inputs.get(sensor)
    if src is not None:
        src = src.reindex(t)
        for out_col, in_col in (('T1', 'T1'), ('T2', 'T2'), ('T3', 'T3'), ('vwc_raw', 'vwc')):
            a, b = df[out_col].to_numpy(float), src[in_col].to_numpy(float)
            both = ~np.isnan(a) & ~np.isnan(b)
            i = np.where((both & (np.abs(a - b) > TOL)) | (np.isnan(a) != np.isnan(b)))[0]
            if len(i):
                rep.err('E11', path.name, f'{out_col} differs from input at {show_times(t, i)}')

    # E12 values passed on as usable (vwc_qc, any flag but 9) must be physically possible
    i = np.where((f != 9) & ((q < VWC_RANGE[0]) | (q > VWC_RANGE[1])))[0]
    if len(i):
        rep.err('E12', path.name, f'vwc_qc outside {VWC_RANGE} % without flag 9: '
                f'{show_times(t, i)} (max {np.nanmax(q[i]):.1f}, flags {sorted(int(x) for x in set(f[i]))})')
    # W5 temperatures have no flag of their own: report values outside the range
    for c in ('T1', 'T2', 'T3'):
        x = df[c].to_numpy(float)
        i = np.where((x < T_RANGE[0]) | (x > T_RANGE[1]))[0]
        if len(i):
            rep.warn('W5', path.name, f'{c} outside {T_RANGE} degC at {show_times(t, i)} '
                     f'(min {np.nanmin(x[i]):.1f})')

    # warnings: missing data
    miss = f == 9
    if len(f) and miss.mean() > args.max_missing:
        rep.warn('W1', path.name, f'flag 9 = {miss.mean():.1%} of {len(f)} rows')
    for s, n in runs(miss):
        if n > LONG_GAP:
            rep.warn('W2', path.name, f'missing {n} records ({n / 4:g} h) from {t[s]:%m-%d %H:%M} UTC')
    if src is not None and (src['err'].fillna(0) != 0).any():
        i = np.where(src['err'].fillna(0).to_numpy() != 0)[0]
        rep.warn('W3', path.name, f'input error column not 0 at {show_times(t, i)}')
    return sensor, t0, t1


def main() -> int:
    ap = argparse.ArgumentParser(description='Check tms4_flag.py output')
    ap.add_argument('in_dir', type=Path)
    ap.add_argument('out_dir', type=Path)
    ap.add_argument('--max-missing', type=float, default=0.05, help='flag-9 share that raises W1')
    args = ap.parse_args()
    rep = Report()

    # inputs, read with the same reader as the processing script
    for p in args.in_dir.rglob('*.csv'):
        if args.out_dir.resolve() not in p.resolve().parents and tf.sensor_from_name(p) is None:
            rep.err('E15', p.name, 'input file name not readable -> skipped by tms4_flag.py')
    files = tf.collect_inputs(args.in_dir, args.out_dir)
    inputs: dict[str, pd.DataFrame] = {}
    for s, paths in files.items():
        try:
            inputs[s] = tf.read_sensor(paths)
        except Exception as e:  # noqa: BLE001
            rep.err('E15', s, f'cannot read input ({type(e).__name__}: {e})')
        if s not in tf.DEPLOY:
            rep.err('E13', s, f'sensor not in DEPLOY ({", ".join(p.name for p in paths)}); '
                    'ask for its site and period, do not guess')

    if not (args.out_dir / 'processing_log.txt').exists():
        rep.err('E1', str(args.out_dir), 'processing_log.txt missing')
    outs = sorted(p for p in args.out_dir.glob('*.csv'))
    spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for p in outs:
        r = check_file(p, inputs, rep, args)
        if r:
            spans.setdefault(r[0], []).append((r[1], r[2], p.name))

    for s in sorted(set(inputs) & set(tf.DEPLOY)):
        if s not in spans:
            rep.err('E14', s, 'input present but no output file')
    for s, v in spans.items():
        v.sort()
        for (a0, a1, an), (b0, b1, bn) in zip(v, v[1:]):
            if b0 <= a1:
                rep.err('E14', s, f'{an} and {bn} overlap')

    log = args.out_dir / 'processing_log.txt'
    n_review = sum('REVIEW' in ln for ln in log.read_text(encoding='utf-8').splitlines()) \
        if log.exists() else 0
    if n_review:
        rep.warn('W4', 'processing_log.txt', f'{n_review} REVIEW lines -> list them under 확인 필요')

    head = [f'check_result.py  inputs {sum(len(v) for v in files.values())} files / '
            f'{len(files)} sensors, outputs {len(outs)} files',
            f'ERROR {rep.n_err}, WARN {rep.n_warn}  ->  '
            + ('FAIL: fix and run again' if rep.n_err else 'PASS'), '']
    text = '\n'.join(head + rep.lines)
    if args.out_dir.exists():
        (args.out_dir / 'check_report.txt').write_text(text + '\n', encoding='utf-8')
    print(text)
    return 1 if rep.n_err else 0


if __name__ == '__main__':
    sys.exit(main())

"""TMS4 soil moisture QC (2026 Svalbard, 2026-07-10 to 2026-08-12 UTC).

Rules follow tms4-download-check in brlee-0/Observation_TMS4.
  Flags: 0 good, 1 during a TMS download visit, 3 contact disturbance then recovered
         (restored values used), 4 contact disturbance without recovery,
         8 spike, 9 missing (NaN, zero value, sensor out of the ground)
  Priority: 9 > 1 > 8 > 4 > 3 > 0
Time: TMS files are in UTC; the field log is in local time (UTC+2).
      Output carries both UTC and local time (UTC+2).
VWC: Lolly soil moisture conversion with the default soil type, applied at all sites.
Raw values are never changed. Decisions go to `flag`; restored values only to `vwc_qc`.

Usage: python tms4_svalbard_qc_en.py <TMS CSV folder> <output folder>
Input file names: <site>_<sensor>_YYYYMMDD.csv (e.g. A_S1_20260807.csv) or <sensor>_MMDD_MMDD.csv.
Subfolders are searched; multiple downloads of the same sensor are merged by time
(on overlap, the latest file wins). The site in the file name is ignored; the site
comes from the DEPLOY table (e.g. A_S1 data before 8/1 belong to C5).
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = '2026-10-20'

# ---------------------------------------------------------------- basic settings
COLS = ['idx', 'time', 'T1', 'T2', 'T3', 'raw', 'vwc', 'err']
DTYPES = {'idx': 'int64', 'time': 'string', 'T1': 'float64', 'T2': 'float64',
          'T3': 'float64', 'raw': 'float64', 'vwc': 'float64', 'err': 'int64'}
STEP = pd.Timedelta('15min')
LOCAL_OFFSET = pd.Timedelta(hours=2)            # field-log time = UTC + 2 (CEST)
WINDOW = (pd.Timestamp('2026-07-10 00:00'), pd.Timestamp('2026-08-12 23:45'))  # UTC
DROP_THR = 0.5          # 15-min drop threshold for review (%p, SKILL.md)
SPIKE_THR = 1.0         # single-record spike threshold (%p)
VISIT_DEFAULT = pd.Timedelta('10min')           # assumed visit length when departure is not logged
N_EDGE = 4              # records (1 h) whose median sets each end of a restored segment
PRIORITY = (9, 1, 8, 4, 3)                      # earlier = higher priority

# ------------------------------------------------- 2026 Svalbard campaign settings
# Deployment periods per sensor (UTC, first/last in-ground record confirmed from data).
# (site, start, end). Relocations within a site (S5/L4 at A, L6 at C2) are split too.
DEPLOY = {
    'S1':  [('C5', '2026-07-11 14:15', '2026-08-01 13:30'),
            ('A',  '2026-08-01 15:15', '2026-08-07 15:15')],
    'L10': [('C5', '2026-07-11 14:15', '2026-08-01 13:30'),
            ('A',  '2026-08-01 15:15', '2026-08-07 15:15')],
    'S5':  [('A',  '2026-07-13 10:15', '2026-07-20 10:45'),
            ('A',  '2026-07-20 11:00', '2026-08-07 15:15')],
    'L4':  [('A',  '2026-07-13 10:15', '2026-07-20 10:45'),   # 10:00 recorded during insertion
            ('A',  '2026-07-20 11:15', '2026-08-07 15:15')],
    'S3':  [('B1', '2026-07-12 13:45', '2026-08-07 15:45')],
    'L12': [('B1', '2026-07-12 14:00', '2026-08-07 15:45')],
    'L9':  [('B1', '2026-07-26 17:30', '2026-08-07 15:45')],
    'L1':  [('B2', '2026-07-12 10:30', '2026-08-07 13:45')],
    'S11': [('B2', '2026-07-12 10:30', '2026-08-07 13:45')],
    'S4':  [('C1', '2026-07-10 14:00', '2026-08-07 12:45')],
    'L3':  [('C1', '2026-07-10 14:00', '2026-08-07 12:45')],
    'S6':  [('C2', '2026-07-10 14:30', '2026-08-07 13:00')],
    'L6':  [('C2', '2026-07-10 14:30', '2026-07-18 14:15'),   # relocated after reindeer damage on 7/18
            ('C2', '2026-07-18 15:15', '2026-08-07 13:00')],
    'S8':  [('C2', '2026-07-18 15:30', '2026-08-07 13:00')],
    'L8':  [('C2', '2026-07-18 15:30', '2026-08-07 13:00')],
    'S2':  [('C3', '2026-07-10 15:15', '2026-08-07 13:15')],
    'L11': [('C3', '2026-07-10 15:30', '2026-08-07 13:15')],
    'S7':  [('C4', '2026-07-10 16:15', '2026-08-07 14:00')],
    'L2':  [('C6', '2026-07-12 15:45', '2026-08-07 15:00')],   # 15:30 recorded during insertion
    'S12': [('C6', '2026-07-12 15:30', '2026-08-07 15:00')],
    'S9':  [('C7', '2026-07-18 16:00', '2026-08-07 15:15')],
    'L5':  [('C7', '2026-07-18 16:00', '2026-08-07 15:15')],
    'S10': [('C8', '2026-07-18 16:15', '2026-08-07 15:30')],
    'L7':  [('C8', '2026-07-18 16:30', '2026-08-07 15:30')],
}

# Installation heights/depths (cm, relative to ground): T3, T2, T1. From field log site_list
HEIGHTS_DEFAULT = {'S': (16.0, 4.0, -6.0), 'L': (14.5, -6.0, -16.5)}
HEIGHTS = {('L10', 'C5'): (18.5, -2.0, -12.5), ('L11', 'C3'): (18.5, -2.0, -12.5),
           ('L9', 'B1'): (24.0, 3.5, -7.0)}

# Visits with TMS data download (local time, arrival-departure).
# Departure None -> arrival + VISIT_DEFAULT
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
    'C1': [('07-15 16:50', '07-15 17:00'), ('07-19 14:03', '07-19 14:12'),
           ('07-23 16:33', '07-23 16:39'), ('07-28 17:20', '07-28 17:24'),
           ('08-01 14:25', '08-01 14:31'), ('08-04 14:11', '08-04 14:17'),
           ('08-07 14:41', '08-07 14:51')],
    'C2': [('07-15 17:10', '07-15 17:20'), ('07-18 16:58', '07-18 17:23'),
           ('07-19 14:20', '07-19 14:30'), ('07-23 16:45', '07-23 16:53'),
           ('07-28 17:02', '07-28 17:12'), ('08-01 14:40', '08-01 14:46'),
           ('08-04 14:25', None), ('08-07 14:58', '08-07 15:06')],
    'C3': [('07-15 14:25', '07-15 14:40'), ('07-19 14:50', '07-19 14:58'),
           ('07-23 17:11', '07-23 17:16'), ('07-28 16:40', '07-28 16:45'),
           ('08-01 15:08', '08-01 15:15'), ('08-04 14:56', '08-04 14:58'),
           ('08-07 15:24', '08-07 15:30')],
    'C4': [('07-15 12:20', '07-15 12:30'), ('07-19 16:45', '07-19 16:49'),
           ('07-23 18:15', '07-23 18:21'), ('07-28 15:40', '07-28 15:43'),
           ('08-01 16:16', '08-01 16:19'), ('08-04 15:59', '08-04 16:01'),
           ('08-07 15:59', '08-07 16:03')],
    'C5': [('07-15 13:55', '07-15 14:05'), ('07-19 15:10', '07-19 15:20'),
           ('07-23 17:30', '07-23 17:35'), ('07-28 16:22', '07-28 16:25'),
           ('08-01 15:15', '08-01 15:44')],
    'C6': [('07-15 16:30', '07-15 16:40'), ('07-19 17:26', '07-19 17:33'),
           ('07-23 18:46', '07-23 18:50'), ('07-28 17:27', None),
           ('08-01 16:35', '08-01 16:39'), ('08-04 14:03', '08-04 14:07'),
           ('08-07 17:07', '08-07 17:11')],
    'C7': [('07-19 18:25', '07-19 18:34'), ('07-23 19:22', '07-23 19:32'),
           ('07-28 14:10', '07-28 14:12'), ('08-01 16:45', '08-01 16:51'),
           ('08-04 16:20', None), ('08-07 17:26', '08-07 17:33')],
    'C8': [('07-19 18:16', '07-19 18:20'), ('07-23 19:13', '07-23 19:17'),
           ('07-28 13:55', '07-28 14:00'), ('08-01 17:15', '08-01 17:19'),
           ('08-04 16:48', '08-04 16:51'), ('08-07 17:37', '08-07 17:45')],
}

# Contact disturbances (UTC).
# (first dropped record, first record after recovery or None, evidence[, inner steps])
# None -> flag 4 until the end of that deployment; otherwise flag 3 and restore.
# Inner steps: records inside the segment with a further contact jump (partial recovery such as
# a soil refill, or a further drop); these jumps are removed before restoring.
CONTACT = {
    'S11': [('2026-07-15 11:15', '2026-07-30 15:45',
             '7/15 drop 12.8->8.7 during B2 visit, 7/19 urine on S11 cap, '
             '7/30 17:42 (local) soil refilled, 12.4->15.1; further drops inside the segment '
             'at 7/19 visit (-1.7), 7/23 visit (-0.7) and 7/27 (-0.7) removed as steps',
             ('2026-07-19 14:30', '2026-07-23 16:00', '2026-07-27 14:00')),
            ('2026-08-01 14:15', None,
             '8/1 16:03 (local) visit and tripod removal, drop 15.9->14.8, no recovery logged')],
    'S2': [('2026-07-26 03:45', '2026-07-30 15:00',
            '7/26 early-morning drop 33.0->29.0 without a visit, 7/28 urine on S2 cap, '
            '7/30 16:54 (local) soil refilled, 28.9->31.8'),
           ('2026-08-02 15:15', None,
            '8/2 17:06 (local) visit found urine on S2 again, drop 31.3->30.6, no recovery logged')],
    'L11': [('2026-07-15 12:45', '2026-07-23 15:15',
             '7/15 drop 6.6->3.6 during C3 visit (Hydrosense comparison), '
             'recovered at 7/23 visit (one-record jump to 8.3, then 6.9)'),
            ('2026-07-27 22:00', '2026-08-01 19:00',
             '7/28 00:00 (local) drop 6.9->3.2 without a visit, urine on S2 found on 7/28, '
             'partial recovery at 7/30 soil refill (+1.8) and 8/1 visit (+0.8, urine on whole L11), '
             'level back to 5.9 on 8/1 19:00',
             ('2026-07-30 15:00', '2026-08-01 13:15'))],
    'L6': [('2026-07-10 14:30', None,
            'stuck at 36.1-37.1 with repeated zeros from installation, no response to 7/13 rain, '
            'large mismatch with Hydrosense on 7/15 and 7/18, dug up by reindeer on 7/18 '
            '-> relocated within the site')],
}


# ---------------------------------------------------------------- functions
def parse_time(s: pd.Series) -> pd.DatetimeIndex:
    """Parse 'YYYY.MM.DD HH:MM'; also handle Lolly's Chinese AM/PM format and other formats."""
    s = s.str.strip()
    out = pd.to_datetime(s, format='%Y.%m.%d %H:%M', errors='coerce')
    ampm = s.str.contains('上午|下午', regex=True, na=False)       # Chinese AM / PM
    if ampm.any():
        t = s[ampm].str.replace('上午', 'AM', regex=False).str.replace('下午', 'PM', regex=False)
        out[ampm] = pd.to_datetime(t, format='%Y/%m/%d %p %I:%M:%S')
    rest = out.isna() & s.notna()
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], format='mixed')
    return pd.DatetimeIndex(out)


def read_tms(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';', header=None, names=COLS, dtype=DTYPES,
                     encoding='cp950', encoding_errors='replace')
    df.index = parse_time(df.pop('time'))
    df.index.name = 'time_utc'
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df.loc[WINDOW[0]:WINDOW[1]]


def local_to_utc(s: str, year: int = 2026) -> pd.Timestamp:
    return pd.Timestamp(f'{year}-{s}') - LOCAL_OFFSET


def visit_mask(t: pd.DatetimeIndex, site: str) -> np.ndarray:
    """From arrival to the first record after departure (departure rounded up to 15 min)."""
    m = np.zeros(len(t), bool)
    for a, b in VISITS.get(site, []):
        t0 = local_to_utc(a)
        t1 = (local_to_utc(b) if b else t0 + VISIT_DEFAULT).ceil(STEP)
        m |= (t >= t0) & (t <= t1)
    return m


def spike_mask(v: pd.Series) -> np.ndarray:
    """A single record that jumps and immediately returns (zeros and NaN excluded)."""
    v = v.where(v > 0)
    d_prev, d_next = v.diff(), v.diff(-1)
    return ((d_prev.abs() >= SPIKE_THR) & (d_next.abs() >= SPIKE_THR)
            & (np.sign(d_prev) == np.sign(d_next))).to_numpy()


def restore(v: pd.Series, drop: pd.Timestamp, recover: pd.Timestamp,
            steps: tuple[str, ...] = ()) -> pd.Series:
    """Dropped segment = anomaly (segment end-to-end line removed)
    + line from the level just before the drop to the level just after recovery.
    Each level is the median of N_EDGE records, so a one-record jump does not set it.
    Inner step changes listed in `steps` are removed first."""
    seg = v.where(v > 0).loc[drop:recover - STEP].interpolate(limit_area='inside')
    for ts in map(pd.Timestamp, steps):
        seg.loc[ts:] -= seg.loc[ts] - seg.loc[ts - STEP]
    x = np.linspace(0.0, 1.0, len(seg))
    a, b = seg.iloc[:N_EDGE].median(), seg.iloc[-N_EDGE:].median()
    pre = v.where(v > 0).loc[:drop - STEP].dropna().iloc[-N_EDGE:].median()
    post = v.where(v > 0).loc[recover:].dropna().iloc[:N_EDGE].median()
    return seg - (a + (b - a) * x) + (pre + (post - pre) * x)


def heights(sensor: str, site: str) -> tuple[float, float, float]:
    return HEIGHTS.get((sensor, site), HEIGHTS_DEFAULT[sensor[0]])


def process_deployment(sensor: str, df: pd.DataFrame, site: str,
                       start: str, end: str) -> tuple[pd.DataFrame, dict]:
    start = pd.Timestamp(start)
    end = min(pd.Timestamp(end), df.index.max())          # files downloaded just before removal
    grid = pd.date_range(start, end, freq=STEP, name='time_utc')
    span = df.loc[start:end]
    off_grid = int((~span.index.isin(grid)).sum())       # records off the 15-min grid (e.g. 1-min)
    d = span.reindex(grid)
    t, v = d.index, d['vwc']

    masks = {f: np.zeros(len(t), bool) for f in PRIORITY}
    masks[9] = (v.isna() | (v <= 0)).to_numpy()
    masks[1] = visit_mask(t, site)
    masks[8] = spike_mask(v)
    vwc_qc = v.copy()
    events = []
    for drop, rec, why, *steps in CONTACT.get(sensor, []):
        drop = pd.Timestamp(drop)
        if not start <= drop <= end:
            continue
        if rec is None:
            masks[4] |= np.asarray(t >= drop)
            events.append(dict(flag=4, start=drop, end=end, why=why, corr=None))
        else:
            rec = pd.Timestamp(rec)
            fixed = restore(v, drop, rec, steps[0] if steps else ())
            vwc_qc.loc[fixed.index] = fixed
            masks[3] |= np.asarray((t >= drop) & (t < rec))
            corr = (fixed - v.loc[fixed.index]).dropna()
            events.append(dict(flag=3, start=drop, end=rec - STEP, why=why,
                               corr=(corr.iloc[0], corr.iloc[-1])))

    flag = np.zeros(len(t), np.int8)
    for f in reversed(PRIORITY):                          # overwrite from lowest priority up
        flag[masks[f]] = f

    out = pd.DataFrame({
        'time_local': t + LOCAL_OFFSET,
        'T1': d['T1'], 'T2': d['T2'], 'T3': d['T3'],
        'raw': d['raw'].round().astype('Int64'), 'vwc_raw': v,
        'vwc_qc': vwc_qc.where(flag != 9),
        'flag': flag,
    }, index=t)

    dv = v.where(v > 0).diff()
    # Drops during a visit (flag 1) are reviewed too, since the lowered level persists.
    # Jumps up during a visit are reviewed as possible recovery of sensor contact.
    # Segments already judged as 3 or 4 (including the drop record) are excluded.
    judged = masks[3] | masks[4]
    review = dv[(((dv <= -DROP_THR) & np.isin(flag, (0, 1)))
                 | ((dv >= DROP_THR) & (flag == 1))) & ~judged]
    info = dict(sensor=sensor, site=site, start=start, end=end, n=len(out),
                missing=int(d['vwc'].isna().sum()), off_grid=off_grid,
                counts=out['flag'].value_counts().sort_index().to_dict(),
                events=events, review=review, heights=heights(sensor, site))
    return out, info


NAME_NEW = re.compile(r'(?:^|[_-])[A-Z]\d?_([SL]\d{1,2})_(\d{8})$')    # A_S1_20260807
NAME_OLD = re.compile(r'(?:^|[_-])([SL]\d{1,2})_\d{4}_(\d{4})$')       # S1_0710_1016


def sensor_from_name(path: Path) -> tuple[str, str] | None:
    """(sensor, download date YYYYMMDD), or None if the name does not match."""
    if m := NAME_NEW.search(path.stem):
        return m.group(1), m.group(2)
    if m := NAME_OLD.search(path.stem):
        return m.group(1), f'{WINDOW[0]:%Y}{m.group(2)}'
    return None


def collect_inputs(in_dir: Path, out_dir: Path) -> dict[str, list[Path]]:
    """Input files per sensor (oldest download -> latest). The output folder is skipped."""
    found: dict[str, list[tuple[str, Path]]] = {}
    for path in in_dir.rglob('*.csv'):
        if out_dir.resolve() in path.resolve().parents:
            continue
        key = sensor_from_name(path)
        if key is None:
            print(f'[skipped] {path.name}: file name does not match')
            continue
        found.setdefault(key[0], []).append((key[1], path))
    return {s: [p for _, p in sorted(v)] for s, v in found.items()}


def read_sensor(paths: list[Path]) -> pd.DataFrame:
    """Merge multiple downloads; for the same timestamp keep the latest file's value."""
    df = pd.concat([read_tms(p) for p in paths])
    return df[~df.index.duplicated(keep='last')].sort_index()


def fmt_t(ts: pd.Timestamp) -> str:
    return f'{ts:%m-%d %H:%M} UTC ({ts + LOCAL_OFFSET:%m-%d %H:%M} local)'


def main() -> None:
    ap = argparse.ArgumentParser(description='TMS4 QC flagging (2026 Svalbard)')
    ap.add_argument('in_dir', type=Path)
    ap.add_argument('out_dir', type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    log = [f'TMS4 QC  version {VERSION}  run {datetime.now():%Y-%m-%d %H:%M}',
           f'Period (UTC) {WINDOW[0]:%Y-%m-%d %H:%M} to {WINDOW[1]:%Y-%m-%d %H:%M}, '
           f'field log and time_local = UTC+2',
           f'Flag priority {PRIORITY}, drop review {DROP_THR}%p, spike {SPIKE_THR}%p',
           'VWC is the Lolly-converted value (%), default Lolly soil type at all sites.', '']
    infos = []
    inputs = collect_inputs(args.in_dir, args.out_dir)
    for sensor in sorted(inputs, key=lambda s: (s[0], int(s[1:]))):
        if sensor not in DEPLOY:
            print(f'[skipped] {sensor}: not in DEPLOY')
            continue
        df = read_sensor(inputs[sensor])
        source = ', '.join(p.name for p in inputs[sensor])
        for site, start, end in DEPLOY[sensor]:
            out, info = process_deployment(sensor, df, site, start, end)
            name = f'{site}_{sensor}_{info["start"]:%Y%m%d%H%M}_{info["end"]:%Y%m%d%H%M}.csv'
            out.to_csv(args.out_dir / name, float_format='%.2f')
            info.update(file=name, source=source)
            infos.append(info)
    missing = sorted(set(DEPLOY) - set(inputs))
    if missing:
        log.append(f'Sensors without input files: {", ".join(missing)}')
        log.append('')

    for i in infos:
        c = ', '.join(f'flag{k}={v}' for k, v in i['counts'].items())
        h = '/'.join(f'{x:g}' for x in i['heights'])
        log.append(f'{i["file"]}  (source {i["source"]}, T3/T2/T1 {h} cm)')
        log.append(f'    {i["n"]} rows, missing rows {i["missing"]}, '
                   f'off-grid records {i["off_grid"]} | {c}')
        for e in i['events']:
            corr = (f', correction {e["corr"][0]:+.1f}->{e["corr"][1]:+.1f}%p' if e['corr'] else '')
            log.append(f'    flag{e["flag"]} {fmt_t(e["start"])} to {fmt_t(e["end"])}{corr}: {e["why"]}')
        for ts, dv in i['review'].items():
            kind = 'drop' if dv < 0 else 'rise during visit'
            log.append(f'    REVIEW {kind} {fmt_t(ts)} {dv:+.1f}%p')
    text = '\n'.join(log)
    (args.out_dir / 'processing_log.txt').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

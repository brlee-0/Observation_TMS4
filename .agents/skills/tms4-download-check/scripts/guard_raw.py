"""PreToolUse hook for tms4-download-check: block writes and deletes on raw data.

Reads the tool call as JSON on stdin ({"tool_name": ..., "tool_input": {...}}).
Exit 0 = allow, exit 2 = block (the message on stderr goes back to the agent).

Raw data = anything under TMS4_RAW_DIR (default data/raw), TMS files named
<site>_<sensor>_YYYYMMDD.csv, and field logs (*.xlsx).
"""
import json
import os
import re
import sys
from pathlib import Path

RAW_DIR = Path(os.environ.get('TMS4_RAW_DIR', 'data/raw')).resolve()
RAW_NAME = re.compile(r'(^|[/\\\s"\'])([A-Z]\d?_)?[SL]\d{1,2}_\d{8}\.csv\b|\.xlsx\b', re.I)
CHANGE = {'rm', 'mv', 'rmdir', 'unlink', 'truncate', 'shred', 'tee', 'touch'}


def is_raw(path: str) -> bool:
    p = Path(path).resolve()
    return p == RAW_DIR or RAW_DIR in p.parents or bool(RAW_NAME.search(p.name))


def main() -> int:
    call = json.load(sys.stdin)
    tool, arg = call.get('tool_name', ''), call.get('tool_input', {}) or {}
    if tool in ('Write', 'Edit', 'MultiEdit', 'NotebookEdit'):
        path = arg.get('file_path') or arg.get('notebook_path') or ''
        if path and is_raw(path):
            print(f'blocked: {path} is raw TMS4 data or a field log. Raw data are read only; '
                  'write results to the output folder.', file=sys.stderr)
            return 2
    if tool == 'Bash':
        cmd = arg.get('command', '')
        bad = [t for t in re.findall(r'>{1,2}\s*([^\s;&|]+)', cmd) if is_raw(t)]   # redirect onto raw
        for part in re.split(r'[;&|]+', cmd):
            words = part.split()
            if not words:
                continue
            verb = Path(words[0]).name
            inplace = verb in ('sed', 'perl') and any(w.startswith('-i') for w in words)
            if verb in CHANGE or inplace:
                bad += [w for w in words[1:] if not w.startswith('-') and is_raw(w.strip('"\''))]
            elif verb == 'cp' and len(words) > 2 and is_raw(words[-1].strip('"\'')):
                bad.append(words[-1])                                       # copy onto raw
        if bad:
            print(f'blocked: command may change or delete raw data: {cmd[:200]}', file=sys.stderr)
            return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())

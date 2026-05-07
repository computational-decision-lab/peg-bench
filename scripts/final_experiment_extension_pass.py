#!/usr/bin/env python3
"""Final experiment extension pass for PEG-Bench.

This script is intentionally conservative: it searches and normalizes existing
artifacts, runs only benchmark-native/lite code when official implementations
are unavailable, and never labels lite/proxy/smoke evidence as official.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path('/root/prediction-execution-gap')
TMP = Path('/root/autodl-tmp')
OUT_DEFAULT = REPO / 'outputs' / 'final_experiment_extension_pass'
P0P1 = REPO / 'outputs' / 'final_guaranteed_evidence_pass'
PYTHON = Path('/root/miniconda3/bin/python')
PANELS = ['time_h5', 'time_h20', 'time_h50', 'volume_h5', 'volume_h20', 'volume_h50']
PANEL_GPU = {
    'time_h5': '0',
    'time_h20': '1',
    'time_h50': '2',
    'volume_h5': '3',
    'volume_h20': '4',
    'volume_h50': '5',
}
PANEL_REPORTS = {
    'time_h5': TMP / 'peg_bench_runs/tardis_90d_time_h5_table2_compare_v1/benchmark_report.json',
    'time_h20': TMP / 'peg_bench_runs/tardis_90d_time_table2_compare_v1/benchmark_report.json',
    'time_h50': TMP / 'peg_bench_runs/tardis_90d_time_h50_table2_compare_v1/benchmark_report.json',
    'volume_h5': TMP / 'peg_bench_runs/tardis_90d_volume_h5_table2_compare_v1/benchmark_report.json',
    'volume_h20': TMP / 'peg_bench_runs/final_90d_missing_panel_runs/protocol_clean_volume_h20_seq96/benchmark_report.json',
    'volume_h50': TMP / 'peg_bench_runs/tardis_90d_volume_h50_table2_compare_v1/benchmark_report.json',
}
TARGET_BASELINE_DIRS = [
    TMP / 'pegbench_ddl_sprint_v25/target_lite_time',
    TMP / 'pegbench_ddl_sprint_v25/target_lite_volume',
    TMP / 'peg_bench_runs/reviewer_target_baselines_20260504_203852',
    TMP / 'peg_bench_runs/reviewer_completion_bundle_20260504_222828/target_baselines',
    TMP / 'pegbench_ddl_sprint_v25/real_gpu_target_baselines',
]
KEYWORDS = {
    'tabl': ['tabl_lite', 'tabl-lite', 'tabllite', 'temporal attention bilinear', 'official TABL', 'Temporal Attention Bilinear'],
    'itransformer': ['itransformer', 'inverted transformer'],
    'timesfm': ['timesfm', 'times-fm', 'google timesfm'],
    'kronos': ['kronos'],
    'ppo_sac': ['ppo', 'sac', 'stable-baselines', 'stable_baselines', 'gymnasium'],
}


def now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def mkdirs(out: Path) -> None:
    for sub in [
        '', 'logs', 'blocker_reports', 'pids',
        'e1_tabl_official_or_blocker',
        'e2_itransformer_official_or_blocker',
        'e3_timesfm_kronos_feasibility',
        'e4_ppo_sac_feasibility',
        'e5_canonical_asset_venue',
        'e6_seed_split_extension',
    ]:
        (out / sub).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline='', encoding='utf-8', errors='ignore') as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    with path.open(encoding='utf-8', errors='ignore') as f:
        return json.load(f)


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + '\n', encoding='utf-8')


def sh(cmd: list[str] | str, *, cwd: Path = REPO, env: dict[str, str] | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True, timeout=timeout, shell=isinstance(cmd, str))


def run_logged(cmd: list[str] | str, log: Path, *, cwd: Path = REPO, env: dict[str, str] | None = None, timeout: int | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8', errors='ignore') as f:
        f.write(f'[{now()}] START cmd={cmd}\n')
        f.flush()
        try:
            proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, text=True, shell=isinstance(cmd, str))
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = 124
            f.write(f'[{now()}] TIMEOUT after {timeout}s\n')
        except Exception as exc:
            rc = 125
            f.write(f'[{now()}] EXCEPTION {type(exc).__name__}: {exc}\n')
        f.write(f'[{now()}] END exit_code={rc}\n')
    return int(rc)


def append_exit(out: Path, task: str, rc: int, status: str, notes: str = '') -> None:
    path = out / 'task_exit_codes.csv'
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        fields = ['time', 'task', 'exit_code', 'status', 'notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({'time': now(), 'task': task, 'exit_code': rc, 'status': status, 'notes': notes})


def blocker(out: Path, task: str, reason: str, attempted: list[str], last_error: str = '', next_fix: str = '') -> None:
    write_md(
        out / 'blocker_reports' / f'{task}_blocker.md',
        '# Blocker Report\n\n'
        f'- task: {task}\n'
        f'- time: {now()}\n'
        f'- reason: {reason}\n'
        f'- last_error: {last_error or "n/a"}\n'
        f'- next_fix: {next_fix or "n/a"}\n\n'
        '## Attempted Fixes / Checks\n'
        + ''.join(f'- {a}\n' for a in attempted)
    )


def find_files(roots: list[Path], patterns: list[str], max_hits: int = 500) -> list[Path]:
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            try:
                hits.extend(root.rglob(pat))
            except Exception:
                continue
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in hits:
        s = str(p)
        if s not in seen:
            seen.add(s)
            uniq.append(p)
    uniq.sort(key=lambda p: (p.stat().st_mtime if p.exists() else 0, str(p)), reverse=True)
    return uniq[:max_hits]


def grep_keywords(roots: list[Path], terms: list[str], max_hits: int = 400) -> list[dict[str, str]]:
    pattern = '|'.join(re.escape(t) for t in terms)
    paths = [str(r) for r in roots if r.exists()]
    if not paths:
        return []
    rows: list[dict[str, str]] = []
    rg_path = shutil.which('rg')
    if rg_path:
        proc = sh([rg_path, '-n', '-i', pattern, *paths], timeout=60)
        for line in (proc.stdout or '').splitlines()[:max_hits]:
            parts = line.split(':', 2)
            if len(parts) == 3:
                rows.append({'path': parts[0], 'line': parts[1], 'text': parts[2][:500]})
        return rows

    # Fallback for minimal remote images without ripgrep. Keep this bounded so
    # feasibility checks do not hang on large binary artifacts.
    regex = re.compile(pattern, re.IGNORECASE)
    allowed_suffixes = {'.py', '.sh', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini'}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob('*'):
            if len(rows) >= max_hits:
                return rows
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            try:
                with path.open(encoding='utf-8', errors='ignore') as f:
                    for idx, line in enumerate(f, start=1):
                        if regex.search(line):
                            rows.append({'path': str(path), 'line': str(idx), 'text': line.strip()[:500]})
                            if len(rows) >= max_hits:
                                return rows
            except Exception:
                continue
    return rows


def sha256_manifest(out: Path) -> None:
    rows: list[str] = []
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name == 'sha256_manifest.txt':
            continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        rows.append(f'{h.hexdigest()}  {p.relative_to(out)}')
    write_md(out / 'sha256_manifest.txt', '\n'.join(rows))


def gpu_snapshot() -> str:
    proc = sh('nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits || true', timeout=20)
    return proc.stdout.strip() or proc.stderr.strip()


def preflight(out: Path) -> None:
    mkdirs(out)
    write_md(out / 'experiment_extension_plan.md', '''# Experiment Extension Plan

Scope: high-value extension pass after completed P0/P1 evidence bundle. This pass reads `outputs/final_guaranteed_evidence_pass/` and writes only `outputs/final_experiment_extension_pass/`.

## Order
- E1 official TABL feasibility, and official run only if a real official implementation plus adapter exists.
- E2 official iTransformer feasibility, and official run only if a real official implementation plus adapter exists.
- E3 TimesFM/Kronos feasibility-first; run only if model, checkpoint, adapter already exist.
- E4 PPO/SAC feasibility-first; run only if env/checkpoint/config already exist.
- E5 canonical asset/venue grouped evaluator upgrade if metadata supports a non-fabricated join.
- E6 seed/split extension from existing sidecars only unless cheap configs are already ready.

## Guardrails
- Lite/proxy/smoke rows are supplemental only.
- Target-only items are not completed claims.
- No P0/P1 rerun and no formal-method rescue.
- Every task writes log, source trace, status, and blocker when needed.
''')

    env_lines = ['# Environment Check', f'- time: {now()}', f'- cwd: {REPO}', f'- python: {PYTHON} exists={PYTHON.exists()}']
    for cmd in ['git rev-parse HEAD || true', 'git status --short || true', f'{PYTHON} --version || true', 'which python || true', 'which rg || true']:
        proc = sh(cmd, timeout=20)
        env_lines.append(f'\n## `{cmd}`\n```\n{(proc.stdout + proc.stderr).strip()[:4000]}\n```')
    write_md(out / 'environment_check.md', '\n'.join(env_lines))

    hw_lines = ['# Available GPU / CPU / Memory', f'- time: {now()}', '\n## GPU', '```', gpu_snapshot(), '```']
    for cmd in ['lscpu | sed -n "1,24p"', 'free -h', 'df -h / /root/autodl-tmp']:
        proc = sh(cmd, timeout=20)
        hw_lines.append(f'\n## `{cmd}`\n```\n{(proc.stdout + proc.stderr).strip()[:4000]}\n```')
    write_md(out / 'available_gpu_cpu_memory.md', '\n'.join(hw_lines))

    script_names = [
        'scripts/run_reviewer_target_baselines.py',
        'scripts/run_peg_bench_comparison_baselines.py',
        'scripts/run_benchmark_evaluator.py',
        'scripts/final_guaranteed_evidence_pass.py',
        'scripts/hft_execution_audit.py',
        'scripts/run_final_experiment_extension_pass.sh',
        'scripts/final_experiment_extension_pass.py',
    ]
    script_lines = ['# Script Existence Check']
    for name in script_names:
        p = REPO / name
        script_lines.append(f'- `{name}`: exists={p.exists()} size={p.stat().st_size if p.exists() else ""}')
    for k, terms in KEYWORDS.items():
        hits = grep_keywords([REPO / 'scripts', REPO / 'src', REPO / 'configs'], terms, max_hits=80)
        script_lines.append(f'\n## Keyword `{k}` hits: {len(hits)}')
        for row in hits[:30]:
            script_lines.append(f'- `{row["path"]}:{row["line"]}` {row["text"]}')
    write_md(out / 'script_existence_check.md', '\n'.join(script_lines))

    inv: list[dict[str, Any]] = []
    for panel, path in PANEL_REPORTS.items():
        inv.append({'type': 'canonical_benchmark_report', 'panel': panel, 'path': str(path), 'exists': path.exists(), 'size_bytes': path.stat().st_size if path.exists() else '', 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime)) if path.exists() else ''})
    for sub in ['p0_main_results', 'p0_gap_metrics', 'p0_cost_ladder', 'p0_bootstrap_ci', 'p0_wrapper_matrix', 'p0_formal_boundary', 'p1_per_asset_venue', 'p1_hft_audit', 'p1_stability']:
        root = P0P1 / sub
        for p in sorted(root.rglob('*')) if root.exists() else []:
            if p.is_file():
                inv.append({'type': sub, 'panel': '', 'path': str(p), 'exists': True, 'size_bytes': p.stat().st_size, 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))})
    for p in find_files([TMP / 'pegbench_ddl_sprint_v25', TMP / 'peg_bench_runs'], ['results_table.csv', 'results_table.json', 'benchmark_report.json'], 500):
        if any(tok in str(p).lower() for tok in ['target', 'elasticnet', 'real_gpu', 'seed', 'tabl', 'itransformer', 'per_asset', 'venue']):
            inv.append({'type': 'existing_target_or_sidecar_artifact', 'panel': '', 'path': str(p), 'exists': True, 'size_bytes': p.stat().st_size, 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime))})
    write_csv(out / 'p0_p1_input_inventory.csv', inv)
    if not (out / 'task_exit_codes.csv').exists():
        write_csv(out / 'task_exit_codes.csv', [], ['time', 'task', 'exit_code', 'status', 'notes'])
    write_progress(out, 'preflight generated; E1/E2 not yet launched')


def existing_lite_artifacts(method: str) -> list[Path]:
    m = method.lower()
    roots = TARGET_BASELINE_DIRS + [REPO / 'outputs', TMP / 'peg_bench_runs']
    candidates = find_files(roots, ['results_table.csv', 'results_table.json', 'benchmark_report.json'], 800)
    keep: list[Path] = []
    for p in candidates:
        s = str(p).lower()
        if method == 'tabl' and ('target_lite' in s or 'reviewer_target' in s or 'real_gpu_target' in s or 'target_baselines' in s):
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore').lower()[:200000]
            except Exception:
                txt = ''
            if 'tabl_lite' in txt or 'tabl-lite' in txt:
                keep.append(p)
        elif method == 'itransformer' and ('target_lite' in s or 'reviewer_target' in s or 'real_gpu_target' in s or 'target_baselines' in s):
            try:
                txt = p.read_text(encoding='utf-8', errors='ignore').lower()[:200000]
            except Exception:
                txt = ''
            if 'itransformer_lite' in txt or 'itransformer-lite' in txt:
                keep.append(p)
    return keep


def official_hits(method: str) -> list[dict[str, str]]:
    terms = KEYWORDS[method]
    hits = grep_keywords([REPO / 'scripts', REPO / 'src', REPO / 'configs', REPO / 'baselines', REPO / 'third_party'], terms, max_hits=500)
    filtered: list[dict[str, str]] = []
    for row in hits:
        text = (row['path'] + ' ' + row['text']).lower()
        if method == 'tabl':
            if 'results_table' in text or 'table2' in text or 'table' in text and 'tabl' not in text:
                continue
            if 'tabl_lite' in text or 'tabl-lite' in text or 'target baselines' in text or 'benchmark-native' in text:
                continue
            if 'official' not in text and 'github' not in text and 'repo' not in text:
                continue
        if method == 'itransformer':
            if 'itransformer_lite' in text or 'itransformer-lite' in text or 'target baselines' in text or 'benchmark-native' in text:
                continue
            if 'official' not in text and 'github' not in text and 'repo' not in text and 'thuml' not in text:
                continue
        filtered.append(row)
    return filtered


def parse_result_rows(path: Path, target_method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        if path.suffix == '.csv':
            loaded_obj: Any = read_csv(path)
        else:
            obj = read_json(path)
            if isinstance(obj, list):
                loaded_obj = obj
            elif isinstance(obj, dict):
                for key in ('methods', 'rows', 'results', 'leaderboard'):
                    if isinstance(obj.get(key), list):
                        loaded_obj = obj[key]
                        break
                else:
                    loaded_obj = []
            else:
                loaded_obj = []
    except Exception:
        return rows
    if not isinstance(loaded_obj, list):
        return rows
    for row in loaded_obj:
        if not isinstance(row, dict):
            continue
        method = str(row.get('method') or row.get('name') or '')
        if target_method == 'tabl' and method != 'tabl_lite':
            continue
        if target_method == 'itransformer' and method != 'itransformer_lite':
            continue
        panel = ''
        for p in PANELS:
            if f'/{p}/' in str(path) or str(path).endswith(f'/{p}_results_table.csv') or str(path).endswith(f'/{p}_results_table.json'):
                panel = p
                break
        out_row = dict(row)
        out_row['panel'] = panel
        out_row['source_artifact'] = str(path)
        out_row['completion_label'] = 'supplemental_lite_not_official'
        rows.append(out_row)
    return rows


def normalize_lite(out: Path, method: str, dst: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    artifacts = existing_lite_artifacts(method)
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    for art in artifacts:
        parsed = parse_result_rows(art, method)
        if parsed:
            rows.extend(parsed)
            trace.append({'method_family': method, 'artifact_type': 'existing_lite_result', 'source_artifact': str(art), 'rows_used': len(parsed), 'status': 'supplemental_lite_not_official'})
    # de-duplicate by panel/method/source, keeping all unique panels but not repeated completion bundles where possible
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (str(row.get('panel', '')), str(row.get('method', '')), str(row.get('source_artifact', '')))
        if key not in seen:
            seen.add(key)
            dedup.append(row)
    rows = dedup
    if method == 'tabl':
        write_csv(dst / 'tabl_panel_results.csv', rows)
        write_csv(dst / 'tabl_normalized_r0_r4_rows.csv', rows)
        write_csv(dst / 'tabl_source_trace.csv', trace)
    else:
        write_csv(dst / 'itransformer_panel_results.csv', rows)
        write_csv(dst / 'itransformer_normalized_r0_r4_rows.csv', rows)
        write_csv(dst / 'itransformer_source_trace.csv', trace)
    return rows, trace


def task_tabl(out: Path) -> int:
    dst = out / 'e1_tabl_official_or_blocker'
    dst.mkdir(parents=True, exist_ok=True)
    official = official_hits('tabl')
    lite = existing_lite_artifacts('tabl')
    rows, trace = normalize_lite(out, 'tabl', dst)
    status = 'target_only_blocker'
    reason = 'No official TABL implementation plus PEG-Bench adapter found in repo; only benchmark-native tabl_lite artifacts/scripts are present.'
    if official:
        reason = 'Potential TABL keyword hits exist, but no verified official TABL entrypoint/adapter was found by conservative filter. Manual review required before official run.'
    if rows:
        status = 'supplemental_lite_not_official'
    report = [
        '# TABL Feasibility Report',
        f'- time: {now()}',
        f'- official_keyword_hits_after_lite_filter: {len(official)}',
        f'- existing_lite_artifacts: {len(lite)}',
        f'- normalized_lite_rows: {len(rows)}',
        f'- decision: {status}',
        '',
        '## Official Decision',
        reason,
        '',
        '## Official Hit Samples',
    ]
    for row in official[:30]:
        report.append(f'- `{row["path"]}:{row["line"]}` {row["text"]}')
    report += ['', '## Lite Artifact Samples']
    for p in lite[:40]:
        report.append(f'- `{p}`')
    write_md(dst / 'tabl_feasibility_report.md', '\n'.join(report))
    write_csv(dst / 'tabl_run_manifest.csv', [{'method_family': 'TABL', 'status': status, 'official_run_started': False, 'gpu_used': False, 'source': 'repo/artifact search', 'notes': reason}])
    write_md(dst / 'tabl_completion_status.md', f'# TABL Completion Status\n\n- status: `{status}`\n- official_completed: false\n- supplemental_lite_rows: {len(rows)}\n- can_claim_official_completed: false\n')
    if status == 'target_only_blocker':
        blocker(out, 'e1_tabl', reason, ['Searched scripts/src/configs/baselines/third_party for TABL', 'Searched /root/autodl-tmp target baseline artifacts', 'Inspected run_reviewer_target_baselines.py and confirmed tabl_lite naming'], next_fix='Add official TABL repo reproduction or approved adapter, then run six-panel normalized benchmark.')
    return 0


def task_itransformer(out: Path) -> int:
    dst = out / 'e2_itransformer_official_or_blocker'
    dst.mkdir(parents=True, exist_ok=True)
    official = official_hits('itransformer')
    lite = existing_lite_artifacts('itransformer')
    rows, trace = normalize_lite(out, 'itransformer', dst)
    status = 'target_only_blocker'
    reason = 'No official iTransformer implementation plus PEG-Bench adapter found in repo; only benchmark-native itransformer_lite artifacts/scripts are present.'
    if official:
        reason = 'Potential iTransformer keyword hits exist, but no verified official iTransformer entrypoint/adapter was found by conservative filter. Manual review required before official run.'
    if rows:
        status = 'supplemental_lite_not_official'
    report = [
        '# iTransformer Feasibility Report',
        f'- time: {now()}',
        f'- official_keyword_hits_after_lite_filter: {len(official)}',
        f'- existing_lite_artifacts: {len(lite)}',
        f'- normalized_lite_rows: {len(rows)}',
        f'- decision: {status}',
        '',
        '## Official Decision',
        reason,
        '',
        '## Official Hit Samples',
    ]
    for row in official[:30]:
        report.append(f'- `{row["path"]}:{row["line"]}` {row["text"]}')
    report += ['', '## Lite Artifact Samples']
    for p in lite[:40]:
        report.append(f'- `{p}`')
    write_md(dst / 'itransformer_feasibility_report.md', '\n'.join(report))
    write_csv(dst / 'itransformer_run_manifest.csv', [{'method_family': 'iTransformer', 'status': status, 'official_run_started': False, 'gpu_used': False, 'source': 'repo/artifact search', 'notes': reason}])
    write_md(dst / 'itransformer_completion_status.md', f'# iTransformer Completion Status\n\n- status: `{status}`\n- official_completed: false\n- supplemental_lite_rows: {len(rows)}\n- can_claim_official_completed: false\n')
    if status == 'target_only_blocker':
        blocker(out, 'e2_itransformer', reason, ['Searched scripts/src/configs/baselines/third_party for iTransformer', 'Searched /root/autodl-tmp target baseline artifacts', 'Inspected run_reviewer_target_baselines.py and confirmed itransformer_lite naming'], next_fix='Add official iTransformer repo reproduction or approved adapter, then run six-panel normalized benchmark.')
    return 0


def task_foundation(out: Path) -> int:
    dst = out / 'e3_timesfm_kronos_feasibility'
    dst.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    smoke: list[dict[str, Any]] = []
    for model in ['timesfm', 'kronos']:
        hits = grep_keywords([REPO / 'scripts', REPO / 'src', REPO / 'configs', REPO / 'baselines', REPO / 'third_party'], KEYWORDS[model], max_hits=120)
        files = find_files([REPO, TMP], [f'*{model}*', f'*{model.capitalize()}*'], 200)
        has_adapter = bool(hits)
        has_checkpoint = any(('ckpt' in str(p).lower() or 'checkpoint' in str(p).lower() or p.suffix in {'.pt', '.pth', '.bin'}) for p in files)
        status = 'target_only_blocker'
        if has_adapter and has_checkpoint:
            status = 'smoke_ready_manual_review_required'
        rows.append({'model': model, 'status': status, 'keyword_hits': len(hits), 'candidate_files': len(files), 'has_adapter': has_adapter, 'has_checkpoint': has_checkpoint, 'run_started': False})
        trace.append({'model': model, 'source': 'repo/tmp search', 'status': status, 'notes': 'No smoke launched unless installed checkpoint and adapter are both verified safe.'})
        md = [f'# {model} Feasibility Report', f'- time: {now()}', f'- status: `{status}`', f'- keyword_hits: {len(hits)}', f'- candidate_files: {len(files)}', f'- has_adapter: {has_adapter}', f'- has_checkpoint: {has_checkpoint}', '', '## Hit Samples']
        for h in hits[:30]:
            md.append(f'- `{h["path"]}:{h["line"]}` {h["text"]}')
        md += ['', '## File Samples']
        for p in files[:40]:
            md.append(f'- `{p}`')
        write_md(dst / f'{model}_feasibility_report.md', '\n'.join(md))
        if status == 'target_only_blocker':
            blocker(out, f'e3_{model}', f'{model} lacks verified local checkpoint+adapter for normalized benchmark rows.', ['Searched repo scripts/src/configs/baselines/third_party', 'Searched /root/autodl-tmp for model/checkpoint filenames'], next_fix=f'Install/approve {model} adapter and local checkpoint, then run one-panel smoke before six-panel benchmark.')
    write_csv(dst / 'foundation_model_smoke_results.csv', smoke, ['model', 'panel', 'status', 'source_artifact', 'notes'])
    write_csv(dst / 'foundation_model_normalized_rows.csv', [], ['model', 'panel', 'status', 'source_artifact', 'notes'])
    write_csv(dst / 'foundation_model_source_trace.csv', trace)
    return 0


def task_ppo_sac(out: Path) -> int:
    dst = out / 'e4_ppo_sac_feasibility'
    dst.mkdir(parents=True, exist_ok=True)
    hits = grep_keywords([REPO / 'scripts', REPO / 'src', REPO / 'configs'], KEYWORDS['ppo_sac'], max_hits=200)
    files = find_files([REPO, TMP], ['*ppo*', '*sac*', '*stable*'], 300)
    artifact_rows: list[dict[str, Any]] = []
    for p in files:
        artifact_rows.append({'path': str(p), 'size_bytes': p.stat().st_size if p.exists() and p.is_file() else '', 'mtime': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.stat().st_mtime)) if p.exists() else ''})
    has_env = bool(hits)
    has_checkpoint = any((p.suffix in {'.pt', '.pth', '.zip'} or 'checkpoint' in str(p).lower()) for p in files)
    status = 'target_only_blocker'
    if has_env and has_checkpoint:
        status = 'existing_artifact_manual_normalization_needed'
    md = ['# PPO/SAC Feasibility Report', f'- time: {now()}', f'- status: `{status}`', f'- keyword_hits: {len(hits)}', f'- candidate_files: {len(files)}', f'- has_env_or_adapter: {has_env}', f'- has_checkpoint: {has_checkpoint}', '- full_training_started: false', '', '## Hit Samples']
    for h in hits[:40]:
        md.append(f'- `{h["path"]}:{h["line"]}` {h["text"]}')
    write_md(dst / 'ppo_sac_feasibility_report.md', '\n'.join(md))
    write_csv(dst / 'ppo_sac_existing_artifact_inventory.csv', artifact_rows)
    write_csv(dst / 'ppo_sac_smoke_results.csv', [], ['method', 'panel', 'status', 'source_artifact', 'notes'])
    write_csv(dst / 'ppo_sac_normalized_rows.csv', [], ['method', 'panel', 'status', 'source_artifact', 'notes'])
    write_csv(dst / 'ppo_sac_source_trace.csv', [{'method_family': 'PPO/SAC', 'status': status, 'source': 'repo/tmp search', 'notes': 'No from-scratch >8h training launched.'}])
    if status == 'target_only_blocker':
        blocker(out, 'e4_ppo_sac', 'No ready canonical PPO/SAC env plus checkpoint/config found; from-scratch training likely high-risk and >8h.', ['Searched PPO/SAC/stable-baselines keywords', 'Searched checkpoints/artifacts'], next_fix='Add ready RL env/checkpoint or approve long training feasibility separately.')
    return 0


def task_asset_venue(out: Path) -> int:
    dst = out / 'e5_canonical_asset_venue'
    dst.mkdir(parents=True, exist_ok=True)
    p1 = P0P1 / 'p1_per_asset_venue'
    source_files = [p1 / 'per_asset_breakdown.csv', p1 / 'per_venue_breakdown.csv', p1 / 'per_asset_venue_breakdown.csv', p1 / 'per_asset_venue_audit.md']
    trace: list[dict[str, Any]] = []
    status = 'supplemental_fast_replay_retained'
    metadata_hits = grep_keywords([REPO / 'scripts', REPO / 'src'], ['grouped', 'stratified', 'asset_key', 'venue', 'per_asset'], max_hits=160)
    can_canonical = any('stratified' in h['text'].lower() or 'grouped' in h['text'].lower() for h in metadata_hits)
    # Conservative: do not run unless an explicit canonical grouped evaluator script exists.
    explicit = [h for h in metadata_hits if 'canonical' in h['text'].lower() and ('group' in h['text'].lower() or 'strat' in h['text'].lower())]
    if explicit:
        status = 'candidate_canonical_evaluator_manual_review_required'
    for src in source_files:
        if src.exists():
            dst_name = src.name.replace('per_asset', 'canonical_per_asset').replace('per_venue', 'canonical_per_venue')
            # If not canonical, copy as trace/reference only with status in source trace, not as upgraded canonical claim.
            shutil.copy2(src, dst / dst_name)
            trace.append({'status': status, 'source_artifact': str(src), 'copied_to': str(dst / dst_name), 'notes': 'P1 source retained; not upgraded unless explicit canonical evaluator is verified.'})
    for required in ['canonical_per_asset_breakdown.csv', 'canonical_per_venue_breakdown.csv', 'canonical_per_asset_venue_breakdown.csv']:
        p = dst / required
        if not p.exists():
            write_csv(p, [], ['status', 'source_artifact', 'notes'])
    write_csv(dst / 'canonical_asset_venue_source_trace.csv', trace)
    write_md(dst / 'canonical_asset_venue_status.md', f'# Canonical Asset/Venue Status\n\n- status: `{status}`\n- metadata_keyword_hits: {len(metadata_hits)}\n- explicit_canonical_candidates: {len(explicit)}\n- canonical_upgrade_completed: false\n- claim_boundary: retain P1 as supplemental fast replay unless manual review approves a canonical grouped evaluator.\n')
    if status != 'candidate_canonical_evaluator_manual_review_required':
        blocker(out, 'e5_canonical_asset_venue', 'No explicit canonical grouped evaluator path verified; not safe to upgrade supplemental fast replay to canonical leaderboard.', ['Read P1 per-asset/venue outputs', 'Searched grouped/stratified/asset_key/venue code paths'], next_fix='Implement/approve canonical grouped evaluator using source metadata/action paths, then rerun grouped report.')
    return 0


def task_seed_split(out: Path) -> int:
    dst = out / 'e6_seed_split_extension'
    dst.mkdir(parents=True, exist_ok=True)
    p1 = P0P1 / 'p1_stability'
    trace: list[dict[str, Any]] = []
    copied = 0
    for name in ['seed_stability_headline.csv', 'split_stability_headline.csv', 'stability_source_trace.csv', 'stability_summary.md', 'stability_table.tex']:
        src = p1 / name
        if src.exists():
            shutil.copy2(src, dst / name.replace('headline', 'extended'))
            copied += 1
            trace.append({'source_artifact': str(src), 'copied_to': str(dst / name.replace('headline', 'extended')), 'status': 'existing_sidecar_only'})
    for required in ['seed_stability_extended.csv', 'split_stability_extended.csv']:
        p = dst / required
        if not p.exists():
            write_csv(p, [], ['status', 'source_artifact', 'notes'])
    write_csv(dst / 'seed_split_source_trace.csv', trace)
    write_md(dst / 'seed_split_extension_plan.md', '# Seed/Split Extension Plan\n\nDefault execution is existing sidecar only. No new seed sweep/model retraining was launched in this extension pass.\n')
    write_md(dst / 'seed_split_status.md', f'# Seed/Split Status\n\n- status: `existing_sidecar_only`\n- copied_sidecar_files: {copied}\n- new_training_started: false\n- claim_boundary: robustness appendix only; frozen headline unchanged.\n')
    return 0


def write_progress(out: Path, message: str) -> None:
    exits = read_csv(out / 'task_exit_codes.csv') if (out / 'task_exit_codes.csv').exists() else []
    lines = ['# Progress Live', f'- time: {now()}', f'- message: {message}', '', '## GPU Snapshot', '```', gpu_snapshot(), '```', '', '## Task Exit Codes']
    for row in exits[-20:]:
        lines.append(f'- {row.get("time")} `{row.get("task")}` exit={row.get("exit_code")} status={row.get("status")} {row.get("notes", "")}')
    write_md(out / 'progress_live.md', '\n'.join(lines))


def latest_task_statuses(out: Path) -> list[dict[str, str]]:
    rows = read_csv(out / 'task_exit_codes.csv')
    latest: dict[str, dict[str, str]] = {}
    for row in rows:
        task = row.get('task', '')
        if task:
            latest[task] = row
    ordered = ['preflight', 'e1_tabl', 'e2_itransformer', 'e3_timesfm_kronos', 'e4_ppo_sac', 'e5_asset_venue', 'e6_seed_split', 'final_reports']
    return [latest[t] for t in ordered if t in latest]


def final_reports(out: Path) -> None:
    statuses = read_csv(out / 'task_exit_codes.csv')
    status_map = {r.get('task', ''): r for r in statuses}
    current_rows = latest_task_statuses(out)
    write_csv(out / 'current_task_status.csv', current_rows, ['time', 'task', 'exit_code', 'status', 'notes'])
    status_map = {r.get('task', ''): r for r in current_rows}
    summary = ['# Final Experiment Extension Summary', f'- time: {now()}', '', '## Current Task Status (latest attempt per task)']
    for task in ['e1_tabl', 'e2_itransformer', 'e3_timesfm_kronos', 'e4_ppo_sac', 'e5_asset_venue', 'e6_seed_split']:
        r = status_map.get(task, {})
        summary.append(f'- {task}: status={r.get("status", "not_run")} exit_code={r.get("exit_code", "")}; {r.get("notes", "")}')
    summary += ['', '## Notes on Repaired Attempts', '- Earlier runner failures in `task_exit_codes.csv` were caused by missing `rg` and parser incompatibility with dict-shaped JSON; both were fixed, and current status is summarized in `current_task_status.csv`.', '', '## Claim Upgrade Guidance', '- Upgrade only artifacts labeled official_completed or completed_normalized_rows.', '- TABL/iTransformer lite rows remain supplemental if official entrypoints were not verified.', '- TimesFM/Kronos/PPO/SAC remain target-only unless normalized rows exist.', '- Per-asset/venue remains supplemental unless canonical grouped evaluator is verified.', '- Seed/split sidecars remain appendix robustness only.', '', '## Need Paper Claim Sync?', '- Yes, rerun paper claim sync only if this pass produced official_completed or completed_normalized_rows; otherwise keep existing claim boundaries.']
    write_md(out / 'final_extension_summary.md', '\n'.join(summary))

    cvst = ['# Completed vs Supplemental vs Target Extension', f'- time: {now()}', '', '| item | status | claim boundary |', '|---|---|---|']
    for item, status, boundary in [
        ('official TABL 90D', _status_from_file(out / 'e1_tabl_official_or_blocker/tabl_completion_status.md'), 'official only if status official_completed; lite is supplemental'),
        ('official iTransformer 90D', _status_from_file(out / 'e2_itransformer_official_or_blocker/itransformer_completion_status.md'), 'official only if status official_completed; lite is supplemental'),
        ('TimesFM/Kronos', 'see E3 reports', 'target-only/blocker unless normalized rows exist'),
        ('PPO/SAC', 'see E4 reports', 'target-only/blocker unless normalized rows exist'),
        ('per-asset/venue canonical', 'see E5 status', 'supplemental fast replay unless canonical grouped evaluator completed'),
        ('seed/split extension', 'existing_sidecar_only', 'appendix robustness only'),
    ]:
        cvst.append(f'| {item} | {status} | {boundary} |')
    write_md(out / 'completed_vs_supplemental_vs_target_extension.md', '\n'.join(cvst))

    write_md(out / 'repaired_attempts.md', '# Repaired Attempts\n\n- First runner attempt failed because remote image lacked `rg`; helper now falls back to Python grep.\n- Second E1/E2 attempt failed on dict-shaped JSON parsing; parser now skips non-row objects and supports common dict list keys.\n- Current status should be read from `current_task_status.csv`; historical failures remain in logs for auditability.\n')
    write_md(out / 'paper_claim_update_recommendation.md', '# Paper Claim Update Recommendation\n\n- Do not upgrade official TABL/iTransformer/TimesFM/Kronos/PPO/SAC unless corresponding normalized rows are present and status is official/completed.\n- If E1/E2 only normalized lite rows, mention them only as supplemental benchmark-native lite baselines.\n- Do not move audit-only/supplemental rows into the main leaderboard.\n- Re-run claim sync only after checking `task_exit_codes.csv` and source traces.\n')
    sha256_manifest(out)
    tar_path = REPO / 'outputs' / 'final_experiment_extension_pass.tar.gz'
    sha_path = REPO / 'outputs' / 'final_experiment_extension_pass.sha256'
    if tar_path.exists():
        tar_path.unlink()
    rc = run_logged(f'tar -czf {tar_path} -C {out.parent} {out.name}', out / 'logs' / 'archive.log', timeout=3600)
    if rc == 0:
        h = hashlib.sha256()
        with tar_path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        write_md(sha_path, f'{h.hexdigest()}  {tar_path.name}')


def _status_from_file(path: Path) -> str:
    if not path.exists():
        return 'not_run'
    text = path.read_text(encoding='utf-8', errors='ignore')
    m = re.search(r'status:\s*`?([^`\n]+)`?', text)
    return m.group(1).strip() if m else 'see_report'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(OUT_DEFAULT))
    parser.add_argument('--task', required=True, choices=['preflight', 'e1_tabl', 'e2_itransformer', 'e3_timesfm_kronos', 'e4_ppo_sac', 'e5_asset_venue', 'e6_seed_split', 'final_reports'])
    args = parser.parse_args()
    out = Path(args.out)
    mkdirs(out)
    if args.task == 'preflight':
        preflight(out)
        append_exit(out, 'preflight', 0, 'completed', 'inventory/environment generated')
    elif args.task == 'e1_tabl':
        rc = task_tabl(out)
        append_exit(out, 'e1_tabl', rc, _status_from_file(out / 'e1_tabl_official_or_blocker/tabl_completion_status.md'), 'official feasibility plus lite normalization if available')
        write_progress(out, 'E1 TABL completed')
    elif args.task == 'e2_itransformer':
        rc = task_itransformer(out)
        append_exit(out, 'e2_itransformer', rc, _status_from_file(out / 'e2_itransformer_official_or_blocker/itransformer_completion_status.md'), 'official feasibility plus lite normalization if available')
        write_progress(out, 'E2 iTransformer completed')
    elif args.task == 'e3_timesfm_kronos':
        rc = task_foundation(out)
        append_exit(out, 'e3_timesfm_kronos', rc, 'feasibility_completed', 'foundation feasibility only')
        write_progress(out, 'E3 foundation feasibility completed')
    elif args.task == 'e4_ppo_sac':
        rc = task_ppo_sac(out)
        append_exit(out, 'e4_ppo_sac', rc, 'feasibility_completed', 'RL feasibility only')
        write_progress(out, 'E4 PPO/SAC feasibility completed')
    elif args.task == 'e5_asset_venue':
        rc = task_asset_venue(out)
        append_exit(out, 'e5_asset_venue', rc, 'feasibility_completed', 'canonical upgrade check')
        write_progress(out, 'E5 asset/venue canonical check completed')
    elif args.task == 'e6_seed_split':
        rc = task_seed_split(out)
        append_exit(out, 'e6_seed_split', rc, 'existing_sidecar_only', 'no new seed sweep')
        write_progress(out, 'E6 seed/split sidecar completed')
    elif args.task == 'final_reports':
        final_reports(out)
        append_exit(out, 'final_reports', 0, 'completed', 'summary/archive generated')
        write_progress(out, 'final reports generated')


if __name__ == '__main__':
    main()

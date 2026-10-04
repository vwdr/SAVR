#!/usr/bin/env python3
"""Freeze the bounded two-stage launch: write both immutable configs with final
SHA-256 for the executed source chain, schedules and caps. Deterministic; the
printed digest block is what the readiness report pins. Run only after all
worker/analyzer/contract/tests/protocol files are final.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import run_adaptive_qualification as s0
import run_adaptive_screen as s1
from savr.openvla.adaptive_screen_contract import (ARMS, CAPS, TRIAGE, ADAPTIVE_ZERO_TOLERANCE_MAX,
                                                   episode_slots, timing_slots)

FIXED_QUAL = ROOT / 'configs/openvla/fixed_compression_qualification_v1.json'
FIXED_SCREEN = ROOT / 'configs/openvla/fixed_compression_screen_v1.json'
REFERENCE_EVAL = ROOT / 'configs/openvla/contemporary_reference_evaluation_v2.json'
BASELINE = ROOT / 'configs/openvla/original_baseline_40task_v1.json'

QUALIFICATION_SUMMARY_SHA = 'ad197c96e75d8c6c753d4c786c924aa49415d5e5632873ff81fcbe9a197ca5a9'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


fixed_qual = json.loads(FIXED_QUAL.read_text())
fixed_screen = json.loads(FIXED_SCREEN.read_text())
reference_eval = json.loads(REFERENCE_EVAL.read_text())
baseline = json.loads(BASELINE.read_text())

s0_config = dict(
    schema_version='adaptive-qualification-v1', launch_ready=True, caps=s0.CAPS,
    automatic_retry=False, arms=list(s0.MODES), fastv_ratios=s0.MODES,
    tolerance=1e-6, adaptive_zero_tolerance_max=1e-3,
    output_root='results/adaptive-qualification-v03',
    gpu=reference_eval['gpu'], observation_ids=fixed_qual['observation_ids'],
    reference_summary=fixed_qual['reference_summary'],
    reference_summary_sha256=fixed_qual['reference_summary_sha256'],
    input_preflight_report=reference_eval['input_preflight_report'],
    worker_sha256=sha(ROOT / 'scripts/run_adaptive_qualification.py'),
    analyzer_sha256=sha(ROOT / 'scripts/analyze_adaptive_qualification.py'),
    authenticated_files={name: sha(ROOT / name) for name in s0.FILES},
)
dump(ROOT / 'configs/openvla/adaptive_qualification_v3.json', s0_config)
adaptive_qualification_sha = sha(ROOT / 'configs/openvla/adaptive_qualification_v3.json')

s1_config = dict(
    schema_version='adaptive-screen-v1', launch_ready=True, caps=CAPS,
    automatic_retry=False, arms=list(ARMS), tolerance=1e-6,
    adaptive_zero_tolerance_max=ADAPTIVE_ZERO_TOLERANCE_MAX,
    output_root='results/adaptive-screen-v03',
    gpu=fixed_qual['gpu'], observation_ids=fixed_qual['observation_ids'],
    initial_state_sha256=fixed_screen['initial_state_sha256'],
    qualification_summary=fixed_qual['output_root'] + '/worker_summary.json',
    qualification_summary_sha256=QUALIFICATION_SUMMARY_SHA,
    adaptive_qualification_summary=s0_config['output_root'] + '/worker_summary.json',
    adaptive_qualification_config_sha256=adaptive_qualification_sha,
    episode_slots=episode_slots(baseline),
    timing_slots=timing_slots(fixed_qual['observation_ids']),
    triage=TRIAGE,
    worker_sha256=sha(ROOT / 'scripts/run_adaptive_screen.py'),
    analyzer_sha256=sha(ROOT / 'scripts/analyze_adaptive_screen.py'),
    authenticated_files={name: sha(ROOT / name) for name in s1.FILES},
)
dump(ROOT / 'configs/openvla/adaptive_screen_v3.json', s1_config)

assert len(s1_config['episode_slots']) == 168 and len(s1_config['timing_slots']) == 240
assert s1_config['initial_state_sha256'] == fixed_screen['initial_state_sha256']
assert sorted(s1_config['authenticated_files']) == sorted(s1.FILES)
assert sorted(s0_config['authenticated_files']) == sorted(s0.FILES)

print(json.dumps(dict(
    adaptive_qualification_v3_sha256=adaptive_qualification_sha,
    adaptive_screen_v3_sha256=sha(ROOT / 'configs/openvla/adaptive_screen_v3.json'),
    freeze_script_sha256=sha(Path(__file__)),
    episode_slots=len(s1_config['episode_slots']),
    timing_slots=len(s1_config['timing_slots']),
    screen_caps=CAPS,
    qualification_caps=s0.CAPS,
    authenticated=dict(s0=s0_config['authenticated_files'], s1=s1_config['authenticated_files']),
), indent=2))

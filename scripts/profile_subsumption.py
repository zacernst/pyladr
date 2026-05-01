#!/usr/bin/env python3
"""Profile subsumption performance during vampire.in execution."""
import time, sys, signal, logging

# Limit search time for profiling
MAX_SECONDS = 30

logging.disable(logging.CRITICAL)

from pyladr.inference import subsumption as sub
from pyladr.inference.subsumption import BackSubsumptionIndex

orig_subsumes = sub.subsumes
orig_forward_subsume = sub.forward_subsume
orig_back_subsume_indexed = sub.back_subsume_indexed
orig_candidates = BackSubsumptionIndex.candidates

stats = {
    'fwd_calls': 0, 'fwd_time': 0.0, 'fwd_found': 0,
    'back_calls': 0, 'back_time': 0.0, 'back_found': 0,
    'sub_calls': 0, 'sub_time': 0.0, 'sub_true': 0,
    'cand_calls': 0, 'cand_total': 0,
    'fwd_sub_calls': 0, 'back_sub_calls': 0,
    'unit_sub': 0, 'nonunit_sub': 0,
    'max_cand_single': 0,
}
_context = [None]
_snapshots = []
_start_time = [0.0]

def snapshot():
    elapsed = time.perf_counter() - _start_time[0]
    _snapshots.append({
        'time': elapsed,
        'fwd_calls': stats['fwd_calls'],
        'fwd_found': stats['fwd_found'],
        'back_calls': stats['back_calls'],
        'back_found': stats['back_found'],
        'sub_calls': stats['sub_calls'],
        'sub_true': stats['sub_true'],
    })

def inst_subsumes(c, d):
    stats['sub_calls'] += 1
    nc = c.num_literals
    if nc == 1: stats['unit_sub'] += 1
    else: stats['nonunit_sub'] += 1
    t0 = time.perf_counter()
    r = orig_subsumes(c, d)
    stats['sub_time'] += time.perf_counter() - t0
    if r: stats['sub_true'] += 1
    if _context[0] == 'fwd': stats['fwd_sub_calls'] += 1
    elif _context[0] == 'back': stats['back_sub_calls'] += 1
    # Periodic snapshot every 1000 calls
    if stats['sub_calls'] % 5000 == 0:
        snapshot()
    return r

def inst_fwd(d, pos_idx, neg_idx):
    stats['fwd_calls'] += 1
    _context[0] = 'fwd'
    t0 = time.perf_counter()
    r = orig_forward_subsume(d, pos_idx, neg_idx)
    stats['fwd_time'] += time.perf_counter() - t0
    _context[0] = None
    if r is not None: stats['fwd_found'] += 1
    return r

def inst_back(c, idx):
    stats['back_calls'] += 1
    _context[0] = 'back'
    t0 = time.perf_counter()
    r = orig_back_subsume_indexed(c, idx)
    stats['back_time'] += time.perf_counter() - t0
    _context[0] = None
    n = len(r)
    stats['back_found'] += n
    if n > stats['max_cand_single']:
        stats['max_cand_single'] = n
    return r

def inst_cand(self, c):
    stats['cand_calls'] += 1
    r = orig_candidates(self, c)
    stats['cand_total'] += len(r)
    return r

sub.subsumes = inst_subsumes
sub.forward_subsume = inst_fwd
sub.back_subsume_indexed = inst_back
BackSubsumptionIndex.candidates = inst_cand

import pyladr.search.given_clause as gc
gc.forward_subsume = inst_fwd
gc.back_subsume_indexed = inst_back
gc.subsumes = inst_subsumes

from pyladr.apps.prover9 import run_prover

_start_time[0] = time.perf_counter()
exit_code = run_prover(['prover9', '-f', '/tmp/vampire_timed.in'])
t_total = time.perf_counter() - _start_time[0]
snapshot()  # Final snapshot

s = stats
print()
print('=' * 70)
print(f'SUBSUMPTION PERFORMANCE PROFILE - vampire.in ({MAX_SECONDS}s limit)')
print('=' * 70)
print(f'Total time:             {t_total:.4f}s')
print(f'Exit code:              {exit_code}')
print()
print('--- Forward Subsumption ---')
print(f'  Calls:                {s["fwd_calls"]:,}')
print(f'  Found (subsumed):     {s["fwd_found"]:,}')
print(f'  Hit rate:             {s["fwd_found"]/max(s["fwd_calls"],1)*100:.1f}%')
print(f'  Total time:           {s["fwd_time"]:.4f}s ({s["fwd_time"]/max(t_total,1e-9)*100:.1f}%)')
print(f'  Avg/call:             {s["fwd_time"]/max(s["fwd_calls"],1)*1e6:.1f} us')
print(f'  subsumes() calls:     {s["fwd_sub_calls"]:,}')
print()
print('--- Backward Subsumption ---')
print(f'  Calls:                {s["back_calls"]:,}')
print(f'  Victims found:        {s["back_found"]:,}')
print(f'  Max single batch:     {s["max_cand_single"]}')
print(f'  Total time:           {s["back_time"]:.4f}s ({s["back_time"]/max(t_total,1e-9)*100:.1f}%)')
print(f'  Avg/call:             {s["back_time"]/max(s["back_calls"],1)*1e6:.1f} us')
print(f'  subsumes() calls:     {s["back_sub_calls"]:,}')
print()
print('--- BackSubsumptionIndex ---')
print(f'  candidates() calls:   {s["cand_calls"]:,}')
print(f'  Total candidates:     {s["cand_total"]:,}')
print(f'  Avg candidates/call:  {s["cand_total"]/max(s["cand_calls"],1):.1f}')
print()
print('--- Core subsumes() ---')
print(f'  Total calls:          {s["sub_calls"]:,}')
print(f'  Unit tests:           {s["unit_sub"]:,}')
print(f'  Non-unit tests:       {s["nonunit_sub"]:,}')
print(f'  Successes:            {s["sub_true"]:,}')
print(f'  Success rate:         {s["sub_true"]/max(s["sub_calls"],1)*100:.1f}%')
print(f'  Total time:           {s["sub_time"]:.4f}s ({s["sub_time"]/max(t_total,1e-9)*100:.1f}%)')
print(f'  Avg/call:             {s["sub_time"]/max(s["sub_calls"],1)*1e6:.1f} us')
print(f'  Non-unit (from module):{sub.nonunit_subsumption_tests():,}')
print()
subsump_total = s['fwd_time'] + s['back_time']
print(f'Combined subsumption:   {subsump_total:.4f}s ({subsump_total/max(t_total,1e-9)*100:.1f}% of total)')
print(f'Non-subsumption:        {t_total - subsump_total:.4f}s ({(t_total - subsump_total)/max(t_total,1e-9)*100:.1f}% of total)')
print()

# Rate analysis
if t_total > 1:
    print('--- Rates ---')
    print(f'  Forward subsumptions/s:  {s["fwd_calls"]/t_total:.0f}')
    print(f'  Backward subsumptions/s: {s["back_calls"]/t_total:.0f}')
    print(f'  subsumes() checks/s:     {s["sub_calls"]/t_total:.0f}')
    print()

# Snapshots over time
if len(_snapshots) > 1:
    print('--- Progression over time ---')
    print(f'  {"Time":>8s}  {"Fwd":>8s}  {"FwdHit":>8s}  {"Back":>8s}  {"BackHit":>8s}  {"SubCalls":>8s}  {"SubTrue":>8s}')
    for snap in _snapshots:
        print(f'  {snap["time"]:8.1f}  {snap["fwd_calls"]:8,}  {snap["fwd_found"]:8,}  {snap["back_calls"]:8,}  {snap["back_found"]:8,}  {snap["sub_calls"]:8,}  {snap["sub_true"]:8,}')

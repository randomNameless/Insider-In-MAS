#!/usr/bin/env python
"""Summarize paired Exp5 CRs and attack effects across seeds/horizons."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--result_dir", default="results/exp5_qwen35_9b_paired_attack_effect_v3")
p.add_argument("--seeds", nargs="+", type=int, default=[9000,9001,9002,9003])
p.add_argument("--benign_counts", nargs="+", type=int, default=[4,5,6,7,8])
p.add_argument("--horizons", nargs="+", type=int, default=[10,15,20,30,40,50])
a = p.parse_args(); root = Path(a.result_dir).expanduser()

def mean_sd(xs):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0

payloads=[]
for seed in a.seeds:
    path=root/f"eval_seed{seed}.json"
    if not path.is_file(): raise SystemExit(f"missing {path}")
    payloads.append(json.loads(path.read_text(encoding="utf-8")))

print("Attack effect = CR(no attack) - CR(PPO); positive means PPO blocks consensus.\n")
print(f"{'B':>2} {'N':>3}  {'no-attack CR mean +/- sd':<27} {'PPO CR mean +/- sd':<23} "
      f"{'attack effect mean +/- sd':<27} {'no-atk pooled':<17} {'PPO pooled'}")
print("-"*135)
for b in a.benign_counts:
  for h in a.horizons:
    nr=[x["summary"][f"B{b}_no_attack"] for x in payloads]
    pr=[x["summary"][f"B{b}_ppo"] for x in payloads]
    ns=[x["consensus_at_horizon"][str(h)]["rate"] for x in nr]
    ps=[x["consensus_at_horizon"][str(h)]["rate"] for x in pr]
    es=[x-y for x,y in zip(ns,ps)]
    nm,nsd=mean_sd(ns); pm,psd=mean_sd(ps); em,esd=mean_sd(es)
    nc=sum(x["consensus_at_horizon"][str(h)]["count"] for x in nr)
    pc=sum(x["consensus_at_horizon"][str(h)]["count"] for x in pr)
    total=sum(x["episodes"] for x in nr)
    print(f"{b:2d} {h:3d}  {nm:.3f} +/- {nsd:.3f}             {pm:.3f} +/- {psd:.3f}         "
          f"{em:+.3f} +/- {esd:.3f}              {nc/total:.3f} ({nc}/{total})  {pc/total:.3f} ({pc}/{total})")

max_h = max(a.horizons)
print(f"\nCensored mean episode length at N={max_h} "
      f"(failures counted as {max_h}):")
print(f"{'B':>2}  {'no attack mean +/- sd':<25} {'PPO mean +/- sd'}")
print("-"*55)
for b in a.benign_counts:
    nl=[x["summary"][f"B{b}_no_attack"]["mean_ep_len"] for x in payloads]
    pl=[x["summary"][f"B{b}_ppo"]["mean_ep_len"] for x in payloads]
    nm,ns=mean_sd(nl); pm,ps=mean_sd(pl)
    print(f"{b:2d}  {nm:.2f} +/- {ns:.2f}              {pm:.2f} +/- {ps:.2f}")

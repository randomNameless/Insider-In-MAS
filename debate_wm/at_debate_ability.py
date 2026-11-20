#!/usr/bin/env python
"""Does adversarial training hurt the honest agents' debate / consensus ability? (2026-11-10)
Post-hoc on the `none` (no attacker) runs: exp10_nn_none (undefended), at_cons_none (AT-SFT), at_dpo_none (AT-DPO);
v1nn, test 400 x seeds 3/4/5. Prints two markdown tables (per-debate means with bootstrap 95% CI; pooled conditionals).
    python debate_wm/at_debate_ability.py > ~/runs/at_debate_ability.md
"""
import json, random
from collections import Counter
N_BOOT = 4000
R = '/path/to/runs'


def load(run, cond):
    st = json.load(open(f'{R}/{run}/state.json'))
    return [d for d in st['debates'] if d['cond'] == cond and d['final'] is not None]


def metrics(d):
    hs = [i for i in range(4) if i != d['seat']]; g = d['gold']
    rounds = [[d['history'][r][i]['answer'] for i in hs] for r in range(3)] + [[x['answer'] for x in d['final']]]
    m = {}
    m['acc_r1'] = sum(a == g for a in rounds[0]) / 3; m['acc_final'] = sum(a == g for a in rounds[3]) / 3
    m['gain_r1_to_final'] = m['acc_final'] - m['acc_r1']
    top = Counter(rounds[3]).most_common(1)[0]
    m['group_correct'] = float(top[1] >= 2 and top[0] == g)
    m['unanimous_final'] = float(len(set(rounds[3])) == 1)
    m['change_rate'] = sum(rounds[r][i] != rounds[r - 1][i] for r in range(1, 4) for i in range(3)) / 9
    fix = [rounds[3][k] == g for k in range(3) if rounds[0][k] != g]
    brk = [rounds[3][k] != g for k in range(3) if rounds[0][k] == g]
    m['n_wrong_r1'] = len(fix); m['fix'] = sum(fix); m['n_corr_r1'] = len(brk); m['break'] = sum(brk)
    fm = [rounds[3][k] == g for k in range(3) if rounds[0][k] != g and all(rounds[0][j] == g for j in range(3) if j != k)]
    hold = [rounds[3][k] == g for k in range(3) if rounds[0][k] == g and all(rounds[0][j] != g for j in range(3) if j != k)]
    m['n_follow'] = len(fm); m['follow'] = sum(fm); m['n_hold'] = len(hold); m['hold'] = sum(hold)
    m['words'] = sum(len(x['argument'].split()) for x in d['final']) / 3
    m['pgold_final'] = sum(p['ABCDEFGHIJ'.index(g)] for p in d['beliefs'][3]) / 3
    return m


def boot_mean(vals):
    rng = random.Random(0); n = len(vals)
    b = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(N_BOOT))
    return sum(vals) / n, b[int(.025 * N_BOOT)], b[int(.975 * N_BOOT) - 1]


runs = {'undefended 9B': load('exp10_nn_none', 'none'), 'AT-SFT (cons)': load('at_cons_none', 'none'), 'AT-DPO': load('at_dpo_none', 'none')}
M = {k: [metrics(d) for d in v] for k, v in runs.items()}
rows = [('final_acc (per-agent)', 'acc_final'), ('acc round 1 (no messages)', 'acc_r1'), ('gain round1→final', 'gain_r1_to_final'),
        ('group acc (majority==gold)', 'group_correct'), ('unanimous at final', 'unanimous_final'), ('answer-change rate', 'change_rate'),
        ('mean P(gold) at final', 'pgold_final'), ('final argument words', 'words')]
print('| metric (none condition, v1nn, test 400 x seeds 3/4/5) | ' + ' | '.join(M) + ' |'); print('|---|' + '---|' * len(M))
for name, key in rows:
    cells = []
    for k, ms in M.items():
        mu, lo, hi = boot_mean([m[key] for m in ms]); cells.append(f"{mu:.3f} [{lo:.3f}, {hi:.3f}]" if key != 'words' else f"{mu:.0f}")
    print(f'| {name} | ' + ' | '.join(cells) + ' |')
print(); print('| conditional (per agent, pooled) | ' + ' | '.join(M) + ' |'); print('|---|' + '---|' * len(M))
for name, num, den in [('P(correct at final | wrong at r1)', 'fix', 'n_wrong_r1'), ('P(wrong at final | correct at r1)', 'break', 'n_corr_r1'),
                       ('P(adopt peers | wrong at r1, both honest peers right)', 'follow', 'n_follow'), ('P(hold | correct at r1, both honest peers wrong)', 'hold', 'n_hold')]:
    cells = []
    for k, ms in M.items():
        a = sum(m[num] for m in ms); b = sum(m[den] for m in ms); cells.append(f"{a / b:.3f} (n={b})")
    print(f'| {name} | ' + ' | '.join(cells) + ' |')

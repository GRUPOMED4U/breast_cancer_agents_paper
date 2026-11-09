"""Judge agreement and stability analysis (#11, options C and D).

Compares the original grades in ``evals/`` (Gemini 2.5 Flash, called "original"
below) with re-grades written by ``regrade.py``:

C. Cross-judge agreement, for each re-grade judge vs the original judge:
   * criterion level: % agreement, Cohen's kappa, Gwet's AC1, overall and by
     criterion sign and category (judge's first repeat);
   * case level: Pearson and Spearman correlation of case scores;
   * configuration level: global score per configuration under each judge,
     Kendall's tau between the configuration rankings, and paired cluster-
     bootstrap differences (other judge - original) per configuration;
   * self-preference: (score on Gemini-generated responses - score on other
     responses) under the original judge minus the same under the other judge.
     A positive value means the Gemini judge favors Gemini responses more than
     the other judge does.
D. Stability, for each judge with >= 2 repeats (for gemini-2.5-flash the
   original grading counts as one repeat, as long as the re-grade used default
   settings):
   * flip rate: share of pairs whose verdict is not identical across repeats;
   * Fleiss' kappa across repeats;
   * SD of case scores and of configuration global scores across repeats,
     compared with the bootstrap SD of the configuration global score (the
     sampling uncertainty reported in the paper).

All confidence intervals are 95% percentile intervals from a bootstrap over
cases (criteria and responses of the same case are resampled together).

The agreement functions (``agreement``, ``gwet_ac1``, ``fleiss_kappa``) are
also meant for the clinician comparison (option B, #14).

Output: validation/results/judge_agreement.json (aggregate numbers only).

Usage:
    .venv/Scripts/python.exe validation/judge_agreement.py [--regrades-dir DIR]
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

from judge_common import OUTPUT_DIR, RESULTS_DIR, SEED, load_evaluated_responses

ORIGINAL = "original"
ORIGINAL_SLUG = "gemini-2.5-flash"
N_BOOT = 2000

Key = tuple[str, str, int]  # (prompt_id, config, criterion_idx)


# ---------------------------------------------------------------- statistics
def gwet_ac1(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet's AC1 for two raters and a binary label."""
    pa = np.mean(a == b)
    pi = (np.mean(a) + np.mean(b)) / 2
    pe = 2 * pi * (1 - pi)
    return float((pa - pe) / (1 - pe)) if pe < 1 else float("nan")


def agreement(a, b) -> dict:
    """Agreement statistics between two binary raters (True = met)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return {"n": 0}
    kappa = cohen_kappa_score(a, b) if len(set(a) | set(b)) > 1 else float("nan")
    return {
        "n": int(len(a)),
        "pct_agreement": float(np.mean(a == b)),
        "cohen_kappa": float(kappa),
        "gwet_ac1": gwet_ac1(a, b),
        "met_rate_a": float(np.mean(a)),
        "met_rate_b": float(np.mean(b)),
        "a_met_b_not": int(np.sum(a & ~b)),
        "a_not_b_met": int(np.sum(~a & b)),
    }


def fleiss_kappa(ratings: np.ndarray) -> float:
    """Fleiss' kappa; ``ratings`` is items x raters, binary."""
    n_raters = ratings.shape[1]
    met = ratings.sum(axis=1)
    counts = np.stack([n_raters - met, met], axis=1)
    p_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / counts.sum()
    pe = np.sum(p_j**2)
    return float((p_bar - pe) / (1 - pe)) if pe < 1 else float("nan")


def boot_ci(stat_fn, case_ids: list[str], rng: np.random.Generator) -> tuple[float, float]:
    """95% CI of ``stat_fn(list_of_case_ids)`` from a bootstrap over cases."""
    vals = []
    for _ in range(N_BOOT):
        sample = [case_ids[i] for i in rng.choice(len(case_ids), len(case_ids))]
        v = stat_fn(sample)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- data
class Grades:
    """All verdicts, keyed by rater label ('original' or '<judge>/rep<k>')."""

    def __init__(self, regrades_dir: Path):
        self.meta: dict[Key, dict] = {}  # points, sign, tags
        self.verdicts: dict[str, dict[Key, bool]] = {}
        self.failures: dict[str, int] = {}
        self.judge_meta: dict[str, dict] = {}

        rows = load_evaluated_responses()
        self.positive_total = {}
        orig = {}
        for r in rows:
            self.positive_total[(r["prompt_id"], r["config"])] = sum(x["points"] for x in r["rubrics"] if x["points"] > 0)
            for idx, (rub, g) in enumerate(zip(r["rubrics"], r["grader_responses"])):
                key = (r["prompt_id"], r["config"], idx)
                self.meta[key] = {
                    "points": rub["points"],
                    "sign": "positive" if rub["points"] > 0 else "negative",
                    "tags": list(rub.get("tags", [])),
                }
                orig[key] = bool(g.get("criteria_met"))
        self.verdicts[ORIGINAL] = orig
        self.criteria_per_response = collections.Counter((k[0], k[1]) for k in self.meta)

        if regrades_dir.exists():
            for jdir in sorted(p for p in regrades_dir.iterdir() if p.is_dir()):
                meta_path = jdir / "meta.json"
                self.judge_meta[jdir.name] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                for rep in sorted(jdir.glob("rep*.jsonl")):
                    label = f"{jdir.name}/{rep.stem}"
                    v, fails = {}, set()
                    with open(rep, encoding="utf-8") as f:
                        for line in f:
                            rec = json.loads(line)
                            key = (str(rec["prompt_id"]), rec["config"], int(rec["criterion_idx"]))
                            if rec.get("criteria_met") is None:
                                fails.add(key)
                            else:
                                v[key] = bool(rec["criteria_met"])
                    self.verdicts[label] = v
                    self.failures[label] = len(fails - set(v))

    def reps(self, judge: str) -> list[str]:
        return sorted((l for l in self.verdicts if l.startswith(judge + "/")), key=lambda l: int(l.split("rep")[-1]))

    def case_scores(self, label: str, keys: set[Key] | None = None) -> dict[tuple[str, str], float]:
        """Unclipped case score per (case, config), only for fully graded responses."""
        v = self.verdicts[label]
        achieved = collections.defaultdict(float)
        n = collections.Counter()
        for key, met in v.items():
            if keys is not None and key not in keys:
                continue
            resp = (key[0], key[1])
            n[resp] += 1
            if met:
                achieved[resp] += self.meta[key]["points"]
        return {
            resp: achieved[resp] / self.positive_total[resp]
            for resp, cnt in n.items()
            if cnt == self.criteria_per_response[resp] and self.positive_total[resp] > 0
        }


def score_matrix(scores: dict[tuple[str, str], float], cfgs: list[str], cases: list[str]) -> np.ndarray:
    m = np.full((len(cfgs), len(cases)), np.nan)
    ci, ki = {c: i for i, c in enumerate(cfgs)}, {c: i for i, c in enumerate(cases)}
    for (pid, cfg), s in scores.items():
        m[ci[cfg], ki[pid]] = s
    return m


def col_means(m: np.ndarray) -> np.ndarray:
    """Global score per configuration (row): mean over cases, clipped to [0, 1]."""
    return np.clip(np.nanmean(m, axis=1), 0, 1)


def is_gemini_response(config: str) -> bool:
    return config.split("::")[1].startswith("gemini")


# ---------------------------------------------------------------- analyses
def cross_judge(g: Grades, judge: str, rng: np.random.Generator) -> dict:
    other = g.reps(judge)[0]
    a, b = g.verdicts[ORIGINAL], g.verdicts[other]
    common = sorted(set(a) & set(b))
    res = {"compared_label": other, "criterion_level": {"overall": agreement([a[k] for k in common], [b[k] for k in common])}}
    for sign in ("positive", "negative"):
        ks = [k for k in common if g.meta[k]["sign"] == sign]
        res["criterion_level"][sign] = agreement([a[k] for k in ks], [b[k] for k in ks])
    tags = sorted({t for k in common for t in g.meta[k]["tags"]})
    res["criterion_level"]["by_category"] = {
        t: agreement([a[k] for k in common if t in g.meta[k]["tags"]], [b[k] for k in common if t in g.meta[k]["tags"]]) for t in tags
    }
    # Criterion-level kappa CI over cases.
    by_case = collections.defaultdict(list)
    for k in common:
        by_case[k[0]].append(k)
    case_ids = sorted(by_case)

    def kappa_on(cases):
        ks = [k for c in cases for k in by_case[c]]
        return cohen_kappa_score([a[k] for k in ks], [b[k] for k in ks])

    res["criterion_level"]["overall"]["cohen_kappa_ci"] = boot_ci(kappa_on, case_ids, rng)

    # Case level.
    common_set = set(common)
    sa, sb = g.case_scores(ORIGINAL, common_set), g.case_scores(other, common_set)
    resp = sorted(set(sa) & set(sb))
    x, y = [sa[r] for r in resp], [sb[r] for r in resp]
    res["case_level"] = {
        "n_responses": len(resp),
        "pearson_r": float(pearsonr(x, y).statistic) if len(resp) > 2 else None,
        "spearman_rho": float(spearmanr(x, y).statistic) if len(resp) > 2 else None,
        "mean_score_original": float(np.mean(x)) if x else None,
        "mean_score_other": float(np.mean(y)) if y else None,
    }

    # Configuration level: config x case score matrices, bootstrap over case columns.
    sa_r = {r: sa[r] for r in resp}
    sb_r = {r: sb[r] for r in resp}
    cfgs = sorted({r[1] for r in resp})
    resp_cases = sorted({r[0] for r in resp})
    ma, mb = score_matrix(sa_r, cfgs, resp_cases), score_matrix(sb_r, cfgs, resp_cases)
    ga, gb = col_means(ma), col_means(mb)
    tau = kendalltau(ga, gb).statistic if len(cfgs) > 2 else float("nan")
    taus, deltas = [], []
    for _ in range(N_BOOT):
        idx = rng.choice(len(resp_cases), len(resp_cases))
        ba, bb = col_means(ma[:, idx]), col_means(mb[:, idx])
        deltas.append(bb - ba)
        if len(cfgs) > 2:
            taus.append(kendalltau(ba, bb).statistic)
    deltas = np.array(deltas)
    per_cfg = {
        c: {
            "original": float(ga[i]),
            "other": float(gb[i]),
            "delta": float(gb[i] - ga[i]),
            "delta_ci": [float(x) for x in np.nanpercentile(deltas[:, i], [2.5, 97.5])],
        }
        for i, c in enumerate(cfgs)
    }
    res["configuration_level"] = {
        "n_configs": len(cfgs),
        "kendall_tau": float(tau),
        "kendall_tau_ci": [float(x) for x in np.nanpercentile(taus, [2.5, 97.5])] if taus else None,
        "per_config": per_cfg,
    }

    # Self-preference (difference in differences).
    def split_by_case(scores):
        d = collections.defaultdict(lambda: ([], []))
        for (pid, cfg), s in scores.items():
            d[pid][0 if is_gemini_response(cfg) else 1].append(s)
        return d

    split_a, split_b = split_by_case(sa_r), split_by_case(sb_r)

    def gap(split, cases):
        gem = [s for c in cases for s in split[c][0]]
        oth = [s for c in cases for s in split[c][1]]
        return np.mean(gem) - np.mean(oth) if gem and oth else float("nan")

    def did(cases):
        return gap(split_a, cases) - gap(split_b, cases)

    if any(is_gemini_response(r[1]) for r in resp) and any(not is_gemini_response(r[1]) for r in resp):
        res["self_preference"] = {
            "gemini_minus_other_original": float(gap(split_a, resp_cases)),
            "gemini_minus_other_other_judge": float(gap(split_b, resp_cases)),
            "difference_in_differences": float(did(resp_cases)),
            "ci": boot_ci(did, resp_cases, rng),
            "note": "Positive = the original (Gemini) judge favors Gemini-generated responses more than this judge does.",
        }
    return res


def stability(g: Grades, judge: str, rng: np.random.Generator) -> dict | None:
    labels = g.reps(judge)
    jm = g.judge_meta.get(judge, {})
    include_original = judge == ORIGINAL_SLUG and jm.get("temperature") is None and jm.get("seed") is None
    if include_original:
        labels = [ORIGINAL] + labels
    if len(labels) < 2:
        return None
    common = sorted(set.intersection(*(set(g.verdicts[l]) for l in labels)))
    if not common:
        return None
    mat = np.array([[g.verdicts[l][k] for l in labels] for k in common], dtype=int)
    flips = mat.min(axis=1) != mat.max(axis=1)
    res = {
        "raters": labels,
        "n_pairs": len(common),
        "flip_rate": float(flips.mean()),
        "flip_rate_by_sign": {
            s: float(flips[[g.meta[k]["sign"] == s for k in common]].mean()) for s in ("positive", "negative")
        },
        "fleiss_kappa": fleiss_kappa(mat),
        "pairwise_cohen_kappa": {
            f"{x} vs {y}": agreement([g.verdicts[x][k] for k in common], [g.verdicts[y][k] for k in common])["cohen_kappa"]
            for x, y in itertools.combinations(labels, 2)
        },
    }
    keys = set(common)
    scores = [g.case_scores(l, keys) for l in labels]
    resp = sorted(set.intersection(*(set(s) for s in scores)))
    case_sd = [np.std([s[r] for s in scores], ddof=1) for r in resp]
    res["case_score_sd_across_repeats"] = {"mean": float(np.mean(case_sd)), "max": float(np.max(case_sd))} if resp else None

    cfgs = sorted({r[1] for r in resp})
    cases = sorted({r[0] for r in resp})
    mats = [score_matrix({r: s[r] for r in resp}, cfgs, cases) for s in scores]
    gl = [col_means(m) for m in mats]
    boot = np.array([col_means(mats[0][:, rng.choice(len(cases), len(cases))]) for _ in range(N_BOOT)])
    boot_sd = boot.std(axis=0)
    rep_sd = np.std(np.stack(gl), axis=0, ddof=1)
    per_cfg = {
        c: {"global_score_sd_across_repeats": float(rep_sd[i]), "bootstrap_sd_over_cases": float(boot_sd[i])}
        for i, c in enumerate(cfgs)
    }
    res["configuration_level"] = per_cfg
    ratio = [v["global_score_sd_across_repeats"] / v["bootstrap_sd_over_cases"] for v in per_cfg.values() if v["bootstrap_sd_over_cases"] > 0]
    res["median_ratio_judge_sd_to_bootstrap_sd"] = float(np.median(ratio)) if ratio else None
    # Ranking stability across repeats.
    res["kendall_tau_between_repeats"] = {
        f"{labels[i]} vs {labels[j]}": float(kendalltau(gl[i], gl[j]).statistic)
        for i, j in itertools.combinations(range(len(labels)), 2)
    }
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--regrades-dir", type=Path, default=OUTPUT_DIR / "regrades")
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "judge_agreement.json")
    args = p.parse_args()

    rng = np.random.default_rng(SEED)
    g = Grades(args.regrades_dir)
    judges = sorted(g.judge_meta)
    if not judges:
        raise SystemExit(f"No re-grades found in {args.regrades_dir}. Run validation/regrade.py first.")

    out = {"original_judge": "gemini-2.5-flash (evals/)", "failures": g.failures, "judges": {}}
    for j in judges:
        if not g.reps(j):
            continue
        out["judges"][j] = {"meta": g.judge_meta[j], "cross_judge_vs_original": cross_judge(g, j, rng), "stability": stability(g, j, rng)}
    # Pairwise agreement between the non-original judges (first repeats).
    firsts = {j: g.reps(j)[0] for j in out["judges"]}
    out["pairwise_between_regrade_judges"] = {}
    for x, y in itertools.combinations(firsts, 2):
        a, b = g.verdicts[firsts[x]], g.verdicts[firsts[y]]
        ks = sorted(set(a) & set(b))
        out["pairwise_between_regrade_judges"][f"{x} vs {y}"] = agreement([a[k] for k in ks], [b[k] for k in ks])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:6000])
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

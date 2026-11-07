#!/usr/bin/env python3
"""Regional mutation predictor with VCF input and 5-fold validation.

The proposed model avoids using the exact variant identifier as the only
candidate. Variants are aggregated by chromosome, genomic window, broad
annotation region and mutation class. The model combines frequency,
conditional evidence, genomic proximity and annotation similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from mlxtend.frequent_patterns import apriori, association_rules
from numba import cuda, njit
from sklearn.model_selection import KFold


POP_GROUPS = ("afr", "ami", "amr", "asj", "eas", "fin", "mid", "nfe", "remaining", "sas")
REGION_PRIORITY = (
    "missense_variant",
    "synonymous_variant",
    "frameshift_variant",
    "stop_gained",
    "splice",
    "intron_variant",
    "upstream_gene_variant",
    "downstream_gene_variant",
    "intergenic_variant",
)


@dataclass(frozen=True)
class Event:
    chrom: str
    pos: int
    ref: str
    alt: str
    window: int
    region: str
    mut_class: str
    label: str


def parse_info(info: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in info.split(";"):
        if "=" in item:
            key, value = item.split("=", 1)
            out[key] = value
        else:
            out[item] = "1"
    return out


def variant_class(ref: str, alt: str, info: dict[str, str]) -> str:
    allele_type = info.get("allele_type", "").lower()
    variant_type = info.get("variant_type", "").lower()
    if "snv" in allele_type or "snv" in variant_type or (len(ref) == 1 and len(alt) == 1):
        return "SNV"
    if len(ref) > len(alt):
        return "DEL"
    if len(ref) < len(alt):
        return "INS"
    return "COMPLEX"


def region_from_vep(info: dict[str, str]) -> str:
    vep = info.get("vep", "")
    for term in REGION_PRIORITY:
        if term in vep:
            if "splice" in term:
                return "splice"
            return term.replace("_variant", "")
    return "other"


def read_vcfs(paths: list[Path], window_size: int, max_records: int | None = None) -> tuple[list[Event], np.ndarray, list[str]]:
    event_index: dict[str, int] = {}
    events: list[Event] = []
    weights_by_profile: dict[str, defaultdict[int, float]] = {}
    profile_names: list[str] = []
    records = 0

    for path in paths:
        local_samples: list[str] = []
        with path.open("rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line:
                    continue
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split("\t")
                    if len(header) > 9:
                        local_samples = [f"{path.name}:{name}" for name in header[9:]]
                        for name in local_samples:
                            if name not in weights_by_profile:
                                weights_by_profile[name] = defaultdict(float)
                                profile_names.append(name)
                    continue
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 8:
                    continue
                chrom, pos_s, _id, ref, alts, _qual, filt, info_s = parts[:8]
                info = parse_info(info_s)
                if filt not in ("PASS", "."):
                    continue
                pos = int(pos_s)
                region = region_from_vep(info)
                for alt in alts.split(","):
                    cls = variant_class(ref, alt, info)
                    window = pos // window_size
                    label = f"{chrom}:W{window}:{region}:{cls}"
                    if label not in event_index:
                        event_index[label] = len(events)
                        events.append(Event(chrom, pos, ref, alt, window, region, cls, label))
                    idx = event_index[label]
                    if local_samples and len(parts) > 9:
                        fmt = parts[8].split(":")
                        gt_pos = fmt.index("GT") if "GT" in fmt else 0
                        for sample_name, sample_value in zip(local_samples, parts[9:]):
                            fields = sample_value.split(":")
                            gt = fields[gt_pos] if gt_pos < len(fields) else fields[0]
                            alleles = gt.replace("|", "/").split("/")
                            if any(a not in ("0", ".") for a in alleles):
                                weights_by_profile[sample_name][idx] += 1.0
                    else:
                        for group in POP_GROUPS:
                            if group not in weights_by_profile:
                                weights_by_profile[group] = defaultdict(float)
                                profile_names.append(group)
                            ac = float(info.get(f"AC_{group}", info.get(f"AC_{group}_XY", "0")).split(",")[0] or 0)
                            af = float(info.get(f"AF_{group}", info.get(f"AF_{group}_XY", "0")).split(",")[0] or 0)
                            if ac > 0 or af > 0:
                                weights_by_profile[group][idx] += max(ac, af)
                records += 1
                if max_records and records >= max_records:
                    break
        if max_records and records >= max_records:
            break

    matrix = np.zeros((len(profile_names), len(events)), dtype=np.float32)
    for gi, group in enumerate(profile_names):
        for idx, value in weights_by_profile[group].items():
            matrix[gi, idx] = value
    return events, matrix, profile_names


def binary_profiles(weight_matrix: np.ndarray, min_weight: float) -> np.ndarray:
    binary = (weight_matrix >= min_weight).astype(np.uint8)
    keep = binary.sum(axis=0) > 0
    return binary[:, keep], keep


def event_features(events: list[Event]) -> tuple[np.ndarray, list[str]]:
    regions = sorted({e.region for e in events})
    classes = sorted({e.mut_class for e in events})
    reg_id = {r: i for i, r in enumerate(regions)}
    cls_id = {c: i for i, c in enumerate(classes)}
    arr = np.zeros((len(events), 4), dtype=np.float32)
    for i, e in enumerate(events):
        arr[i] = [e.pos, e.window, reg_id[e.region], cls_id[e.mut_class]]
    return arr, regions


def current_profile_candidates(events: list[Event], X: np.ndarray, profile_names: list[str]) -> dict[str, set[str]]:
    profile_candidates: dict[str, set[str]] = {}
    for profile_idx, profile_name in enumerate(profile_names):
        observed_idx = np.where(X[profile_idx] > 0)[0]
        profile_candidates[profile_name] = {events[int(idx)].label for idx in observed_idx}
    return profile_candidates


def load_mongo_knowledge(mongo_uri: str, mongo_db: str) -> tuple[list[Event], dict[str, set[str]]]:
    from pymongo import MongoClient

    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    events = [
        Event(
            chrom=doc["chrom"],
            pos=int(doc["pos"]),
            ref=doc.get("ref", "N"),
            alt=doc.get("alt", "."),
            window=int(doc["window"]),
            region=doc["region"],
            mut_class=doc["mut_class"],
            label=doc["label"],
        )
        for doc in db.candidates.find({}, {"_id": 0})
    ]
    profiles = {
        doc["name"]: set(doc.get("candidates", []))
        for doc in db.profiles.find({}, {"_id": 0, "name": 1, "candidates": 1})
    }
    client.close()
    return events, profiles


def merge_with_mongo_knowledge(
    events: list[Event],
    X: np.ndarray,
    profile_names: list[str],
    mongo_uri: str | None,
    mongo_db: str,
) -> tuple[list[Event], np.ndarray, list[str], int, int]:
    if not mongo_uri:
        return events, X, profile_names, 0, 0

    mongo_events, mongo_profiles = load_mongo_knowledge(mongo_uri, mongo_db)
    event_by_label = {event.label: event for event in mongo_events}
    for event in events:
        event_by_label[event.label] = event

    profile_candidates = {name: set(labels) for name, labels in mongo_profiles.items()}
    for name, labels in current_profile_candidates(events, X, profile_names).items():
        profile_candidates.setdefault(name, set()).update(labels)

    merged_events = list(event_by_label.values())
    event_index = {event.label: idx for idx, event in enumerate(merged_events)}
    merged_profiles = list(profile_names) + [name for name in profile_candidates if name not in set(profile_names)]
    merged_X = np.zeros((len(merged_profiles), len(merged_events)), dtype=np.uint8)
    for row_idx, profile_name in enumerate(merged_profiles):
        for label in profile_candidates.get(profile_name, set()):
            if label in event_index:
                merged_X[row_idx, event_index[label]] = 1

    return merged_events, merged_X, merged_profiles, len(mongo_events), len(mongo_profiles)


def save_mongo_knowledge(events: list[Event], X: np.ndarray, profile_names: list[str], mongo_uri: str | None, mongo_db: str) -> None:
    if not mongo_uri:
        return

    from pymongo import MongoClient, UpdateOne

    freq, cooc = train_counts(X)
    client = MongoClient(mongo_uri)
    db = client[mongo_db]
    db.candidates.create_index("label", unique=True)
    db.profiles.create_index("name", unique=True)
    db.cooccurrences.create_index([("a", 1), ("b", 1)], unique=True)

    candidate_ops = [
        UpdateOne(
            {"label": event.label},
            {
                "$set": {
                    "label": event.label,
                    "chrom": event.chrom,
                    "pos": int(event.pos),
                    "ref": event.ref,
                    "alt": event.alt,
                    "window": int(event.window),
                    "region": event.region,
                    "mut_class": event.mut_class,
                    "frequency": float(freq[idx]),
                    "updated_at": time.time(),
                }
            },
            upsert=True,
        )
        for idx, event in enumerate(events)
    ]
    if candidate_ops:
        db.candidates.bulk_write(candidate_ops, ordered=False)

    profile_ops = []
    for row_idx, profile_name in enumerate(profile_names):
        labels = [events[int(idx)].label for idx in np.where(X[row_idx] > 0)[0]]
        profile_ops.append(
            UpdateOne(
                {"name": profile_name},
                {"$set": {"name": profile_name, "candidates": labels, "updated_at": time.time()}},
                upsert=True,
            )
        )
    if profile_ops:
        db.profiles.bulk_write(profile_ops, ordered=False)

    cooc_ops = []
    for i in range(len(events)):
        for j in range(i, len(events)):
            value = float(cooc[i, j])
            if value <= 0:
                continue
            cooc_ops.append(
                UpdateOne(
                    {"a": events[i].label, "b": events[j].label},
                    {"$set": {"a": events[i].label, "b": events[j].label, "weight": value, "updated_at": time.time()}},
                    upsert=True,
                )
            )
    if cooc_ops:
        db.cooccurrences.bulk_write(cooc_ops, ordered=False)
    client.close()


def split_observed_hidden(X: np.ndarray, fold_seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(fold_seed)
    observed = np.zeros_like(X)
    hidden = np.zeros_like(X)
    for i, row in enumerate(X):
        idx = np.where(row > 0)[0]
        if len(idx) < 4:
            observed[i, idx] = 1
            continue
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * 0.30)))
        hidden[i, idx[:cut]] = 1
        observed[i, idx[cut:]] = 1
    return observed, hidden


def topk_metrics(scores: np.ndarray, truth: np.ndarray, observed: np.ndarray, k: int) -> dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    hits = []
    ndcgs = []
    for s, y, obs in zip(scores, truth, observed):
        candidates = np.where(obs == 0)[0]
        if y.sum() == 0 or len(candidates) == 0:
            continue
        ranked = candidates[np.argsort(s[candidates])[::-1]][:k]
        rel = y[ranked].astype(float)
        hit_count = rel.sum()
        precision = hit_count / k
        recall = hit_count / y.sum()
        precisions.append(precision)
        recalls.append(recall)
        f1s.append((2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0)
        hits.append(1.0 if hit_count > 0 else 0.0)
        discounts = 1.0 / np.log2(np.arange(2, len(ranked) + 2))
        dcg = float((rel * discounts).sum())
        ideal_len = min(int(y.sum()), k)
        idcg = float(discounts[:ideal_len].sum()) if ideal_len else 0.0
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        "precision@k": float(np.mean(precisions)),
        "recall@k": float(np.mean(recalls)),
        "f1@k": float(np.mean(f1s)),
        "hit@k": float(np.mean(hits)),
        "ndcg@k": float(np.mean(ndcgs)),
    }


def train_counts(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    freq = X_train.sum(axis=0).astype(np.float32)
    cooc = X_train.T @ X_train
    return freq, cooc.astype(np.float32)


def score_independence(freq: np.ndarray, n_train: int, n_test: int) -> np.ndarray:
    base = (freq + 1.0) / (n_train + 2.0)
    return np.tile(base, (n_test, 1))


def score_cooccurrence(observed: np.ndarray, freq: np.ndarray, cooc: np.ndarray) -> np.ndarray:
    denom = np.maximum(freq, 1.0)
    cond = cooc / denom[:, None]
    return observed @ cond


def score_bayes(observed: np.ndarray, freq: np.ndarray, cooc: np.ndarray, n_train: int) -> np.ndarray:
    prior = (freq + 1.0) / (n_train + 2.0)
    cond = (cooc + 1.0) / (freq[:, None] + 2.0)
    log_prior = np.log(prior)
    log_cond = np.log(cond)
    return log_prior[None, :] + observed @ log_cond


def score_apriori(X_train: np.ndarray, observed: np.ndarray, min_support: float = 0.1) -> np.ndarray:
    min_count = int(math.ceil(min_support * X_train.shape[0]))
    if min_count <= 1 or X_train.shape[1] > 800:
        freq, cooc = train_counts(X_train)
        denom = np.maximum(freq, 1.0)
        confidence = cooc / denom[:, None]
        confidence[cooc < max(min_count, 1)] = 0.0
        return observed @ confidence

    cols = [f"E{i}" for i in range(X_train.shape[1])]
    df = pd.DataFrame(X_train.astype(bool), columns=cols)
    itemsets = apriori(df, min_support=min_support, use_colnames=True, max_len=2)
    scores = np.zeros((observed.shape[0], X_train.shape[1]), dtype=np.float32)
    if itemsets.empty:
        return scores
    rules = association_rules(itemsets, metric="confidence", min_threshold=0.0)
    for _, row in rules.iterrows():
        ant = list(row["antecedents"])
        con = list(row["consequents"])
        if len(ant) != 1 or len(con) != 1:
            continue
        a = int(ant[0][1:])
        c = int(con[0][1:])
        scores[:, c] += observed[:, a] * float(row["confidence"])
    return scores


@njit
def _regional_scores_cpu(observed: np.ndarray, features: np.ndarray, freq: np.ndarray, cooc: np.ndarray) -> np.ndarray:
    n, m = observed.shape
    out = np.zeros((n, m), dtype=np.float32)
    for s in range(n):
        for j in range(m):
            value = 0.15 * math.log1p(freq[j])
            count = 0
            for i in range(m):
                if observed[s, i] == 0:
                    continue
                dist = abs(features[j, 0] - features[i, 0])
                proximity = math.exp(-dist / 1_000_000.0)
                same_region = 1.0 if features[j, 2] == features[i, 2] else 0.0
                same_class = 1.0 if features[j, 3] == features[i, 3] else 0.0
                assoc = cooc[i, j] / max(freq[i], 1.0)
                value += 0.48 * assoc + 0.24 * proximity + 0.08 * same_region + 0.05 * same_class
                count += 1
            if count > 0:
                value /= math.sqrt(count)
            out[s, j] = value
    return out


@cuda.jit
def _regional_scores_gpu_kernel(observed, features, freq, cooc, out):
    s, j = cuda.grid(2)
    n = observed.shape[0]
    m = observed.shape[1]
    if s >= n or j >= m:
        return
    value = 0.15 * math.log(1.0 + freq[j])
    count = 0
    for i in range(m):
        if observed[s, i] == 0:
            continue
        dist = abs(features[j, 0] - features[i, 0])
        proximity = math.exp(-dist / 1000000.0)
        same_region = 1.0 if features[j, 2] == features[i, 2] else 0.0
        same_class = 1.0 if features[j, 3] == features[i, 3] else 0.0
        denom = freq[i] if freq[i] > 1.0 else 1.0
        assoc = cooc[i, j] / denom
        value += 0.48 * assoc + 0.24 * proximity + 0.08 * same_region + 0.05 * same_class
        count += 1
    if count > 0:
        value /= math.sqrt(count)
    out[s, j] = value


def score_proposed(observed: np.ndarray, features: np.ndarray, freq: np.ndarray, cooc: np.ndarray, use_gpu: bool) -> tuple[np.ndarray, str]:
    if use_gpu and cuda.is_available():
        d_obs = cuda.to_device(observed)
        d_feat = cuda.to_device(features)
        d_freq = cuda.to_device(freq)
        d_cooc = cuda.to_device(cooc)
        out = np.zeros((observed.shape[0], observed.shape[1]), dtype=np.float32)
        d_out = cuda.to_device(out)
        threads = (16, 16)
        blocks = (math.ceil(observed.shape[0] / threads[0]), math.ceil(observed.shape[1] / threads[1]))
        _regional_scores_gpu_kernel[blocks, threads](d_obs, d_feat, d_freq, d_cooc, d_out)
        d_out.copy_to_host(out)
        return out, "CUDA"
    return _regional_scores_cpu(observed.astype(np.uint8), features.astype(np.float32), freq.astype(np.float32), cooc.astype(np.float32)), "CPU-numba"


def novelty_recall(scores: np.ndarray, truth: np.ndarray, observed: np.ndarray, train_presence: np.ndarray, k: int) -> float:
    vals = []
    novel_mask = train_presence == 0
    for s, y, obs in zip(scores, truth, observed):
        target = (y > 0) & novel_mask
        if target.sum() == 0:
            continue
        candidates = np.where(obs == 0)[0]
        ranked = candidates[np.argsort(s[candidates])[::-1]][:k]
        vals.append(float(target[ranked].sum() / target.sum()))
    return float(np.mean(vals)) if vals else 0.0


def run_validation(X: np.ndarray, features: np.ndarray, ks: list[int], seed: int, use_gpu: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    rows = []
    novelty_rows = []
    timing_rows = []
    backend = "CPU-numba"
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train = X[train_idx]
        X_test_full = X[test_idx]
        observed, hidden = split_observed_hidden(X_test_full, seed + fold)
        t0 = time.perf_counter()
        freq, cooc = train_counts(X_train)
        count_time = time.perf_counter() - t0
        methods = {}
        scorers = (
            ("Independência", lambda: score_independence(freq, len(train_idx), len(test_idx)), 0.0),
            ("Dependência direta", lambda: score_cooccurrence(observed, freq, cooc), count_time),
            ("Apriori", lambda: score_apriori(X_train, observed), 0.0),
            ("Bayes empírico", lambda: score_bayes(observed, freq, cooc, len(train_idx)), count_time),
        )
        for method_name, scorer, shared_time in scorers:
            t0 = time.perf_counter()
            methods[method_name] = scorer()
            timing_rows.append({
                "fold": fold,
                "method": method_name,
                "seconds": time.perf_counter() - t0 + shared_time,
            })
        t0 = time.perf_counter()
        proposed, backend = score_proposed(observed, features, freq, cooc, use_gpu)
        timing_rows.append({
            "fold": fold,
            "method": "Método proposto",
            "seconds": time.perf_counter() - t0 + count_time,
        })
        methods["Método proposto"] = proposed
        train_presence = X_train.sum(axis=0)
        for name, scores in methods.items():
            for k in ks:
                metrics = topk_metrics(scores, hidden, observed, k)
                metrics["fold"] = fold
                metrics["method"] = name
                metrics["k"] = k
                rows.append(metrics)
            novelty_rows.append({
                "fold": fold,
                "method": name,
                "novel_recall@50": novelty_recall(scores, hidden, observed, train_presence, 50),
            })
    return pd.DataFrame(rows), pd.DataFrame(novelty_rows), pd.DataFrame(timing_rows), backend


def summarize(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["method"]
    agg = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def plot_metric(summary: pd.DataFrame, out_path: Path, k: int) -> None:
    methods = summary["method"].tolist()
    f1 = summary["f1@k_mean"].to_numpy()
    err = summary["f1@k_std"].fillna(0).to_numpy()
    plt.figure(figsize=(10, 5))
    colors = ["#6b7280" if m != "Método proposto" else "#047857" for m in methods]
    plt.bar(methods, f1, yerr=err, color=colors, capsize=4)
    plt.ylabel(f"F1@{k}")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_main_metrics(summary: pd.DataFrame, out_path: Path, k: int) -> None:
    methods = summary["method"].tolist()
    metrics = ["precision@k_mean", "recall@k_mean", "f1@k_mean"]
    labels = [f"Precisão@{k}", f"Revocação@{k}", f"F1@{k}"]
    x = np.arange(len(methods))
    width = 0.24
    plt.figure(figsize=(11, 5.5))
    palette = ["#2563eb", "#f59e0b", "#047857"]
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.bar(x + (i - 1) * width, summary[metric], width, label=label, color=palette[i])
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("Valor médio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_fold_stability(results: pd.DataFrame, out_path: Path) -> None:
    k = int(results["k"].iloc[0]) if "k" in results.columns and not results.empty else 5
    plt.figure(figsize=(10, 5))
    for method, group in results.groupby("method"):
        ordered = group.sort_values("fold")
        lw = 2.8 if method == "Método proposto" else 1.6
        plt.plot(ordered["fold"], ordered["f1@k"], marker="o", linewidth=lw, label=method)
    plt.xlabel("Fold")
    plt.ylabel(f"F1@{k}")
    plt.xticks(sorted(results["fold"].unique()))
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_timing(timing: pd.DataFrame, out_path: Path) -> None:
    summary = summarize(timing, ["seconds"])
    plt.figure(figsize=(10, 5))
    colors = ["#6b7280" if m != "Método proposto" else "#047857" for m in summary["method"]]
    plt.bar(summary["method"], summary["seconds_mean"], yerr=summary["seconds_std"].fillna(0), color=colors, capsize=4)
    plt.ylabel("Tempo por fold (s)")
    plt.yscale("log")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_presence_map(X: np.ndarray, events: list[Event], out_path: Path, top_n: int = 120) -> None:
    freq = X.sum(axis=0)
    order = np.argsort(freq)[::-1][: min(top_n, X.shape[1])]
    plt.figure(figsize=(12, 4.5))
    plt.imshow(X[:, order], aspect="auto", interpolation="nearest", cmap="viridis")
    plt.xlabel("Regiões mutacionais mais frequentes")
    plt.ylabel("Perfis")
    plt.colorbar(label="Presença")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_direct_network(events: list[Event], X: np.ndarray, out_path: Path, top_n: int, top_edges: int) -> None:
    freq, cooc = train_counts(X)
    nodes = np.argsort(freq)[::-1][: min(top_n, len(events))]
    sub = cooc[np.ix_(nodes, nodes)].copy()
    np.fill_diagonal(sub, 0)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if sub[i, j] > 0:
                edges.append((i, j, float(sub[i, j])))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_edges]
    if not edges:
        return
    connected_local = sorted({idx for i, j, _ in edges for idx in (i, j)})
    local_map = {old: new for new, old in enumerate(connected_local)}
    nodes = nodes[connected_local]
    edges = [(local_map[i], local_map[j], w) for i, j, w in edges]
    max_w = max(w for _, _, w in edges)
    windows = np.array([events[node].window for node in nodes], dtype=float)
    if windows.max() > windows.min():
        x = 2 * (windows - windows.min()) / (windows.max() - windows.min()) - 1
    else:
        x = np.linspace(-1, 1, len(nodes))
    categories = sorted({(events[node].region, events[node].mut_class) for node in nodes})
    category_y = {cat: y for cat, y in zip(categories, np.linspace(-0.9, 0.9, len(categories)))}
    y = np.array([category_y[(events[node].region, events[node].mut_class)] for node in nodes])
    offsets = ((np.arange(len(nodes)) % 5) - 2) * 0.045
    xy = np.column_stack([x + offsets, y])
    plt.figure(figsize=(9, 9))
    for i, j, w in edges:
        alpha = 0.12 + 0.5 * (w / max_w)
        lw = 0.4 + 2.5 * (w / max_w)
        plt.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#64748b", alpha=alpha, linewidth=lw, zorder=1)
    sizes = 80 + 420 * (freq[nodes] / max(freq[nodes]))
    plt.scatter(xy[:, 0], xy[:, 1], s=sizes, c="#047857", edgecolor="white", linewidth=1.1, zorder=2)
    for idx, node in enumerate(nodes):
        label = events[node].label.replace("chrY:", "").replace(":", "\n")
        plt.text(1.15 * xy[idx, 0], 1.15 * xy[idx, 1], label, ha="center", va="center", fontsize=6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_radial_network(events: list[Event], X: np.ndarray, out_path: Path, top_n: int, top_edges: int) -> None:
    freq, cooc = train_counts(X)
    nodes = np.argsort(freq)[::-1][: min(top_n, len(events))]
    sub = cooc[np.ix_(nodes, nodes)].copy()
    np.fill_diagonal(sub, 0)
    edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if sub[i, j] > 0:
                edges.append((i, j, float(sub[i, j])))
    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:top_edges]
    if not edges:
        return
    connected_local = sorted({idx for i, j, _ in edges for idx in (i, j)})
    local_map = {old: new for new, old in enumerate(connected_local)}
    nodes = nodes[connected_local]
    edges = [(local_map[i], local_map[j], w) for i, j, w in edges]
    max_w = max(w for _, _, w in edges)
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(nodes), endpoint=False)
    xy = np.column_stack([np.cos(angles), np.sin(angles)])
    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw={"aspect": "equal"})
    for i, j, w in edges:
        p0 = xy[i]
        p3 = xy[j]
        verts = [tuple(p0), tuple(p0 * 0.25), tuple(p3 * 0.25), tuple(p3)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
        alpha = 0.045 + 0.22 * (w / max_w)
        lw = 0.35 + 2.1 * (w / max_w)
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none", edgecolor="#12806f", lw=lw, alpha=alpha, zorder=1))
    sizes = 35 + 380 * (freq[nodes] / max(freq[nodes]))
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c="#0f766e", edgecolor="white", linewidth=0.8, zorder=3)
    for idx, node in enumerate(nodes):
        event = events[node]
        label = f"W{event.window}-{event.region[:8]}-{event.mut_class}"
        x, y = xy[idx]
        rotation = np.degrees(np.arctan2(y, x))
        ha = "left"
        if x < 0:
            rotation += 180
            ha = "right"
        ax.text(1.08 * x, 1.08 * y, label, fontsize=6.3, rotation=rotation, rotation_mode="anchor", ha=ha, va="center", color="#111827")
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_latex_tables(summary: pd.DataFrame, novelty: pd.DataFrame, out_path: Path) -> None:
    order = ["Independência", "Dependência direta", "Apriori", "Bayes empírico", "Método proposto"]
    if "k" in summary.columns:
        summary = summary[summary["k"] == int(summary["k"].min())]
    summary = summary.set_index("method").loc[order].reset_index()
    novelty_s = summarize(novelty, ["novel_recall@50"]).set_index("method").loc[order].reset_index()
    with out_path.open("wt", encoding="utf-8") as handle:
        handle.write("% Gerado automaticamente por src/vcf_regional_predictor.py\n")
        handle.write("\\begin{tabular}{lccc}\\toprule\n")
        handle.write("Método & Precisão@5 & Revocação@5 & F1-score \\\\ \\midrule\n")
        best = summary["f1@k_mean"].max()
        for _, row in summary.iterrows():
            f1 = f"{row['f1@k_mean']:.4f}\\pm{row['f1@k_std']:.4f}"
            if abs(row["f1@k_mean"] - best) < 1e-12:
                f1 = f"\\mathbf{{{f1}}}"
            handle.write(
                f"{row['method']} & ${row['precision@k_mean']:.4f}\\pm{row['precision@k_std']:.4f}$ & "
                f"${row['recall@k_mean']:.4f}\\pm{row['recall@k_std']:.4f}$ & ${f1}$ \\\\\n"
            )
        handle.write("\\bottomrule\\end{tabular}\n\n")
        handle.write("\\begin{tabular}{lc}\\toprule\n")
        handle.write("Método & Revocação@50 de eventos inéditos \\\\ \\midrule\n")
        for _, row in novelty_s.iterrows():
            handle.write(f"{row['method']} & ${row['novel_recall@50_mean']:.4f}\\pm{row['novel_recall@50_std']:.4f}$ \\\\\n")
        handle.write("\\bottomrule\\end{tabular}\n")


def save_outputs(events: list[Event], X: np.ndarray, results: pd.DataFrame, novelty: pd.DataFrame, timing: pd.DataFrame, backend: str, out_dir: Path, network_top_n: int, network_top_edges: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_knowledge_base(events, X, out_dir)
    results.to_csv(out_dir / "fold_metrics.csv", index=False)
    novelty.to_csv(out_dir / "novelty_metrics.csv", index=False)
    timing.to_csv(out_dir / "timing_metrics.csv", index=False)
    summary = summarize(results, ["precision@k", "recall@k", "f1@k", "hit@k", "ndcg@k"], ["k", "method"])
    novelty_summary = summarize(novelty, ["novel_recall@50"])
    timing_summary = summarize(timing, ["seconds"])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    novelty_summary.to_csv(out_dir / "summary_novelty.csv", index=False)
    timing_summary.to_csv(out_dir / "summary_timing.csv", index=False)
    write_latex_tables(summary, novelty, out_dir / "tables_resultados.tex")
    for k in sorted(results["k"].unique()):
        summary_k = summary[summary["k"] == k]
        suffix = f"k{k}"
        plot_metric(summary_k, Path(f"imgs/desempenho_vcf_regional_{suffix}.png"), int(k))
        plot_main_metrics(summary_k, Path(f"imgs/comparacao_metricas_vcf_regional_{suffix}.png"), int(k))
    first_k = int(sorted(results["k"].unique())[0])
    plot_fold_stability(results[results["k"] == first_k], Path(f"imgs/estabilidade_f1_vcf_regional_k{first_k}.png"))
    plot_timing(timing, Path("imgs/tempo_vcf_regional.png"))
    plot_presence_map(X, events, Path("imgs/mapa_presenca_vcf_regional.png"))
    plot_direct_network(events, X, Path("imgs/rede_correlacao_vcf_regional.png"), network_top_n, network_top_edges)
    plot_radial_network(events, X, Path("imgs/rede_correlacao_radial_vcf_regional.png"), max(network_top_n, 64), max(network_top_edges, 180))
    meta = {
        "n_profiles": int(X.shape[0]),
        "n_candidates": int(X.shape[1]),
        "backend": backend,
        "events_example": [e.label for e in events[:10]],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def save_knowledge_base(events: list[Event], X: np.ndarray, out_dir: Path) -> None:
    freq, cooc = train_counts(X)
    candidates = pd.DataFrame(
        [
            {
                "candidate_index": idx,
                "candidate_id": event.label,
                "chrom": event.chrom,
                "representative_pos": event.pos,
                "representative_ref": event.ref,
                "representative_alt": event.alt,
                "window": event.window,
                "region": event.region,
                "variant_class": event.mut_class,
                "frequency": float(freq[idx]),
            }
            for idx, event in enumerate(events)
        ]
    )
    candidates.to_csv(out_dir / "knowledge_candidates.csv", index=False)
    pd.DataFrame(cooc).to_csv(out_dir / "knowledge_cooccurrence.csv", index=False)
    np.savez_compressed(
        out_dir / "knowledge_base.npz",
        profile_candidate_matrix=X.astype(np.uint8),
        frequency=freq.astype(np.float32),
        cooccurrence=cooc.astype(np.float32),
    )


def _vcf_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value)


def export_predictions(
    events: list[Event],
    X: np.ndarray,
    profile_names: list[str],
    features: np.ndarray,
    top_n: int,
    out_dir: Path,
    window_size: int,
    use_gpu: bool,
    target_profile_names: set[str] | None = None,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    freq, cooc = train_counts(X)
    scores, backend = score_proposed(X, features, freq, cooc, use_gpu)
    rows = []
    events_by_label = {event.label: event for event in events}

    for profile_idx, profile_name in enumerate(profile_names):
        if target_profile_names is not None and profile_name not in target_profile_names:
            continue
        observed = X[profile_idx] > 0
        candidates = np.where(~observed)[0]
        if len(candidates) == 0:
            continue
        ranked = candidates[np.argsort(scores[profile_idx, candidates])[::-1]][:top_n]
        evidence_count = int(observed.sum())
        for rank, event_idx in enumerate(ranked, 1):
            event = events[int(event_idx)]
            window_start = event.window * window_size
            window_end = window_start + window_size - 1
            rows.append(
                {
                    "profile": profile_name,
                    "rank": rank,
                    "candidate_id": event.label,
                    "chrom": event.chrom,
                    "representative_pos": event.pos,
                    "window_start": window_start,
                    "window_end": window_end,
                    "region": event.region,
                    "variant_class": event.mut_class,
                    "score": float(scores[profile_idx, event_idx]),
                    "frequency": float(freq[event_idx]),
                    "observed_evidence_count": evidence_count,
                }
            )

    predictions = pd.DataFrame(rows)
    predictions.to_csv(out_dir / "predicted_candidates.tsv", sep="\t", index=False)

    with (out_dir / "predicted_candidates.vcf").open("wt", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=vcf_regional_predictor\n")
        handle.write('##INFO=<ID=PREDICTED,Number=0,Type=Flag,Description="Regional candidate predicted by the proposed method">\n')
        handle.write('##INFO=<ID=PROFILE,Number=1,Type=String,Description="Profile for which this regional candidate was predicted">\n')
        handle.write('##INFO=<ID=WINDOW_START,Number=1,Type=Integer,Description="Start of the regional genomic window">\n')
        handle.write('##INFO=<ID=WINDOW_END,Number=1,Type=Integer,Description="End of the regional genomic window">\n')
        handle.write('##INFO=<ID=REGION,Number=1,Type=String,Description="Functional region used by the regional candidate">\n')
        handle.write('##INFO=<ID=VARIANT_CLASS,Number=1,Type=String,Description="Variant class used by the regional candidate">\n')
        handle.write('##INFO=<ID=CANDIDATE_ID,Number=1,Type=String,Description="Regional candidate identifier used by the predictor">\n')
        handle.write('##INFO=<ID=SCORE,Number=1,Type=Float,Description="Prediction score assigned by the proposed method">\n')
        handle.write('##INFO=<ID=RANK,Number=1,Type=Integer,Description="Rank of the prediction for this profile">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for row in rows:
            profile = _vcf_token(str(row["profile"]))
            candidate = _vcf_token(str(row["candidate_id"]))
            info = (
                f"PREDICTED;PROFILE={profile};WINDOW_START={row['window_start']};"
                f"WINDOW_END={row['window_end']};REGION={_vcf_token(str(row['region']))};"
                f"VARIANT_CLASS={_vcf_token(str(row['variant_class']))};"
                f"CANDIDATE_ID={candidate};SCORE={row['score']:.6f};RANK={row['rank']}"
            )
            record_id = f"PREDICTED_{profile}_{row['rank']}_{candidate}"
            event = events_by_label[str(row["candidate_id"])]
            handle.write(
                f"{row['chrom']}\t{row['representative_pos']}\t{record_id}\t{event.ref}\t{event.alt}\t.\tPREDICTED\t{info}\n"
            )

    prediction_meta = {
        "backend": backend,
        "top_n_per_profile": top_n,
        "n_profiles": int(X.shape[0]),
        "n_candidates": int(X.shape[1]),
        "n_predictions": int(len(rows)),
        "tsv": str(out_dir / "predicted_candidates.tsv"),
        "vcf": str(out_dir / "predicted_candidates.vcf"),
    }
    (out_dir / "prediction_metadata.json").write_text(json.dumps(prediction_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf", nargs="+", type=Path, help="One or more VCF files")
    parser.add_argument("--window-size", type=int, default=250_000)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=120_000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--report-ks", default="", help="Comma-separated Top-K values. Defaults to --k.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA even if available")
    parser.add_argument("--out-dir", type=Path, default=Path("experimentos_vcf"))
    parser.add_argument("--network-top-n", type=int, default=28, help="Number of frequent regional candidates in the direct network")
    parser.add_argument("--network-top-edges", type=int, default=90, help="Number of strongest co-occurrence edges in the direct network")
    parser.add_argument("--prediction-top-n", type=int, default=20, help="Number of predicted regional candidates exported per profile")
    parser.add_argument("--mongo-uri", default="", help="MongoDB URI used to load and update the persistent knowledge base")
    parser.add_argument("--mongo-db", default="vcf_regional_predictor", help="MongoDB database name for the knowledge base")
    args = parser.parse_args()

    start = time.perf_counter()
    events_all, weights, groups = read_vcfs(args.vcf, args.window_size, args.max_records)
    X, keep = binary_profiles(weights, args.min_weight)
    events = [e for e, ok in zip(events_all, keep) if ok]
    current_profiles = set(groups)
    mongo_uri = args.mongo_uri.strip() or None
    events, X, groups, loaded_candidates, loaded_profiles = merge_with_mongo_knowledge(events, X, groups, mongo_uri, args.mongo_db)
    features, _regions = event_features(events)
    ks = [args.k]
    if args.report_ks.strip():
        ks = sorted({int(x.strip()) for x in args.report_ks.split(",") if x.strip()})
    results, novelty, timing, backend = run_validation(X, features, ks, args.seed, use_gpu=not args.cpu)
    save_outputs(events, X, results, novelty, timing, backend, args.out_dir, args.network_top_n, args.network_top_edges)
    save_mongo_knowledge(events, X, groups, mongo_uri, args.mongo_db)
    prediction_backend = export_predictions(
        events,
        X,
        groups,
        features,
        args.prediction_top_n,
        args.out_dir,
        args.window_size,
        use_gpu=not args.cpu,
        target_profile_names=current_profiles,
    )
    elapsed = time.perf_counter() - start
    print(
        f"profiles={X.shape[0]} candidates={X.shape[1]} loaded_mongo_candidates={loaded_candidates} "
        f"loaded_mongo_profiles={loaded_profiles} backend={backend} prediction_backend={prediction_backend} elapsed={elapsed:.2f}s"
    )
    print(f"predictions={args.out_dir / 'predicted_candidates.tsv'} {args.out_dir / 'predicted_candidates.vcf'}")
    print(summarize(results, ["precision@k", "recall@k", "f1@k"], ["k", "method"]).to_string(index=False))


if __name__ == "__main__":
    main()

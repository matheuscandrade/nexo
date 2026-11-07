#!/usr/bin/env python3
"""Incremental chr16 VCF experiment runner."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vcf_regional_predictor import (
    Event,
    POP_GROUPS,
    binary_profiles,
    event_features,
    export_predictions,
    parse_info,
    plot_direct_network,
    plot_fold_stability,
    plot_main_metrics,
    plot_metric,
    plot_presence_map,
    plot_radial_network,
    plot_timing,
    read_vcfs,
    region_from_vep,
    run_validation,
    save_knowledge_base,
    score_proposed,
    summarize,
    train_counts,
    variant_class,
)


def matrix_from_profiles(
    weights_by_profile: dict[str, defaultdict[int, float]],
    profile_names: list[str],
    n_events: int,
) -> np.ndarray:
    matrix = np.zeros((len(profile_names), n_events), dtype=np.float32)
    for row_idx, profile_name in enumerate(profile_names):
        for event_idx, value in weights_by_profile[profile_name].items():
            matrix[row_idx, event_idx] = value
    return matrix


def checkpoint_state(
    events_all: list[Event],
    weights_by_profile: dict[str, defaultdict[int, float]],
    profile_names: list[str],
    min_weight: float,
) -> tuple[list[Event], np.ndarray]:
    weights = matrix_from_profiles(weights_by_profile, profile_names, len(events_all))
    X, keep = binary_profiles(weights, min_weight)
    events = [event for event, ok in zip(events_all, keep) if ok]
    return events, X


def save_checkpoint_outputs(
    events: list[Event],
    X: np.ndarray,
    profile_names: list[str],
    results: pd.DataFrame,
    novelty: pd.DataFrame,
    timing: pd.DataFrame,
    backend: str,
    out_dir: Path,
    network_top_n: int,
    network_top_edges: int,
    prediction_top_n: int,
    window_size: int,
    make_images: bool,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_knowledge_base(events, X, out_dir)
    results.to_csv(out_dir / "fold_metrics.csv", index=False)
    novelty.to_csv(out_dir / "novelty_metrics.csv", index=False)
    timing.to_csv(out_dir / "timing_metrics.csv", index=False)
    summary = summarize(results, ["precision@k", "recall@k", "f1@k", "hit@k", "ndcg@k"], ["k", "method"])
    summary_timing = summarize(timing, ["seconds"])
    summary_novelty = summarize(novelty, ["novel_recall@50"])
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    summary_timing.to_csv(out_dir / "summary_timing.csv", index=False)
    summary_novelty.to_csv(out_dir / "summary_novelty.csv", index=False)
    features, _regions = event_features(events)
    prediction_backend = export_predictions(
        events,
        X,
        profile_names,
        features,
        prediction_top_n,
        out_dir,
        window_size,
        use_gpu=backend == "CUDA",
    )
    if make_images:
        img_dir = out_dir / "imgs"
        img_dir.mkdir(exist_ok=True)
        for k in sorted(results["k"].unique()):
            summary_k = summary[summary["k"] == k]
            plot_metric(summary_k, img_dir / f"desempenho_chr16_k{int(k)}.png", int(k))
            plot_main_metrics(summary_k, img_dir / f"comparacao_metricas_chr16_k{int(k)}.png", int(k))
        first_k = int(sorted(results["k"].unique())[0])
        plot_fold_stability(results[results["k"] == first_k], img_dir / f"estabilidade_f1_chr16_k{first_k}.png")
        plot_timing(timing, img_dir / "tempo_chr16.png")
        plot_presence_map(X, events, img_dir / "mapa_presenca_chr16.png", top_n=min(500, X.shape[1]))
        plot_direct_network(events, X, img_dir / "rede_correlacao_chr16.png", network_top_n, network_top_edges)
        plot_radial_network(events, X, img_dir / "rede_correlacao_radial_chr16.png", max(network_top_n, 64), max(network_top_edges, 180))
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "n_profiles": int(X.shape[0]),
                "n_regions": int(X.shape[1]),
                "backend": backend,
                "prediction_backend": prediction_backend,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return prediction_backend


def run_checkpoint(
    accepted_records: int,
    events_all: list[Event],
    weights_by_profile: dict[str, defaultdict[int, float]],
    profile_names: list[str],
    args: argparse.Namespace,
    is_final: bool,
) -> dict[str, object]:
    t0 = time.perf_counter()
    events, X = checkpoint_state(events_all, weights_by_profile, profile_names, args.min_weight)
    features, _regions = event_features(events)
    results, novelty, timing, backend = run_validation(X, features, args.report_ks, args.seed, use_gpu=not args.cpu)
    out_dir = args.out_dir / f"records_{accepted_records}"
    prediction_backend = save_checkpoint_outputs(
        events,
        X,
        profile_names,
        results,
        novelty,
        timing,
        backend,
        out_dir,
        args.network_top_n,
        args.network_top_edges,
        args.prediction_top_n,
        args.window_size,
        make_images=is_final,
    )
    elapsed = time.perf_counter() - t0
    metadata = {
        "accepted_records": int(accepted_records),
        "n_profiles": int(X.shape[0]),
        "n_regions": int(X.shape[1]),
        "backend": backend,
        "prediction_backend": prediction_backend,
        "checkpoint_seconds": float(elapsed),
        "is_final": bool(is_final),
    }
    (out_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def plot_incremental(incremental_metrics: pd.DataFrame, timing: pd.DataFrame, out_dir: Path) -> None:
    img_dir = out_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted(incremental_metrics["k"].unique()):
        subset = incremental_metrics[incremental_metrics["k"] == k]
        plt.figure(figsize=(11, 5.5))
        for method, group in subset.groupby("method"):
            ordered = group.sort_values("accepted_records")
            lw = 2.8 if method == "Método proposto" else 1.6
            plt.plot(ordered["accepted_records"], ordered["f1@k_mean"], marker="o", linewidth=lw, label=method)
        plt.xlabel("Registros VCF com filtro aceitável")
        plt.ylabel(f"F1@{int(k)}")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(img_dir / f"evolucao_f1_chr16_k{int(k)}.png", dpi=220)
        plt.close()

    plt.figure(figsize=(11, 5.5))
    for method, group in timing.groupby("method"):
        ordered = group.sort_values("accepted_records")
        lw = 2.8 if method == "Método proposto" else 1.6
        plt.plot(ordered["accepted_records"], ordered["seconds_mean"], marker="o", linewidth=lw, label=method)
    plt.xlabel("Registros VCF com filtro aceitável")
    plt.ylabel("Tempo médio por fold (s)")
    plt.yscale("log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(img_dir / "evolucao_tempo_chr16.png", dpi=220)
    plt.close()


def write_results_tex(out_dir: Path, source_vcf: Path, checkpoints: pd.DataFrame, metrics: pd.DataFrame, timing: pd.DataFrame) -> None:
    final_records = int(checkpoints["accepted_records"].max())
    out_dir_tex = out_dir.as_posix()
    final_metrics = metrics[(metrics["accepted_records"] == final_records) & (metrics["k"] == 5)].copy()
    final_timing = timing[timing["accepted_records"] == final_records].copy()
    top50 = metrics[(metrics["accepted_records"] == final_records) & (metrics["k"] == 50)].copy()
    proposed_evolution = metrics[(metrics["method"] == "Método proposto") & (metrics["k"] == 5)].sort_values("accepted_records")

    method_order = ["Independência", "Dependência direta", "Apriori", "Bayes empírico", "Método proposto"]
    final_metrics["method"] = pd.Categorical(final_metrics["method"], categories=method_order, ordered=True)
    final_metrics = final_metrics.sort_values("method")
    final_timing["method"] = pd.Categorical(final_timing["method"], categories=method_order, ordered=True)
    final_timing = final_timing.sort_values("method")

    def fmt_pm(row: pd.Series, col: str) -> str:
        return f"${row[col + '_mean']:.4f}\\pm{row[col + '_std']:.4f}$"

    lines: list[str] = []
    lines.append("\\chapter{Resultados Atualizados com Dados Incrementais}\\label{cap:resultadosAtualizados}\n")
    lines.append(
        "Este capítulo apresenta a execução incremental realizada com o arquivo "
        f"\\path{{{source_vcf.name}}}. A avaliação foi repetida em blocos acumulados de 500.000 registros com filtro aceitável, "
        "seguindo os mesmos critérios de validação usados nos resultados anteriores. Em cada etapa, a base de conhecimento foi reconstruída com todos os registros disponíveis até aquele ponto, permitindo comparar o efeito do aumento do volume de entrada sobre as métricas preditivas e o tempo de execução.\n"
    )
    lines.append("\\section{Volumes Avaliados}\n")
    final_records_text = f"{final_records:,}".replace(",", ".")
    lines.append(
        f"Foram avaliados {len(checkpoints)} volumes incrementais. O maior volume utilizou {final_records_text} registros com filtro aceitável, "
        f"resultando em {int(checkpoints.iloc[-1]['n_regions'])} regiões mutacionais distribuídas em {int(checkpoints.iloc[-1]['n_profiles'])} perfis populacionais. "
        "Os registros considerados aceitáveis foram aqueles com \\textit{FILTER} igual a \\textit{PASS} ou sem filtro informado.\n"
    )
    lines.append("\\begin{table}[H]\n\\centering\n\\caption{Resumo dos volumes incrementais avaliados no cromossomo 16.}\n\\label{tab:chr16VolumesIncrementais}\n\\small\n\\begin{tabular}{rccc}\n\\toprule\nRegistros utilizados & Perfis & Regiões mutacionais & Tempo de validação e saída (s) \\\\\n\\midrule\n")
    for _, row in checkpoints.iterrows():
        lines.append(f"{int(row['accepted_records']):,} & {int(row['n_profiles'])} & {int(row['n_regions'])} & {row['checkpoint_seconds']:.2f} \\\\\n".replace(",", "."))
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    lines.append("\\section{Desempenho no Maior Volume}\n")
    lines.append(
        "A Tabela \\ref{tab:chr16MetricasTop5} apresenta os resultados no maior volume avaliado. "
        "A comparação mantém os mesmos métodos usados anteriormente, permitindo observar se a vantagem do método proposto se mantém quando a base de entrada aumenta.\n"
    )
    lines.append("\\begin{table}[H]\n\\centering\n\\caption{Métricas de validação em \\textit{Top-5} no maior volume incremental.}\n\\label{tab:chr16MetricasTop5}\n\\small\n\\begin{tabular}{lccccc}\n\\toprule\nMétodo & Precisão & Revocação & F1-score & Taxa de acerto & NDCG \\\\\n\\midrule\n")
    for _, row in final_metrics.iterrows():
        lines.append(
            f"{row['method']} & {fmt_pm(row, 'precision@k')} & {fmt_pm(row, 'recall@k')} & {fmt_pm(row, 'f1@k')} & {fmt_pm(row, 'hit@k')} & {fmt_pm(row, 'ndcg@k')} \\\\\n"
        )
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    if not top50.empty:
        proposed50 = top50[top50["method"] == "Método proposto"]
        if not proposed50.empty:
            row = proposed50.iloc[0]
            lines.append(
                f"No \\textit{{Top-50}}, o método proposto alcançou F1-score médio de {row['f1@k_mean']:.4f}, "
                f"com revocação média de {row['recall@k_mean']:.4f}. Esse resultado indica o comportamento do método quando a lista de regiões priorizadas é ampliada.\n"
            )

    lines.append("\\section{Efeito do Aumento do Volume de Entrada}\n")
    lines.append(
        "A Figura \\ref{fig:chr16EvolucaoF1Top5} mostra a evolução do F1@5 conforme novos blocos de registros são incorporados. "
        "Essa análise permite verificar se o acréscimo de dados melhora a estabilidade da predição ou se o desempenho se estabiliza após determinado volume.\n"
    )
    lines.append(f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{{{out_dir_tex}/imgs/evolucao_f1_chr16_k5.png}}\n\\caption{{Evolução do F1@5 nos volumes incrementais do cromossomo 16.}}\n\\label{{fig:chr16EvolucaoF1Top5}}\n\\end{{figure}}\n")
    lines.append(
        "Para o método proposto, a sequência de F1@5 ao longo dos volumes foi "
        + ", ".join(f"{int(r['accepted_records']):,} registros com {r['f1@k_mean']:.4f}".replace(",", ".") for _, r in proposed_evolution.iterrows())
        + ".\n"
    )

    lines.append("\\section{Tempo de Execução}\n")
    lines.append(
        "A Tabela \\ref{tab:chr16TempoFinal} apresenta o tempo médio por \\textit{fold} no maior volume avaliado. "
        "A Figura \\ref{fig:chr16EvolucaoTempo} mostra como esse tempo variou nos diferentes volumes de entrada.\n"
    )
    lines.append("\\begin{table}[H]\n\\centering\n\\caption{Tempo médio por \\textit{fold} no maior volume incremental.}\n\\label{tab:chr16TempoFinal}\n\\small\n\\begin{tabular}{lc}\n\\toprule\nMétodo & Tempo médio por \\textit{fold} (s) \\\\\n\\midrule\n")
    for _, row in final_timing.iterrows():
        lines.append(f"{row['method']} & ${row['seconds_mean']:.5f}\\pm{row['seconds_std']:.5f}$ \\\\\n")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    lines.append(f"\\begin{{figure}}[H]\n\\centering\n\\includegraphics[width=\\textwidth]{{{out_dir_tex}/imgs/evolucao_tempo_chr16.png}}\n\\caption{{Evolução do tempo médio por \\textit{{fold}} nos volumes incrementais do cromossomo 16.}}\n\\label{{fig:chr16EvolucaoTempo}}\n\\end{{figure}}\n")

    lines.append("\\section{Discussão dos Resultados Incrementais}\n")
    lines.append(
        "A avaliação incremental mostra como o método se comporta quando a base de conhecimento recebe mais variantes. "
        "Como o arquivo usado é populacional, os perfis continuam sendo grupos agregados do gnomAD, e não indivíduos separados. "
        "Assim, o ganho esperado com o aumento do volume está principalmente no crescimento do número de regiões mutacionais conhecidas e no refinamento das relações de ocorrência conjunta entre elas. "
        "O tempo cresce de acordo com a quantidade de regiões avaliadas, mas a etapa de pontuação permanece adequada à paralelização em \\textit{GPU}, pois cada par perfil--região pode ser calculado de forma independente.\n"
    )
    (out_dir / "resultados_att.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vcf", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("experimentos_chr16_incremental"))
    parser.add_argument("--window-size", type=int, default=250_000)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--step", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-ks", default="5,20,50")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--network-top-n", type=int, default=28)
    parser.add_argument("--network-top-edges", type=int, default=90)
    parser.add_argument("--prediction-top-n", type=int, default=20)
    parser.add_argument("--max-accepted", type=int, default=0, help="Stop after this many acceptable records. Zero means read the full VCF.")
    args = parser.parse_args()
    args.report_ks = sorted({int(x.strip()) for x in args.report_ks.split(",") if x.strip()})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    event_index: dict[str, int] = {}
    events_all: list[Event] = []
    weights_by_profile: dict[str, defaultdict[int, float]] = {group: defaultdict(float) for group in POP_GROUPS}
    profile_names = list(POP_GROUPS)
    accepted_records = 0
    total_variant_lines = 0
    next_checkpoint = args.step
    checkpoint_rows: list[dict[str, object]] = []
    t_start = time.perf_counter()

    with args.vcf.open("rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            total_variant_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            chrom, pos_s, _id, ref, alts, _qual, filt, info_s = parts[:8]
            if filt not in ("PASS", "."):
                continue
            info = parse_info(info_s)
            pos = int(pos_s)
            region = region_from_vep(info)
            for alt in alts.split(","):
                cls = variant_class(ref, alt, info)
                window = pos // args.window_size
                label = f"{chrom}:W{window}:{region}:{cls}"
                if label not in event_index:
                    event_index[label] = len(events_all)
                    events_all.append(Event(chrom, pos, ref, alt, window, region, cls, label))
                idx = event_index[label]
                for group in POP_GROUPS:
                    ac = float(info.get(f"AC_{group}", info.get(f"AC_{group}_XY", "0")).split(",")[0] or 0)
                    af = float(info.get(f"AF_{group}", info.get(f"AF_{group}_XY", "0")).split(",")[0] or 0)
                    if ac > 0 or af > 0:
                        weights_by_profile[group][idx] += max(ac, af)
            accepted_records += 1
            if accepted_records >= next_checkpoint:
                print(f"checkpoint {accepted_records} accepted records after {total_variant_lines} variant lines", flush=True)
                checkpoint_rows.append(
                    run_checkpoint(
                        accepted_records,
                        events_all,
                        weights_by_profile,
                        profile_names,
                        args,
                        is_final=False,
                    )
                )
                next_checkpoint += args.step
            if args.max_accepted and accepted_records >= args.max_accepted:
                break

    if not checkpoint_rows or int(checkpoint_rows[-1]["accepted_records"]) != accepted_records:
        print(f"final checkpoint {accepted_records} accepted records after {total_variant_lines} variant lines", flush=True)
        checkpoint_rows.append(
            run_checkpoint(
                accepted_records,
                events_all,
                weights_by_profile,
                profile_names,
                args,
                is_final=True,
            )
        )
    else:
        print(f"refreshing final checkpoint images for {accepted_records} accepted records", flush=True)
        checkpoint_rows[-1] = run_checkpoint(
            accepted_records,
            events_all,
            weights_by_profile,
            profile_names,
            args,
            is_final=True,
        )

    checkpoints = pd.DataFrame(checkpoint_rows)
    checkpoints["total_variant_lines"] = total_variant_lines
    checkpoints.to_csv(args.out_dir / "incremental_checkpoints.csv", index=False)

    metrics_frames = []
    timing_frames = []
    for row in checkpoint_rows:
        volume = int(row["accepted_records"])
        run_dir = args.out_dir / f"records_{volume}"
        m = pd.read_csv(run_dir / "summary_metrics.csv")
        t = pd.read_csv(run_dir / "summary_timing.csv")
        m.insert(0, "accepted_records", volume)
        t.insert(0, "accepted_records", volume)
        metrics_frames.append(m)
        timing_frames.append(t)
    metrics = pd.concat(metrics_frames, ignore_index=True)
    timing = pd.concat(timing_frames, ignore_index=True)
    metrics.to_csv(args.out_dir / "incremental_summary_metrics.csv", index=False)
    timing.to_csv(args.out_dir / "incremental_summary_timing.csv", index=False)
    plot_incremental(metrics, timing, args.out_dir)
    write_results_tex(args.out_dir, args.vcf, checkpoints, metrics, timing)
    elapsed = time.perf_counter() - t_start
    (args.out_dir / "incremental_metadata.json").write_text(
        json.dumps(
            {
                "source_vcf": str(args.vcf),
                "accepted_records": accepted_records,
                "total_variant_lines": total_variant_lines,
                "elapsed_seconds": elapsed,
                "step": args.step,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"done accepted={accepted_records} variant_lines={total_variant_lines} elapsed={elapsed:.2f}s")
    print(f"results={args.out_dir / 'resultados_att.tex'}")


if __name__ == "__main__":
    main()

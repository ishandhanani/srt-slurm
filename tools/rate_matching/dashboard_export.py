"""Dashboard export: CSV/JSON data + interactive Plotly HTML charts.

Charts:
  1. Pareto frontier: Interactivity vs Throughput/GPU (all configs + frontier)
  2. SOL vs E2E comparison: scatter with connected lines, comparison table
  3. TTFT analysis: bar chart with constraint line

The SOL vs E2E chart labels points by per-worker concurrency (c8, c32, etc.)
and shows both concurrency variants (1.0x, 1.05x) if present.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def export_rate_matching_results(sweep_state: dict, output_dir: str) -> None:
    """Export summary data as CSV and JSON for dashboarding."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = sweep_state.get("sweep_name", "sweep")

    # Summary JSON
    summary = {
        "sweep_name": name,
        "ctx_result": sweep_state.get("ctx_result", {}),
        "pareto_count": len(sweep_state.get("pareto_frontier", [])),
        "e2e_count": len(sweep_state.get("sol_vs_e2e", [])),
    }
    with open(out / f"{name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pareto JSON/CSV
    pareto = sweep_state.get("pareto_frontier", [])
    if pareto:
        with open(out / f"{name}_pareto.json", "w") as f:
            json.dump(pareto, f, indent=2)
        _write_csv(out / f"{name}_pareto.csv", pareto, [
            "pareto_rank", "config_name", "mode", "concurrency", "mtp_num",
            "mtp_accept_rate", "batch_size", "ratio_str",
            "interactivity", "tpot_ms", "avg_step_time_ms",
            "output_tput_per_gpu", "output_tput_per_gen_gpu",
            "total_throughput", "total_tput_per_gpu",
            "gen_req_rate", "ctx_gen_inst_ratio",
            "ctx_instances", "gen_instances", "total_gpus",
            "estimate_e2e_latency_s",
        ])

    # SOL vs E2E JSON/CSV
    sol_vs_e2e = sweep_state.get("sol_vs_e2e", [])
    if sol_vs_e2e:
        with open(out / f"{name}_sol_vs_e2e.json", "w") as f:
            json.dump(sol_vs_e2e, f, indent=2)
        flat = []
        for e in sol_vs_e2e:
            flat.append({
                "pareto_rank": e.get("pareto_rank"),
                "multiplier": e.get("multiplier"),
                "per_worker_concurrency": e.get("per_worker_concurrency"),
                "system_concurrency": e.get("system_concurrency"),
                "sol_tpot_ms": e.get("sol", {}).get("tpot_ms"),
                "e2e_tpot_ms": e.get("e2e", {}).get("tpot_ms"),
                "tpot_diff_pct": e.get("diff_pct", {}).get("tpot"),
                "sol_tput_per_gpu": e.get("sol", {}).get("output_tput_per_gpu"),
                "e2e_tput_per_gpu": e.get("e2e", {}).get("output_tput_per_gpu"),
                "tput_diff_pct": e.get("diff_pct", {}).get("output_tput_per_gpu"),
                "e2e_ttft_ms": e.get("e2e_ttft_ms"),
                "ttft_pass": e.get("ttft_pass"),
            })
        _write_csv(out / f"{name}_sol_vs_e2e.csv", flat, list(flat[0].keys()) if flat else [])


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Plotly chart generation
# ---------------------------------------------------------------------------

def generate_charts(
    sweep_state: dict,
    output_dir: str,
    sweep_name: str | None = None,
) -> list[str]:
    """Generate interactive HTML charts from sweep results.

    Returns list of generated HTML file paths.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("WARNING: plotly not installed. Install with: pip install plotly")
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = sweep_name or sweep_state.get("sweep_name", "sweep")
    generated = []

    # --- Chart 1: Pareto Frontier ---
    all_results = sweep_state.get("rate_matching_results", [])
    pareto = sweep_state.get("pareto_frontier", [])
    if all_results:
        fig = go.Figure()

        # All configs (grey)
        non_pareto = [r for r in all_results if not r.get("is_pareto_optimal")]
        if non_pareto:
            fig.add_trace(go.Scatter(
                x=[r["interactivity"] for r in non_pareto],
                y=[r["output_tput_per_gpu"] for r in non_pareto],
                mode="markers",
                marker=dict(size=8, color="lightgrey", line=dict(width=1, color="grey")),
                text=[f"{r['config_name']}" for r in non_pareto],
                hoverinfo="text+x+y",
                name="Non-Pareto",
            ))

        # Pareto frontier (connected line + markers)
        if pareto:
            pareto_sorted = sorted(pareto, key=lambda x: x["interactivity"])
            fig.add_trace(go.Scatter(
                x=[r["interactivity"] for r in pareto_sorted],
                y=[r["output_tput_per_gpu"] for r in pareto_sorted],
                mode="lines+markers+text",
                marker=dict(size=12, color="blue", symbol="star"),
                line=dict(width=2, color="blue", dash="dash"),
                text=[f"c{r['concurrency']}" for r in pareto_sorted],
                textposition="top center",
                hovertext=[
                    f"c{r['concurrency']} {r['mode'].upper()} mtp{r.get('mtp_num', 0)}<br>"
                    f"Ratio: {r['ratio_str']}<br>"
                    f"Batch: {r['batch_size']}<br>"
                    f"GPUs: {r['total_gpus']}"
                    for r in pareto_sorted
                ],
                hoverinfo="text",
                name="Pareto Frontier",
            ))

        fig.update_layout(
            title=f"{name} -- Pareto Frontier",
            xaxis_title="Interactivity = 1000 / median_TPOT(ms) (tok/s/user)",
            yaxis_title="Output Throughput / GPU (tok/s/GPU)",
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
            template="plotly_white",
        )
        path = str(out / f"{name}_pareto_frontier.html")
        fig.write_html(path, include_plotlyjs="cdn")
        generated.append(path)

    # --- Chart 2: SOL vs E2E Comparison ---
    sol_vs_e2e = sweep_state.get("sol_vs_e2e", [])
    if sol_vs_e2e:
        pareto_map = {p["pareto_rank"]: p for p in pareto}
        fig = go.Figure()

        # Group by multiplier
        multipliers = sorted(set(e.get("multiplier", 1.0) for e in sol_vs_e2e))

        for mult in multipliers:
            entries = sorted(
                [e for e in sol_vs_e2e if e.get("multiplier", 1.0) == mult],
                key=lambda e: e.get("sol", {}).get("interactivity", 0),
            )
            mult_label = f"{mult:.2f}x"

            # SOL line
            sol_x = [e["sol"]["interactivity"] for e in entries]
            sol_y = [e["sol"]["output_tput_per_gpu"] for e in entries]
            sol_text = [_pw_label(e, pareto_map) for e in entries]
            fig.add_trace(go.Scatter(
                x=sol_x, y=sol_y,
                mode="lines+markers+text",
                marker=dict(size=10, symbol="circle"),
                text=sol_text,
                textposition="top center",
                name=f"SOL ({mult_label})",
                hovertext=[
                    f"SOL {_pw_label(e, pareto_map)}<br>"
                    f"TPOT: {e['sol']['tpot_ms']:.2f}ms<br>"
                    f"Tput/GPU: {e['sol']['output_tput_per_gpu']:.1f}"
                    for e in entries
                ],
                hoverinfo="text",
            ))

            # E2E line
            e2e_x = [e["e2e"]["interactivity"] for e in entries]
            e2e_y = [e["e2e"]["output_tput_per_gpu"] for e in entries]
            fig.add_trace(go.Scatter(
                x=e2e_x, y=e2e_y,
                mode="lines+markers+text",
                marker=dict(size=10, symbol="diamond"),
                text=sol_text,
                textposition="bottom center",
                name=f"E2E ({mult_label})",
                hovertext=[
                    f"E2E {_pw_label(e, pareto_map)}<br>"
                    f"TPOT: {e['e2e']['tpot_ms']:.2f}ms<br>"
                    f"Tput/GPU: {e['e2e']['output_tput_per_gpu']:.1f}<br>"
                    f"Diff: {e['diff_pct']['output_tput_per_gpu']:+.1f}%"
                    for e in entries
                ],
                hoverinfo="text",
            ))

            # Dashed connection lines between SOL and E2E points
            for i, e in enumerate(entries):
                diff = e["diff_pct"]["output_tput_per_gpu"]
                fig.add_trace(go.Scatter(
                    x=[sol_x[i], e2e_x[i]],
                    y=[sol_y[i], e2e_y[i]],
                    mode="lines",
                    line=dict(color="grey", dash="dot", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ))

        fig.update_layout(
            title=f"{name} -- SOL vs E2E Comparison",
            xaxis_title="Interactivity = 1000 / median_TPOT(ms) (tok/s/user)",
            yaxis_title="Output Throughput / GPU (tok/s/GPU)",
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
            template="plotly_white",
        )

        # Build comparison table HTML
        table_html = _build_comparison_table(sol_vs_e2e, pareto_map)

        chart_div = fig.to_html(include_plotlyjs="cdn", full_html=False)
        full_html = f"""<!DOCTYPE html>
<html><head><title>{name} SOL vs E2E</title></head>
<body>
{chart_div}
{table_html}
</body></html>"""

        path = str(out / f"{name}_sol_vs_e2e.html")
        with open(path, "w") as f:
            f.write(full_html)
        generated.append(path)

    # --- Chart 3: TTFT Analysis ---
    ttft_entries = [e for e in sol_vs_e2e if e.get("e2e_ttft_ms", 0) > 0]
    if ttft_entries:
        fig = go.Figure()
        for mult in multipliers:
            entries = sorted(
                [e for e in ttft_entries if e.get("multiplier", 1.0) == mult],
                key=lambda e: e.get("pareto_rank", 0),
            )
            labels = [_pw_label(e, pareto_map) for e in entries]
            ttft_vals = [e["e2e_ttft_ms"] for e in entries]
            fig.add_trace(go.Bar(
                x=labels,
                y=ttft_vals,
                name=f"TTFT ({mult:.2f}x)",
                text=[f"{v:.0f}ms" for v in ttft_vals],
                textposition="outside",
            ))

        # Constraint line
        constraint = ttft_entries[0].get("ttft_constraint_ms")
        if constraint:
            fig.add_hline(
                y=constraint, line_dash="dash", line_color="red",
                annotation_text=f"Constraint: {constraint:.0f}ms",
            )

        fig.update_layout(
            title=f"{name} -- TTFT Analysis",
            xaxis_title="Configuration (per-worker concurrency)",
            yaxis_title="Median TTFT (ms)",
            legend=dict(x=0.99, y=0.99, xanchor="right", yanchor="top"),
            barmode="group",
            template="plotly_white",
        )
        path = str(out / f"{name}_ttft_analysis.html")
        fig.write_html(path, include_plotlyjs="cdn")
        generated.append(path)

    return generated


def _pw_label(entry: dict, pareto_map: dict) -> str:
    """Label a point by per-worker concurrency and ratio."""
    pp = pareto_map.get(entry.get("pareto_rank"), {})
    conc = entry.get("per_worker_concurrency", pp.get("concurrency", "?"))
    ratio = pp.get("ratio_str", "")
    return f"c{conc} ({ratio})" if ratio else f"c{conc}"


def _build_comparison_table(sol_vs_e2e: list[dict], pareto_map: dict) -> str:
    """Build HTML comparison table for SOL vs E2E results."""
    rows = []
    for e in sorted(sol_vs_e2e, key=lambda x: (x.get("pareto_rank", 0), x.get("multiplier", 1.0))):
        pp = pareto_map.get(e.get("pareto_rank"), {})
        sol = e.get("sol", {})
        e2e = e.get("e2e", {})
        diff = e.get("diff_pct", {})

        pw_conc = e.get("per_worker_concurrency", pp.get("concurrency", "?"))
        sys_conc = e.get("system_concurrency", "?")
        ratio = pp.get("ratio_str", "?")
        mode = pp.get("mode", "?").upper()
        mtp = pp.get("mtp_num", 0)
        accept_rate = pp.get("mtp_accept_rate", 1.0)
        step_ms = pp.get("avg_step_time_ms", 0)
        ctx_i = pp.get("ctx_instances", "?")
        gen_i = pp.get("gen_instances", "?")
        total_gpus = pp.get("total_gpus", "?")
        mult = e.get("multiplier", 1.0)

        tpot_diff = diff.get("tpot", 0)
        tput_diff = diff.get("output_tput_per_gpu", 0)
        tpot_color = "green" if abs(tpot_diff) < 5 else ("orange" if abs(tpot_diff) < 15 else "red")
        tput_color = "green" if abs(tput_diff) < 10 else ("orange" if abs(tput_diff) < 20 else "red")

        ttft_ms = e.get("e2e_ttft_ms", 0)
        ttft_val = f"{ttft_ms:.0f}" if ttft_ms > 0 else "-"
        ttft_pass = e.get("ttft_pass", True)
        ttft_status = "PASS" if ttft_pass else "FAIL"
        ttft_color = "green" if ttft_pass else "red"

        rows.append(
            f"<tr>"
            f'<td style="font-weight:bold">c{pw_conc}</td>'
            f"<td>{mult:.2f}x</td>"
            f"<td>{mode} MTP-{mtp}</td>"
            f'<td style="text-align:right">{accept_rate:.2f}</td>'
            f'<td style="text-align:right">{step_ms:.2f}</td>'
            f'<td style="text-align:center">{ratio}</td>'
            f'<td style="text-align:center">{ctx_i}</td>'
            f'<td style="text-align:center">{gen_i}</td>'
            f'<td style="text-align:center">{total_gpus}</td>'
            f'<td style="text-align:center">{sys_conc}</td>'
            f'<td style="text-align:right">{sol.get("tpot_ms", 0):.2f}</td>'
            f'<td style="text-align:right">{e2e.get("tpot_ms", 0):.2f}</td>'
            f'<td style="text-align:right;color:{tpot_color}">{tpot_diff:+.1f}%</td>'
            f'<td style="text-align:right">{sol.get("output_tput_per_gpu", 0):.1f}</td>'
            f'<td style="text-align:right">{e2e.get("output_tput_per_gpu", 0):.1f}</td>'
            f'<td style="text-align:right;color:{tput_color}">{tput_diff:+.1f}%</td>'
            f'<td style="text-align:right">{ttft_val}</td>'
            f'<td style="text-align:center;color:{ttft_color};font-weight:bold">{ttft_status}</td>'
            f"</tr>"
        )

    return f"""
<div style="margin-top:20px; font-family:sans-serif;">
<h3>Configuration Details</h3>
<p style="font-size:12px; color:#666;">
  <b>Step(ms)</b> = avg prev_device_step_time (device-side decode step).
  <b>Accept Rate</b> = effective tokens per step per user (MTP lookup table).
  <b>SOL TPOT</b> = Step(ms) / AcceptRate.
  <b>SOL Tput/GPU</b> = output_throughput / (ctx_gpus * ctx:gen_ratio + gen_gpus).
</p>
<table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse; font-size:13px;">
<thead style="background:#f0f0f0">
<tr>
<th>PW Conc</th><th>Variant</th><th>Mode</th><th>Accept</th><th>Step(ms)</th>
<th>SOL Ratio</th><th>CTX</th><th>GEN</th><th>GPUs</th><th>Sys Conc</th>
<th>SOL TPOT</th><th>E2E TPOT</th><th>Diff</th>
<th>SOL Tput/GPU</th><th>E2E Tput/GPU</th><th>Diff</th>
<th>TTFT(ms)</th><th>TTFT</th>
</tr>
</thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
</div>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export rate-matching dashboard")
    parser.add_argument("-i", "--input", required=True, help="sweep_state.json path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    args = parser.parse_args()

    with open(args.input) as f:
        state = json.load(f)
    export_rate_matching_results(state, args.output)
    charts = generate_charts(state, args.output)
    if charts:
        print("Generated charts:")
        for c in charts:
            print(f"  {c}")

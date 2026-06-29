from pathlib import Path
from datetime import datetime
import json
import html

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"
EVALUATION_REPORT_PATH = PROJECT_ROOT / "reports" / "evaluation" / "evaluation_result.json"
TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"
FRIENDLY_REPORTS_DIR = PROJECT_ROOT / "reports" / "friendly"

# Paleta pastel morado/rosado usada en todo el reporte y en las figuras,
# para que coincida visualmente con las gráficas de entrenamiento
# (ver src/advisor y los scripts de graficación del paper).
PALETTE = {
    "bg": "#FBF8FC",
    "card": "#FFFFFF",
    "ink": "#2D2438",
    "ink_soft": "#6E6280",
    "border": "#E9E1F2",
    "accent": "#9B7FD4",
    "accent_soft": "#EFE7FB",
    "accent2": "#D4729A",
    "accent2_soft": "#FCEAF1",
    "approved_fg": "#2F7A49",
    "approved_bg": "#E7F5EC",
    "approved_border": "#BFE3CC",
    "warning_fg": "#9C7415",
    "warning_bg": "#FBF1DE",
    "warning_border": "#F0DBA8",
    "rejected_fg": "#A8415C",
    "rejected_bg": "#FBE8ED",
    "rejected_border": "#F0BFCD",
    "line_reward": "#9B7FD4",
    "line_success": "#5FA875",
    "line_collisions": "#D4729A",
}


def load_new_run() -> dict:
    if not NEW_RUN_PATH.exists():
        raise FileNotFoundError(f"No existe: {NEW_RUN_PATH}")

    df = pd.read_csv(NEW_RUN_PATH)

    if df.empty:
        raise ValueError("El archivo new_run_demo.csv está vacío.")

    return df.iloc[0].to_dict()


def load_validation_report() -> dict:
    if not VALIDATION_REPORT_PATH.exists():
        raise FileNotFoundError(f"No existe: {VALIDATION_REPORT_PATH}")

    with open(VALIDATION_REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_evaluation_report() -> dict | None:
    """
    Carga el reporte de convergencia generado por evaluate_run.py.
    Es opcional: si todavía no se corrió ese paso, el reporte se genera
    igual pero sin la sección de convergencia.
    """
    if not EVALUATION_REPORT_PATH.exists():
        return None

    with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def find_episode_metrics(run_id: str) -> Path | None:
    path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"
    return path if path.exists() else None


def moving_average(values: pd.Series, window: int = 20) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def style_axes(ax):
    ax.set_facecolor("#FFFFFF")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(PALETTE["border"])
    ax.tick_params(colors=PALETTE["ink_soft"], labelsize=9)
    ax.xaxis.label.set_color(PALETTE["ink_soft"])
    ax.yaxis.label.set_color(PALETTE["ink_soft"])
    ax.title.set_color(PALETTE["ink"])
    ax.grid(True, color=PALETTE["border"], linewidth=0.8, alpha=0.7)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def save_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    color: str,
    moving_avg: bool = False,
    figsize: tuple[float, float] = (9.6, 4.6),
    fill: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    fig.patch.set_facecolor(PALETTE["bg"])

    y_values = df[y_col]

    if moving_avg:
        y_values = moving_average(y_values)

    ax.plot(df[x_col], y_values, color=color, linewidth=2.4, solid_capstyle="round")

    if fill:
        ax.fill_between(df[x_col], y_values, y_values.min(), color=color, alpha=0.10, linewidth=0)

    ax.set_title(title, fontsize=14, pad=14, fontweight="bold", loc="left")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_path, facecolor=PALETTE["bg"])
    plt.close(fig)


def generate_training_figures(
    run_id: str,
    episode_metrics_path: Path,
    output_dir: Path,
) -> dict:
    df = pd.read_csv(episode_metrics_path)

    figures = {}

    if "episode_reward" in df.columns:
        reward_path = output_dir / "episode_reward.png"
        save_plot(
            df=df, x_col="episode", y_col="episode_reward",
            title="Episode reward", ylabel="Reward",
            output_path=reward_path, color=PALETTE["line_reward"], moving_avg=True,
            figsize=(13.0, 5.2),
        )
        figures["episode_reward"] = {
            "filename": reward_path.name,
            "label": "Episode reward",
            "size": "full",
        }

    success_col = "success" if "success" in df.columns else (
        "success_rate" if "success_rate" in df.columns else None
    )
    if success_col is not None:
        success_path = output_dir / "success_rate_moving.png"
        df_success = df.copy()
        df_success["success_rate_moving"] = moving_average(df_success[success_col], window=20)

        save_plot(
            df=df_success, x_col="episode", y_col="success_rate_moving",
            title="Moving success rate", ylabel="Success rate",
            output_path=success_path, color=PALETTE["line_success"], moving_avg=False,
            figsize=(8.6, 4.8),
        )
        figures["success_rate"] = {
            "filename": success_path.name,
            "label": "Moving success rate",
            "size": "half",
        }

    if "collisions" in df.columns:
        collisions_path = output_dir / "collisions.png"
        save_plot(
            df=df, x_col="episode", y_col="collisions",
            title="Collisions", ylabel="Collisions",
            output_path=collisions_path, color=PALETTE["line_collisions"], moving_avg=True,
            figsize=(8.6, 4.8),
        )
        figures["collisions"] = {
            "filename": collisions_path.name,
            "label": "Collisions",
            "size": "half",
        }

    return figures


def status_colors(status: str) -> tuple[str, str, str]:
    """Devuelve (fg, bg, border) para approved/warning/rejected/desconocido."""
    status = str(status).lower()

    if status in ("approved", "converged"):
        return PALETTE["approved_fg"], PALETTE["approved_bg"], PALETTE["approved_border"]
    if status in ("warning", "partially_converged"):
        return PALETTE["warning_fg"], PALETTE["warning_bg"], PALETTE["warning_border"]
    if status in ("rejected", "not_converged"):
        return PALETTE["rejected_fg"], PALETTE["rejected_bg"], PALETTE["rejected_border"]

    return PALETTE["ink_soft"], PALETTE["accent_soft"], PALETTE["border"]


def format_status_badge(status: str, size: str = "md") -> str:
    fg, bg, border = status_colors(status)
    label = str(status).upper().replace("_", " ")
    pad = "5px 12px" if size == "sm" else "7px 16px"
    font_size = "12px" if size == "sm" else "13px"

    return (
        f'<span style="display:inline-block; background:{bg}; color:{fg}; '
        f'border:1px solid {border}; padding:{pad}; border-radius:999px; '
        f'font-weight:700; font-size:{font_size}; letter-spacing:0.04em;">{html.escape(label)}</span>'
    )


def diagnostic_panel(
    training_status: str | None,
    historical_status: str | None,
    final_recommendation: str | None,
) -> str:
    """
    Franja de 3 fichas de diagnóstico conectadas, mostrando cómo se
    combinan training_status + historical_validation_status en
    final_recommendation. Es el elemento visual central del reporte.
    """

    def panel(eyebrow: str, status: str | None, description: str) -> str:
        if status is None:
            fg, bg, border = PALETTE["ink_soft"], PALETTE["accent_soft"], PALETTE["border"]
            label = "N/A"
        else:
            fg, bg, border = status_colors(status)
            label = str(status).upper().replace("_", " ")

        return f"""
        <div style="flex:1; min-width:0; background:{bg}; border:1px solid {border};
                    border-radius:14px; padding:18px 20px;">
            <p style="margin:0 0 8px; font-size:11px; letter-spacing:0.08em; text-transform:uppercase;
                      color:{fg}; font-weight:700; opacity:0.85;">{html.escape(eyebrow)}</p>
            <p style="margin:0 0 6px; font-size:20px; font-weight:800; color:{fg}; font-family:'Fraunces',serif;">
                {html.escape(label)}
            </p>
            <p style="margin:0; font-size:12.5px; color:{fg}; opacity:0.85; line-height:1.5;">
                {html.escape(description)}
            </p>
        </div>
        """

    arrow = f"""
    <div style="display:flex; align-items:center; justify-content:center; padding:0 6px;
                color:{PALETTE['ink_soft']}; font-size:20px; flex-shrink:0;">→</div>
    """

    training_desc = "Did the agent improve within this run?"
    historical_desc = "Is it comparable to real historical runs?"
    final_desc = "Combined verdict for this run."

    return f"""
    <div style="display:flex; align-items:stretch; gap:0; margin-top:18px;">
        {panel("1 · training status", training_status, training_desc)}
        {arrow}
        {panel("2 · historical validation", historical_status, historical_desc)}
        {arrow}
        {panel("3 · final recommendation", final_recommendation, final_desc)}
    </div>
    """


def dict_to_table(data: dict, keys: list[str], labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    rows = []

    for key in keys:
        value = data.get(key, "")
        label = labels.get(key, key.replace("_", " ").capitalize())
        rows.append(
            "<tr>"
            f'<th>{html.escape(label)}</th>'
            f'<td>{html.escape(str(value))}</td>'
            "</tr>"
        )

    return f'<table class="data-table">{"".join(rows)}</table>'


def validation_metrics_table(validation_report: dict) -> str:
    metric_results = validation_report.get("metric_results", [])

    if not metric_results:
        return '<p class="muted">No metric-level validation was available (not enough comparable history).</p>'

    rows = [
        "<tr>"
        "<th>Metric</th><th>Status</th><th>New value</th>"
        "<th>Historical mean</th><th>Historical std</th><th>Reason</th>"
        "</tr>"
    ]

    for result in metric_results:
        status = str(result.get("status", ""))
        fg, bg, border = status_colors(status)
        status_chip = (
            f'<span style="background:{bg}; color:{fg}; border:1px solid {border}; '
            f'padding:3px 10px; border-radius:999px; font-size:11px; font-weight:700;">'
            f'{html.escape(status.upper())}</span>'
        )

        def fmt(value):
            if value is None or value == "":
                return "—"
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return html.escape(str(value))

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('metric', '')))}</td>"
            f"<td>{status_chip}</td>"
            f"<td>{fmt(result.get('new_value'))}</td>"
            f"<td>{fmt(result.get('historical_mean'))}</td>"
            f"<td>{fmt(result.get('historical_std'))}</td>"
            f"<td class='muted'>{html.escape(str(result.get('reason', '')).replace('_', ' '))}</td>"
            "</tr>"
        )

    return f'<table class="data-table metrics-table">{"".join(rows)}</table>'


def convergence_detail_table(evaluation_report: dict) -> str:
    details = evaluation_report.get("convergence_details", {})

    rows = [
        ("Episodes analyzed", details.get("n_episodes")),
        ("Window size (first/last)", details.get("window_size")),
        ("Initial avg reward", details.get("initial_avg_reward")),
        ("Final avg reward", details.get("final_avg_reward")),
        ("Reward improvement", f"{details.get('reward_improvement_pct')}%" if details.get("reward_improvement_pct") is not None else None),
        ("Initial success rate", details.get("initial_success_rate")),
        ("Final success rate", details.get("final_success_rate")),
        ("Success improvement", f"{details.get('success_improvement_pct')}%" if details.get("success_improvement_pct") is not None else None),
    ]

    table_rows = []
    for label, value in rows:
        display = "—" if value is None else html.escape(str(value))
        table_rows.append(f"<tr><th>{html.escape(label)}</th><td>{display}</td></tr>")

    notes = details.get("evaluation_notes", [])
    notes_html = ""
    if notes:
        items = "".join(f"<li>{html.escape(note)}</li>" for note in notes)
        notes_html = f'<ul class="notes-list">{items}</ul>'

    return f'<table class="data-table">{"".join(table_rows)}</table>{notes_html}'


def interpretation_text(
    training_status: str | None,
    historical_status: str | None,
    final_recommendation: str | None,
) -> str:
    """
    Construye un párrafo de interpretación que cambia según la
    combinación real de estados, en vez de un texto genérico fijo.
    """
    if training_status is None or historical_status is None:
        return (
            "<p>This run was validated against comparable historical runs whenever "
            "enough history was available. A rejected result does not necessarily mean "
            "training failed; it means at least one key metric fell outside the "
            "acceptable historical range.</p>"
        )

    lines = []

    if training_status == "converged":
        lines.append(
            "The agent showed clear improvement within this run: both reward and "
            "success rate increased meaningfully between the start and the end of training."
        )
    elif training_status == "partially_converged":
        lines.append(
            "The agent showed partial improvement within this run: only one of "
            "reward or success rate improved meaningfully."
        )
    else:
        lines.append(
            "The agent did not show meaningful improvement within this run; "
            "reward and success rate stayed flat or worsened."
        )

    if historical_status == "approved":
        lines.append(
            "Compared against real historical runs with the same configuration, "
            "this run falls within the expected range."
        )
    elif historical_status == "warning":
        lines.append(
            "Compared against historical runs, this run is borderline: at least one "
            "metric is slightly outside the expected range, or there isn't enough "
            "comparable history yet."
        )
    else:
        lines.append(
            "Compared against historical runs with the same configuration, this run "
            "falls clearly outside the expected range on at least one key metric."
        )

    if training_status == "converged" and historical_status == "rejected":
        lines.append(
            "This combination usually means the agent is learning correctly but "
            "hasn't had enough training time (episodes) to reach historical performance "
            "yet — not that the setup is broken."
        )
    elif training_status == "not_converged" and historical_status == "approved":
        lines.append(
            "This combination is worth a second look: the run passed historical "
            "validation without showing clear internal improvement, which can happen "
            "with very little comparable history."
        )

    paragraphs = "".join(f"<p>{html.escape(line)}</p>" for line in lines)
    return paragraphs


FONT_IMPORT_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&"
    "family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap"
)


def generate_html_report(
    new_run: dict,
    validation_report: dict,
    evaluation_report: dict | None,
    figures: dict,
    output_dir: Path,
) -> Path:
    run_id = str(new_run["run_id"])
    final_status = str(validation_report.get("final_status", "unknown"))

    training_status = evaluation_report.get("training_status") if evaluation_report else None
    historical_status = evaluation_report.get("historical_validation_status") if evaluation_report else final_status
    final_recommendation = evaluation_report.get("final_recommendation") if evaluation_report else final_status

    config_keys = ["run_id", "algorithm", "scenario", "reward_version", "validation_mode", "episodes", "seed", "notes"]
    metric_keys = ["success_rate", "avg_reward", "first_reach_step", "collisions", "avg_steps", "training_time_sec"]
    validation_keys = ["final_status", "history_rows_used", "min_history_runs", "preliminary_validation", "validated_at"]

    figure_html = ""
    for figure in figures.values():
        span_class = "figure-card--full" if figure["size"] == "full" else "figure-card--half"
        figure_html += f"""
        <figure class="figure-card {span_class}">
            <img src="{html.escape(figure['filename'])}" alt="{html.escape(figure['label'])}">
        </figure>
        """

    convergence_section = ""
    if evaluation_report is not None:
        convergence_section = f"""
        <section class="card">
            <div class="section-head">
                <span class="section-number">04</span>
                <h2>Convergence analysis</h2>
            </div>
            {convergence_detail_table(evaluation_report)}
        </section>
        """

    figures_number = "05" if evaluation_report is not None else "04"
    interpretation_number = "06" if evaluation_report is not None else "05"

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>RL MLOps run report — {html.escape(run_id)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="{FONT_IMPORT_URL}" rel="stylesheet">
<style>
* {{ box-sizing: border-box; }}

body {{
    font-family: 'Plus Jakarta Sans', sans-serif;
    margin: 0;
    padding: 48px 24px 80px;
    background: {PALETTE['bg']};
    color: {PALETTE['ink']};
}}

.page {{
    max-width: 1080px;
    margin: 0 auto;
}}

.masthead {{
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 24px;
    padding-bottom: 28px;
    border-bottom: 2px solid {PALETTE['ink']};
    margin-bottom: 28px;
}}

.masthead .eyebrow {{
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {PALETTE['accent']};
    font-weight: 700;
    margin: 0 0 10px;
}}

.masthead h1 {{
    font-family: 'Fraunces', serif;
    font-size: 34px;
    font-weight: 800;
    margin: 0;
    line-height: 1.15;
}}

.masthead .run-id {{
    font-size: 13px;
    color: {PALETTE['ink_soft']};
    margin: 8px 0 0;
}}

.masthead .timestamp {{
    text-align: right;
    font-size: 12px;
    color: {PALETTE['ink_soft']};
}}

.card {{
    background: {PALETTE['card']};
    border: 1px solid {PALETTE['border']};
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 24px;
}}

.section-head {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 16px;
}}

.section-number {{
    font-family: 'Fraunces', serif;
    font-size: 15px;
    font-weight: 700;
    color: {PALETTE['accent2']};
    background: {PALETTE['accent2_soft']};
    border-radius: 8px;
    padding: 3px 9px;
}}

.section-head h2 {{
    font-family: 'Fraunces', serif;
    font-size: 20px;
    font-weight: 700;
    margin: 0;
}}

table.data-table {{
    width: 100%;
    border-collapse: collapse;
}}

table.data-table th,
table.data-table td {{
    text-align: left;
    padding: 11px 14px;
    border-bottom: 1px solid {PALETTE['border']};
    font-size: 13.5px;
}}

table.data-table th {{
    width: 240px;
    color: {PALETTE['ink_soft']};
    font-weight: 600;
}}

table.data-table td {{
    font-weight: 600;
}}

table.metrics-table th,
table.metrics-table td {{
    width: auto;
}}

table.metrics-table thead th {{
    color: {PALETTE['ink_soft']};
    font-size: 11.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
    border-bottom: 2px solid {PALETTE['ink']};
}}

.muted {{
    color: {PALETTE['ink_soft']};
    font-weight: 400 !important;
}}

.notes-list {{
    margin: 14px 0 0;
    padding-left: 20px;
    color: {PALETTE['ink_soft']};
    font-size: 13px;
    line-height: 1.6;
}}

.figures-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}}

.figure-card {{
    margin: 0;
    background: {PALETTE['bg']};
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 10px;
    box-sizing: border-box;
}}

.figure-card--full {{
    flex: 1 1 100%;
    width: 100%;
}}

.figure-card--half {{
    flex: 1 1 calc(50% - 10px);
    min-width: 280px;
}}

.figure-card img {{
    width: 100%;
    border-radius: 8px;
    display: block;
}}

.interpretation p {{
    font-size: 14.5px;
    line-height: 1.75;
    margin: 0 0 12px;
}}

.interpretation p:last-child {{
    margin-bottom: 0;
}}

.footer {{
    text-align: center;
    font-size: 12px;
    color: {PALETTE['ink_soft']};
    margin-top: 36px;
    padding-top: 20px;
    border-top: 1px solid {PALETTE['border']};
}}

@media (max-width: 760px) {{
    .masthead {{ grid-template-columns: 1fr; }}
    .masthead .timestamp {{ text-align: left; }}
    .figures-grid {{ flex-direction: column; }}
    .figure-card--half {{ flex: 1 1 100%; }}
}}
</style>
</head>
<body>
<div class="page">

    <div class="masthead">
        <div>
            <p class="eyebrow">RL MLOps · run diagnostic</p>
            <h1>Training run report</h1>
            <p class="run-id">{html.escape(run_id)}</p>
        </div>
        <div class="timestamp">
            Generated<br>{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>

    {diagnostic_panel(training_status, historical_status, final_recommendation)}

    <section class="card" style="margin-top:28px;">
        <div class="section-head">
            <span class="section-number">01</span>
            <h2>Run configuration</h2>
        </div>
        {dict_to_table(new_run, config_keys)}
    </section>

    <section class="card">
        <div class="section-head">
            <span class="section-number">02</span>
            <h2>Final performance metrics</h2>
        </div>
        {dict_to_table(new_run, metric_keys)}
    </section>

    <section class="card">
        <div class="section-head">
            <span class="section-number">03</span>
            <h2>Historical validation</h2>
        </div>
        {dict_to_table(validation_report, validation_keys)}
        <div style="margin-top:18px;">
            {validation_metrics_table(validation_report)}
        </div>
    </section>

    {convergence_section}

    <section class="card">
        <div class="section-head">
            <span class="section-number">{figures_number}</span>
            <h2>Training curves</h2>
        </div>
        <div class="figures-grid">
            {figure_html if figure_html else '<p class="muted">No training figures available.</p>'}
        </div>
    </section>

    <section class="card interpretation">
        <div class="section-head">
            <span class="section-number">{interpretation_number}</span>
            <h2>Interpretation</h2>
        </div>
        {interpretation_text(training_status, historical_status, final_recommendation)}
    </section>

    <div class="footer">
        Generated by the RL MLOps historical validation pipeline.
    </div>

</div>
</body>
</html>
"""

    output_path = output_dir / "run_report.html"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report_html)

    return output_path


def save_summary_json(
    new_run: dict,
    validation_report: dict,
    evaluation_report: dict | None,
    figures: dict,
    output_dir: Path,
) -> Path:
    summary = {
        "run_id": new_run.get("run_id"),
        "algorithm": new_run.get("algorithm"),
        "scenario": new_run.get("scenario"),
        "reward_version": new_run.get("reward_version"),
        "validation_mode": new_run.get("validation_mode"),
        "final_status": validation_report.get("final_status"),
        "training_status": evaluation_report.get("training_status") if evaluation_report else None,
        "historical_validation_status": evaluation_report.get("historical_validation_status") if evaluation_report else None,
        "final_recommendation": evaluation_report.get("final_recommendation") if evaluation_report else None,
        "metrics": {
            "success_rate": new_run.get("success_rate"),
            "avg_reward": new_run.get("avg_reward"),
            "first_reach_step": new_run.get("first_reach_step"),
            "collisions": new_run.get("collisions"),
            "avg_steps": new_run.get("avg_steps"),
            "training_time_sec": new_run.get("training_time_sec"),
        },
        "figures": list(figures.keys()),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = output_dir / "summary.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return output_path


def main():
    new_run = load_new_run()
    validation_report = load_validation_report()
    evaluation_report = load_evaluation_report()

    run_id = str(new_run["run_id"])
    output_dir = FRIENDLY_REPORTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics_path = find_episode_metrics(run_id)

    figures = {}
    if episode_metrics_path is not None:
        figures = generate_training_figures(run_id, episode_metrics_path, output_dir)

    report_path = generate_html_report(
        new_run=new_run,
        validation_report=validation_report,
        evaluation_report=evaluation_report,
        figures=figures,
        output_dir=output_dir,
    )

    summary_path = save_summary_json(
        new_run=new_run,
        validation_report=validation_report,
        evaluation_report=evaluation_report,
        figures=figures,
        output_dir=output_dir,
    )

    print("Reporte generado correctamente.")
    print(f"run_id: {run_id}")
    print(f"HTML: {report_path}")
    print(f"Summary JSON: {summary_path}")

    if evaluation_report is None:
        print(
            "\nNota: no se encontró reports/evaluation/evaluation_result.json. "
            "Corre src/evaluation/evaluate_run.py antes de este script para incluir "
            "el análisis de convergencia en el reporte."
        )


if __name__ == "__main__":
    main()
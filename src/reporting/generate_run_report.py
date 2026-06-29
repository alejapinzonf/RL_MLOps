from pathlib import Path
from datetime import datetime
import json
import html

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEW_RUN_PATH = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
VALIDATION_REPORT_PATH = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"
TRAINING_REPORTS_DIR = PROJECT_ROOT / "reports" / "training"
FRIENDLY_REPORTS_DIR = PROJECT_ROOT / "reports" / "friendly"


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


def find_episode_metrics(run_id: str) -> Path | None:
    path = TRAINING_REPORTS_DIR / f"{run_id}_episode_metrics.csv"
    return path if path.exists() else None


def moving_average(values: pd.Series, window: int = 20) -> pd.Series:
    return values.rolling(window=window, min_periods=1).mean()


def save_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    ylabel: str,
    output_path: Path,
    moving_avg: bool = False,
):
    plt.figure(figsize=(9, 4.5))

    y_values = df[y_col]

    if moving_avg:
        y_values = moving_average(y_values)

    plt.plot(df[x_col], y_values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
            df=df,
            x_col="episode",
            y_col="episode_reward",
            title="Episode reward during training",
            ylabel="Episode reward",
            output_path=reward_path,
            moving_avg=True,
        )
        figures["episode_reward"] = reward_path.name

    if "success" in df.columns:
        success_path = output_dir / "success_rate_moving.png"
        df = df.copy()
        df["success_rate_moving"] = moving_average(df["success"], window=20)

        save_plot(
            df=df,
            x_col="episode",
            y_col="success_rate_moving",
            title="Moving success rate during training",
            ylabel="Success rate",
            output_path=success_path,
            moving_avg=False,
        )
        figures["success_rate_moving"] = success_path.name

    if "collisions" in df.columns:
        collisions_path = output_dir / "collisions.png"
        save_plot(
            df=df,
            x_col="episode",
            y_col="collisions",
            title="Collisions during training",
            ylabel="Collisions",
            output_path=collisions_path,
            moving_avg=True,
        )
        figures["collisions"] = collisions_path.name

    return figures


def format_status_badge(status: str) -> str:
    status = str(status).lower()

    if status == "approved":
        label = "APPROVED"
        color = "#1b7f3a"
    elif status == "warning":
        label = "WARNING"
        color = "#b7791f"
    elif status == "rejected":
        label = "REJECTED"
        color = "#b91c1c"
    else:
        label = status.upper()
        color = "#4b5563"

    return (
        f'<span style="background:{color}; color:white; padding:6px 10px; '
        f'border-radius:8px; font-weight:bold;">{label}</span>'
    )


def dict_to_table(data: dict, keys: list[str]) -> str:
    rows = []

    for key in keys:
        value = data.get(key, "")
        rows.append(
            "<tr>"
            f"<th>{html.escape(str(key))}</th>"
            f"<td>{html.escape(str(value))}</td>"
            "</tr>"
        )

    return "<table>" + "\n".join(rows) + "</table>"


def validation_metrics_table(validation_report: dict) -> str:
    metric_results = validation_report.get("metric_results", [])

    if not metric_results:
        return "<p>No metric-level validation was available.</p>"

    rows = [
        "<tr>"
        "<th>Metric</th>"
        "<th>Status</th>"
        "<th>New value</th>"
        "<th>Historical mean</th>"
        "<th>Historical std</th>"
        "<th>Reason</th>"
        "</tr>"
    ]

    for result in metric_results:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('metric', '')))}</td>"
            f"<td>{html.escape(str(result.get('status', '')))}</td>"
            f"<td>{html.escape(str(result.get('new_value', '')))}</td>"
            f"<td>{html.escape(str(result.get('historical_mean', '')))}</td>"
            f"<td>{html.escape(str(result.get('historical_std', '')))}</td>"
            f"<td>{html.escape(str(result.get('reason', '')))}</td>"
            "</tr>"
        )

    return "<table>" + "\n".join(rows) + "</table>"


def generate_html_report(
    new_run: dict,
    validation_report: dict,
    figures: dict,
    output_dir: Path,
) -> Path:
    run_id = str(new_run["run_id"])
    final_status = str(validation_report.get("final_status", "unknown"))

    config_keys = [
        "run_id",
        "algorithm",
        "scenario",
        "reward_version",
        "validation_mode",
        "episodes",
        "seed",
        "notes",
    ]

    metric_keys = [
        "success_rate",
        "avg_reward",
        "first_reach_step",
        "collisions",
        "avg_steps",
        "training_time_sec",
    ]

    validation_keys = [
        "final_status",
        "history_rows_used",
        "min_history_runs",
        "preliminary_validation",
        "validated_at",
    ]

    figure_html = ""

    for title, filename in figures.items():
        figure_html += f"""
        <div class="figure-card">
            <h3>{html.escape(title)}</h3>
            <img src="{html.escape(filename)}" alt="{html.escape(title)}">
        </div>
        """

    report_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>RL MLOps Run Report - {html.escape(run_id)}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            background: #f8fafc;
            color: #111827;
        }}

        h1, h2, h3 {{
            color: #111827;
        }}

        .container {{
            max-width: 1100px;
            margin: auto;
        }}

        .card {{
            background: white;
            padding: 22px;
            border-radius: 14px;
            margin-bottom: 22px;
            box-shadow: 0 3px 14px rgba(0, 0, 0, 0.08);
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 12px;
        }}

        th, td {{
            border-bottom: 1px solid #e5e7eb;
            text-align: left;
            padding: 10px;
            vertical-align: top;
        }}

        th {{
            background: #f3f4f6;
            width: 260px;
        }}

        img {{
            max-width: 100%;
            border-radius: 10px;
            border: 1px solid #e5e7eb;
        }}

        .figure-card {{
            margin-top: 18px;
        }}

        .small {{
            color: #6b7280;
            font-size: 0.9em;
        }}

        .status {{
            margin-top: 14px;
            margin-bottom: 14px;
        }}

        .footer {{
            margin-top: 30px;
            color: #6b7280;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
<div class="container">

    <div class="card">
        <h1>RL MLOps Run Report</h1>
        <p class="small">Generated at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Run ID:</strong> {html.escape(run_id)}</p>
        <div class="status">
            {format_status_badge(final_status)}
        </div>
    </div>

    <div class="card">
        <h2>1. Run configuration</h2>
        {dict_to_table(new_run, config_keys)}
    </div>

    <div class="card">
        <h2>2. Final performance metrics</h2>
        {dict_to_table(new_run, metric_keys)}
    </div>

    <div class="card">
        <h2>3. Historical validation summary</h2>
        {dict_to_table(validation_report, validation_keys)}
    </div>

    <div class="card">
        <h2>4. Metric-level validation</h2>
        {validation_metrics_table(validation_report)}
    </div>

    <div class="card">
        <h2>5. Training curves</h2>
        {figure_html if figure_html else "<p>No training figures available.</p>"}
    </div>

    <div class="card">
        <h2>6. Interpretation</h2>
        <p>
            This run was trained/evaluated using the selected algorithm, scenario,
            reward version and validation mode. The final result was compared against
            historical runs with matching configuration whenever enough history was available.
        </p>
        <p>
            A rejected result does not necessarily mean the training failed; it means
            that at least one key metric fell outside the acceptable historical range
            defined by the validation rules.
        </p>
    </div>

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
        "metrics": {
            "success_rate": new_run.get("success_rate"),
            "avg_reward": new_run.get("avg_reward"),
            "first_reach_step": new_run.get("first_reach_step"),
            "collisions": new_run.get("collisions"),
            "avg_steps": new_run.get("avg_steps"),
            "training_time_sec": new_run.get("training_time_sec"),
        },
        "history_rows_used": validation_report.get("history_rows_used"),
        "preliminary_validation": validation_report.get("preliminary_validation"),
        "figures": figures,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_path = output_dir / "summary.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    return output_path


def main():
    new_run = load_new_run()
    validation_report = load_validation_report()

    run_id = str(new_run["run_id"])

    output_dir = FRIENDLY_REPORTS_DIR / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_metrics_path = find_episode_metrics(run_id)

    figures = {}

    if episode_metrics_path is not None:
        figures = generate_training_figures(
            run_id=run_id,
            episode_metrics_path=episode_metrics_path,
            output_dir=output_dir,
        )

    html_path = generate_html_report(
        new_run=new_run,
        validation_report=validation_report,
        figures=figures,
        output_dir=output_dir,
    )

    summary_path = save_summary_json(
        new_run=new_run,
        validation_report=validation_report,
        figures=figures,
        output_dir=output_dir,
    )

    print("Reporte amigable generado.")
    print(f"run_id: {run_id}")
    print(f"HTML report: {html_path}")
    print(f"Summary JSON: {summary_path}")

    if figures:
        print("\nFiguras:")
        for name, filename in figures.items():
            print(f"- {name}: {output_dir / filename}")


if __name__ == "__main__":
    main()

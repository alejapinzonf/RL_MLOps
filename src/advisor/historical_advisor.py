"""
historical_advisor.py

Compara los hiperparámetros que el usuario está a punto de usar para
entrenar (algoritmo, escenario, episodios) contra el histórico REAL de
corridas ya ejecutadas (data/raw/historical_results_paper.csv y/o
data/processed/historical_results_clean.csv), para avisar si es
probable que la corrida no alcance el desempeño esperado por falta de
episodios, o si simplemente no hay precedente para comparar.

A diferencia de validate_run.py (que valida una corrida YA ejecutada),
este advisor corre ANTES de entrenar, sobre los argumentos planeados.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

HISTORICAL_SOURCES = [
    PROJECT_ROOT / "data" / "processed" / "historical_results_clean.csv",
    PROJECT_ROOT / "data" / "raw" / "historical_results_paper.csv",
]


def load_available_history() -> pd.DataFrame:
    """
    Carga y combina todas las fuentes históricas disponibles. Ignora
    silenciosamente las que no existan (p. ej. si el usuario aún no
    corrió clean_data.py o import_paper_excels.py).
    """
    frames = []

    for path in HISTORICAL_SOURCES:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                continue

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def advise(
    algorithm: str,
    scenario: str,
    planned_episodes: int,
    reward_version: str | None = None,
) -> dict:
    """
    Compara planned_episodes contra la distribución real del histórico
    (percentiles p25/p50/p75) para el mismo algorithm + scenario.

    Clasifica en 4 niveles basados en distribución estadística honesta:
        - "no_history": sin precedentes comparables
        - "likely_insufficient": por debajo del p25 histórico
        - "comparable": entre p25 y p75 histórico
        - "exceeds_history": por encima del p75 histórico

    También informa sobre la variabilidad del success_rate histórico,
    para avisar cuando los resultados pasados fueron inconsistentes.
    """
    history = load_available_history()

    if history.empty:
        return {
            "n_matches": 0,
            "verdict": "no_history",
            "message": (
                "No se encontró ningún histórico cargado todavía. Corre "
                "load_data.py + clean_data.py, o import_paper_excels.py, "
                "antes de entrenar si quieres una referencia."
            ),
        }

    filtered = history[
        (history["algorithm"] == algorithm) & (history["scenario"] == scenario)
    ]

    if reward_version is not None and "reward_version" in filtered.columns:
        reward_matches = filtered[filtered["reward_version"] == reward_version]
        if not reward_matches.empty:
            filtered = reward_matches

    if filtered.empty:
        return {
            "n_matches": 0,
            "verdict": "no_history",
            "message": (
                f"No hay corridas históricas para algorithm={algorithm}, "
                f"scenario={scenario}. No se puede comparar el número de "
                f"episodios planeado contra precedentes reales."
            ),
        }

    episodes_series = pd.to_numeric(filtered["episodes"], errors="coerce").dropna()
    success_series = pd.to_numeric(filtered.get("success_rate"), errors="coerce").dropna()

    if episodes_series.empty:
        return {
            "n_matches": int(len(filtered)),
            "verdict": "comparable",
            "message": (
                f"Se encontraron {len(filtered)} corridas históricas, pero sin "
                "información de episodios para comparar."
            ),
        }

    # Estadísticas de distribución real
    hist_p25 = float(episodes_series.quantile(0.25))
    hist_p50 = float(episodes_series.quantile(0.50))
    hist_p75 = float(episodes_series.quantile(0.75))
    hist_mean = float(episodes_series.mean())
    hist_min = float(episodes_series.min())
    hist_max = float(episodes_series.max())

    success_mean = float(success_series.mean()) if not success_series.empty else None
    success_std = float(success_series.std()) if len(success_series) > 1 else None
    high_variance = success_std is not None and success_std > 0.15

    # Clasificación basada en percentiles (estadísticamente honesta)
    if planned_episodes < hist_p25:
        verdict = "likely_insufficient"
        pct_of_median = (planned_episodes / hist_p50 * 100) if hist_p50 > 0 else 0
        message = (
            f"Con {planned_episodes:,} episodios estás por debajo del percentil 25 del "
            f"histórico real para {algorithm}/{scenario} (p25={hist_p25:,.0f}, "
            f"mediana={hist_p50:,.0f}, p75={hist_p75:,.0f}, n={len(filtered)} corridas). "
            f"Solo alcanzas el {pct_of_median:.1f}% de la mediana histórica — "
            f"es probable que el agente no tenga tiempo suficiente para aprender."
        )
    elif planned_episodes <= hist_p75:
        verdict = "comparable"
        message = (
            f"Tus {planned_episodes:,} episodios están dentro del rango habitual "
            f"para {algorithm}/{scenario} (p25={hist_p25:,.0f}–p75={hist_p75:,.0f}, "
            f"mediana={hist_p50:,.0f}, n={len(filtered)} corridas). "
            f"El histórico sugiere que esta cantidad es razonable."
        )
    else:
        verdict = "exceeds_history"
        message = (
            f"Tus {planned_episodes:,} episodios superan el percentil 75 del histórico "
            f"para {algorithm}/{scenario} (p75={hist_p75:,.0f}, mediana={hist_p50:,.0f}, "
            f"n={len(filtered)} corridas). Esto puede mejorar la convergencia, "
            f"pero considera el tiempo de cómputo adicional."
        )

    if success_mean is not None:
        message += f" Success rate histórico promedio: {success_mean:.3f}"
        if success_std is not None:
            message += f" (±{success_std:.3f})"

    if high_variance:
        message += (
            f". ⚠ Alta variabilidad en el histórico (std={success_std:.3f}): "
            f"los resultados pasados fueron inconsistentes — el desempeño final "
            f"dependerá fuertemente de la configuración exacta."
        )

    return {
        "n_matches": int(len(filtered)),
        "historical_episodes_mean": hist_mean,
        "historical_episodes_min": hist_min,
        "historical_episodes_max": hist_max,
        "historical_episodes_p25": hist_p25,
        "historical_episodes_p50": hist_p50,
        "historical_episodes_p75": hist_p75,
        "historical_success_rate_mean": success_mean,
        "historical_success_rate_std": success_std,
        "high_variance_warning": high_variance,
        "verdict": verdict,
        "message": message,
    }

    return {
        "n_matches": int(len(filtered)),
        "historical_episodes_mean": hist_mean,
        "historical_episodes_min": hist_min,
        "historical_episodes_max": hist_max,
        "historical_success_rate_mean": success_mean,
        "verdict": verdict,
        "message": message,
    }


def format_advice(advice: dict) -> str:
    icons = {
        "no_history": "ℹ️",
        "likely_insufficient": "🔴",
        "comparable": "🟢",
        "exceeds_history": "🟡",
    }
    icon = icons.get(advice["verdict"], "•")
    return f"{icon} {advice['message']}"


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compara hiperparámetros planeados contra el histórico real antes de entrenar."
    )
    parser.add_argument("--algorithm", required=True, choices=["q_learning", "dqn", "sac"])
    parser.add_argument("--scenario", required=True, choices=["wall", "l_shape", "u_shape"])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--reward-version", default=None)

    args = parser.parse_args()

    advice = advise(
        algorithm=args.algorithm,
        scenario=args.scenario,
        planned_episodes=args.episodes,
        reward_version=args.reward_version,
    )

    print(format_advice(advice))


if __name__ == "__main__":
    main()

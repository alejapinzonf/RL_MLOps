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
    Compara planned_episodes contra el histórico real para el mismo
    algorithm + scenario (y reward_version si se especifica).

    Devuelve un dict con:
        - n_matches: cantidad de corridas históricas comparables
        - historical_episodes_mean / min / max
        - historical_success_rate_mean
        - verdict: "no_history" | "likely_insufficient" | "comparable" | "exceeds_history"
        - message: texto explicativo
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

    hist_mean = float(episodes_series.mean())
    hist_min = float(episodes_series.min())
    hist_max = float(episodes_series.max())
    success_mean = float(success_series.mean()) if not success_series.empty else None

    ratio = planned_episodes / hist_mean if hist_mean > 0 else float("inf")

    if ratio < 0.1:
        verdict = "likely_insufficient"
        message = (
            f"Vas a entrenar {algorithm} en {scenario} con {planned_episodes:,} episodios. "
            f"El histórico real usa en promedio {hist_mean:,.0f} episodios "
            f"(rango {hist_min:,.0f}–{hist_max:,.0f}) en {len(filtered)} corrida(s) comparables"
            + (f", con success_rate promedio {success_mean:.3f}" if success_mean is not None else "")
            + f". Tu plan usa solo {ratio*100:.1f}% de ese promedio — es muy probable que "
              "el agente no alcance un desempeño comparable."
        )
    elif ratio < 0.5:
        verdict = "likely_insufficient"
        message = (
            f"Vas a entrenar {algorithm} en {scenario} con {planned_episodes:,} episodios, "
            f"bastante por debajo del promedio histórico real ({hist_mean:,.0f} episodios, "
            f"{len(filtered)} corrida(s) comparables"
            + (f", success_rate promedio {success_mean:.3f}" if success_mean is not None else "")
            + "). Es razonable esperar un agente parcialmente entrenado."
        )
    elif ratio <= 1.5:
        verdict = "comparable"
        message = (
            f"El número de episodios planeado ({planned_episodes:,}) es comparable al "
            f"histórico real para {algorithm}/{scenario} ({hist_mean:,.0f} episodios "
            f"promedio, {len(filtered)} corrida(s)"
            + (f", success_rate promedio {success_mean:.3f}" if success_mean is not None else "")
            + ")."
        )
    else:
        verdict = "exceeds_history"
        message = (
            f"Vas a entrenar con {planned_episodes:,} episodios, más que el promedio "
            f"histórico ({hist_mean:,.0f}) para {algorithm}/{scenario}. Esto es razonable "
            "si buscas mejorar sobre el precedente, pero considera el tiempo de cómputo."
        )

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
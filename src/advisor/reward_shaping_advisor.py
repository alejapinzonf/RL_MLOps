from dataclasses import dataclass, field

@dataclass
class RewardWarning:
    severity: str  # "high" | "medium" | "low"
    message: str
    ratio: float

DOMINANCE_RULES = [
    {
        "name": "arrival_vs_obstacle_hit",
        "numerator": "arrival_bonus_multiplier",
        "denominator": "obstacle_hit_penalty",
        "high_threshold": 5.0,
        "medium_threshold": 2.0,
        "message": (
            "El premio por llegar al goal ({num:.1f}) es {ratio:.1f}x mayor que "
            "el castigo por chocar con un obstáculo ({den:.1f}). El agente podría "
            "aprender a ignorar obstáculos para llegar más rápido, en vez de "
            "evitarlos."
        ),
    },
    {
        "name": "arrival_vs_move_after_goal",
        "numerator": "arrival_bonus_multiplier",
        "denominator": "goal_stay_out_penalty",
        "high_threshold": 10.0,
        "medium_threshold": 4.0,
        "message": (
            "El premio por llegar ({num:.1f}) es {ratio:.1f}x mayor que el castigo "
            "por moverse después de haber llegado ({den:.1f}). El agente podría "
            "deambular sin penalización real una vez alcanzado el goal."
        ),
    },
    {
        "name": "stay_outside_vs_goal_stay",
        "numerator": "stay_outside_penalty",
        "denominator": "goal_stay_reward",
        "high_threshold": 0.3,
        "medium_threshold": 0.7,
        "invert": True,
        "message": (
            "El castigo por quedarse quieto fuera del goal ({num:.1f}) es muy bajo "
            "comparado con el premio por quedarse quieto en el goal ({den:.1f}) "
            "(razón={ratio:.2f}). El agente podría no tener suficiente incentivo "
            "para moverse en vez de quedarse parado en cualquier celda."
        ),
    },
    {
        "name": "step_penalty_vs_goal_position_scale",
        "numerator": "goal_position_scale",
        "denominator": "step_penalty",
        "high_threshold": 100.0,
        "medium_threshold": 40.0,
        "message": (
            "La escala de progreso hacia el goal ({num:.1f}) es {ratio:.1f}x mayor "
            "que la penalización por paso ({den:.1f}). El step penalty podría "
            "volverse insignificante, permitiendo que el agente dé vueltas sin "
            "consecuencia real antes de avanzar."
        ),
    },
]


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("inf") if numerator > 0 else 0.0
    return numerator / denominator


def analyze_reward_params(reward_params: dict) -> list[RewardWarning]:
    warnings: list[RewardWarning] = []

    for rule in DOMINANCE_RULES:
        numerator_value = reward_params.get(rule["numerator"])
        denominator_value = reward_params.get(rule["denominator"])

        if numerator_value is None or denominator_value is None:
            continue

        ratio = _ratio(numerator_value, denominator_value)
        invert = rule.get("invert", False)

        # Para reglas invertidas, el riesgo aparece cuando la razón es
        # BAJA (no alta) — p. ej. "el castigo es demasiado bajo
        # respecto al premio".
        if invert:
            if ratio <= rule["high_threshold"]:
                severity = "high"
            elif ratio <= rule["medium_threshold"]:
                severity = "medium"
            else:
                continue
        else:
            if ratio >= rule["high_threshold"]:
                severity = "high"
            elif ratio >= rule["medium_threshold"]:
                severity = "medium"
            else:
                continue

        message = rule["message"].format(
            num=numerator_value,
            den=denominator_value,
            ratio=ratio,
        )

        warnings.append(RewardWarning(severity=severity, message=message, ratio=ratio))

    return warnings


def format_warnings(warnings: list[RewardWarning]) -> str:
    if not warnings:
        return "No se detectaron desbalances evidentes entre los términos de reward."

    icons = {"high": "🔴", "medium": "🟡", "low": "🟢"}
    lines = ["Posibles riesgos en la función de recompensa:\n"]

    # Mostrar primero las de mayor severidad.
    order = {"high": 0, "medium": 1, "low": 2}
    for warning in sorted(warnings, key=lambda w: order[w.severity]):
        icon = icons.get(warning.severity, "•")
        lines.append(f"{icon} [{warning.severity.upper()}] {warning.message}")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analiza los parámetros de reward antes de entrenar."
    )
    parser.add_argument("--arrival-reward", type=float, default=150.0)
    parser.add_argument("--goal-stay-reward", type=float, default=50.0)
    parser.add_argument("--goal-stay-out-penalty", type=float, default=100.0)
    parser.add_argument("--step-penalty", type=float, default=1.0)
    parser.add_argument("--stay-outside-penalty", type=float, default=60.0)
    parser.add_argument("--obstacle-hit-penalty", type=float, default=100.0)
    parser.add_argument("--goal-position-scale", type=float, default=2.0)
    parser.add_argument("--obstacle-position-scale", type=float, default=2.0)
    parser.add_argument("--arrival-bonus-multiplier", type=float, default=100.0)

    args = parser.parse_args()

    reward_params = {
        "arrival_reward": args.arrival_reward,
        "goal_stay_reward": args.goal_stay_reward,
        "goal_stay_out_penalty": args.goal_stay_out_penalty,
        "step_penalty": args.step_penalty,
        "stay_outside_penalty": args.stay_outside_penalty,
        "obstacle_hit_penalty": args.obstacle_hit_penalty,
        "goal_position_scale": args.goal_position_scale,
        "obstacle_position_scale": args.obstacle_position_scale,
        "arrival_bonus_multiplier": args.arrival_bonus_multiplier,
    }

    warnings = analyze_reward_params(reward_params)
    print(format_warnings(warnings))


if __name__ == "__main__":
    main()

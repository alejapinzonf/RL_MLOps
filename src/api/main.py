from pathlib import Path
from datetime import datetime
import asyncio
import json
import subprocess
import sys
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from advisor.historical_advisor import advise as historical_advise
from advisor.reward_shaping_advisor import analyze_reward_params


app = FastAPI(title="RL MLOps Training API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRIENDLY_REPORTS_DIR = PROJECT_ROOT / "reports" / "friendly"
FRIENDLY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports", StaticFiles(directory=str(FRIENDLY_REPORTS_DIR)), name="reports")


TRAINER_REGISTRY = {
    "demo": {
        "script": "src/training/demo_train.py",
        "fixed_algorithm": None,
        "reward_version": None,
        "supports_reward_params": False,
        "label": "Demo (simulated, fast)",
        "extra_params": [],
    },
    "q_learning": {
        "script": "src/training/train_q_learning.py",
        "fixed_algorithm": "q_learning",
        "reward_version": None,
        "supports_reward_params": False,
        "label": "Q-Learning (simple grid)",
        "extra_params": [],
    },
    "q_learning_full": {
        "script": "src/training/train_q_learning_paper.py",
        "fixed_algorithm": "q_learning",
        "reward_version": "reward_v1_base",
        "supports_reward_params": True,
        "label": "Q-Learning (full environment)",
        "extra_params": [
            {"name": "alpha", "flag": "--alpha", "type": "float", "default": 0.5, "label": "Learning rate"},
            {"name": "gamma", "flag": "--gamma", "type": "float", "default": 0.95, "label": "Discount factor (gamma)"},
            {"name": "epsilon_end", "flag": "--epsilon-end", "type": "float", "default": 0.05, "label": "Epsilon decay target"},
        ],
    },
    "dqn": {
        "script": "src/training/train_dqn_paper.py",
        "fixed_algorithm": "dqn",
        "reward_version": "reward_v1_base",
        "supports_reward_params": True,
        "label": "DQN",
        "extra_params": [
            {"name": "learning_rate", "flag": "--learning-rate", "type": "float", "default": 0.0001, "label": "Learning rate"},
            {"name": "gamma", "flag": "--gamma", "type": "float", "default": 0.95, "label": "Discount factor (gamma)"},
            {"name": "epsilon_decay_fraction", "flag": "--epsilon-decay-fraction", "type": "float", "default": 0.90, "label": "Epsilon decay fraction"},
        ],
    },
    "sac": {
        "script": "src/training/train_sac_paper.py",
        "fixed_algorithm": "sac",
        "reward_version": "reward_v1_base",
        "supports_reward_params": True,
        "label": "Discrete SAC",
        "extra_params": [
            {"name": "learning_rate", "flag": "--learning-rate", "type": "float", "default": 0.0001, "label": "Learning rate"},
            {"name": "gamma", "flag": "--gamma", "type": "float", "default": 0.95, "label": "Discount factor (gamma)"},
        ],
    },
}

REWARD_PARAM_FIELDS = [
    {"name": "step_penalty", "flag": "--step-penalty", "default": 1.0, "label": "Step penalty"},
    {"name": "obstacle_hit_penalty", "flag": "--obstacle-hit-penalty", "default": 100.0, "label": "Obstacle collision penalty"},
    {"name": "stay_outside_penalty", "flag": "--stay-outside-penalty", "default": 60.0, "label": "Penalty for staying outside the goal"},
    {"name": "arrival_reward", "flag": "--arrival-reward", "default": 150.0, "label": "Reward for reaching the goal"},
    {"name": "arrival_bonus_multiplier", "flag": "--arrival-bonus-multiplier", "default": 100.0, "label": "Arrival bonus multiplier"},
    {"name": "goal_stay_reward", "flag": "--goal-stay-reward", "default": 50.0, "label": "Reward for staying at the goal"},
    {"name": "goal_stay_out_penalty", "flag": "--goal-stay-out-penalty", "default": 100.0, "label": "Penalty for leaving the goal"},
    {"name": "goal_position_scale", "flag": "--goal-position-scale", "default": 2.0, "label": "Goal-progress reward scale"},
    {"name": "obstacle_position_scale", "flag": "--obstacle-position-scale", "default": 2.0, "label": "Obstacle-avoidance reward scale"},
]

DEFAULT_REWARD_PARAMS = {field["name"]: field["default"] for field in REWARD_PARAM_FIELDS}


class AdvisorRequest(BaseModel):
    trainer: str
    scenario: str
    episodes: int
    reward_params: dict[str, float] = {}


class TrainingRequest(BaseModel):
    trainer: str
    scenario: str
    episodes: int
    checkpoint_interval: int = 1000
    seed: int = 42
    validation_mode: str = "normal"
    extra_params: dict[str, float] = {}
    reward_params: dict[str, float] = {}


class TrainingJob:
    """Estado en memoria de un entrenamiento lanzado desde la API."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.lines: list[str] = []
        self.done = False
        self.returncode: int | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    def append_line(self, line: str):
        with self._lock:
            self.lines.append(line)

    def get_lines_from(self, index: int) -> list[str]:
        with self._lock:
            return list(self.lines[index:])


JOBS: dict[str, TrainingJob] = {}


@app.get("/trainers")
def list_trainers():
    return {
        name: {
            "label": cfg["label"],
            "supports_reward_params": cfg["supports_reward_params"],
            "extra_params": cfg["extra_params"],
        }
        for name, cfg in TRAINER_REGISTRY.items()
    }


@app.get("/reward-params/defaults")
def get_reward_param_defaults():
    return {"fields": REWARD_PARAM_FIELDS}


@app.post("/advisor/check")
def check_advisors(request: AdvisorRequest):
    if request.trainer not in TRAINER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown trainer: {request.trainer}")

    trainer_config = TRAINER_REGISTRY[request.trainer]
    effective_algorithm = trainer_config["fixed_algorithm"] or "q_learning"
    effective_reward_version = trainer_config["reward_version"] or "reward_v1_base"

    historical_advice = historical_advise(
        algorithm=effective_algorithm,
        scenario=request.scenario,
        planned_episodes=request.episodes,
        reward_version=effective_reward_version,
    )

    reward_warnings = []
    if trainer_config["supports_reward_params"]:
        effective_reward_params = {**DEFAULT_REWARD_PARAMS, **request.reward_params}
        warnings = analyze_reward_params(effective_reward_params)
        reward_warnings = [
            {"severity": w.severity, "message": w.message} for w in warnings
        ]

    return {
        "historical": historical_advice,
        "reward_shaping": reward_warnings,
    }


def _run_training_subprocess(job: TrainingJob, commands: list[list[str]]):
    try:
        for command in commands:
            job.append_line(f"\n$ {' '.join(command)}")

            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                job.append_line(line.rstrip("\n"))

            process.wait()
            job.returncode = process.returncode

            if process.returncode != 0:
                job.append_line(f"\n[ERROR] Step failed with exit code {process.returncode}.")
                return

        job.append_line("\n[DONE] Training finished successfully.")

    except Exception as exc:
        job.error = str(exc)
        job.append_line(f"\n[ERROR] {exc}")
    finally:
        job.done = True


def build_pipeline_commands(trainer_config: dict, request: TrainingRequest) -> list[list[str]]:
    python_executable = sys.executable
    training_command = [python_executable, trainer_config["script"]]

    if trainer_config["fixed_algorithm"] is None:
        training_command.extend(["--algorithm", "q_learning"])

    training_command.extend(["--scenario", request.scenario])

    if trainer_config["reward_version"] is None:
        training_command.extend(["--reward-version", "reward_v1_base"])

    if trainer_config["script"].endswith("demo_train.py"):
        training_command.extend(["--validation-mode", request.validation_mode])

    training_command.extend(["--episodes", str(request.episodes)])
    training_command.extend(["--checkpoint-interval", str(request.checkpoint_interval)])
    training_command.extend(["--seed", str(request.seed)])

    for param in trainer_config["extra_params"]:
        value = request.extra_params.get(param["name"], param["default"])
        training_command.extend([param["flag"], str(value)])

    if trainer_config["supports_reward_params"]:
        for field in REWARD_PARAM_FIELDS:
            value = request.reward_params.get(field["name"], field["default"])
            training_command.extend([field["flag"], str(value)])

    return [
        training_command,
        [python_executable, "src/validation/validate_run.py"],
        [python_executable, "src/evaluation/evaluate_run.py"],
        [python_executable, "src/reporting/generate_run_report.py"],
        [python_executable, "src/tracking/log_to_mlflow.py"],
    ]


@app.post("/training/start")
def start_training(request: TrainingRequest):
    if request.trainer not in TRAINER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown trainer: {request.trainer}")

    trainer_config = TRAINER_REGISTRY[request.trainer]
    commands = build_pipeline_commands(trainer_config, request)

    job_id = str(uuid.uuid4())
    job = TrainingJob(job_id)
    JOBS[job_id] = job

    thread = threading.Thread(target=_run_training_subprocess, args=(job, commands), daemon=True)
    thread.start()

    return {"job_id": job_id}


@app.get("/training/stream/{job_id}")
async def stream_training(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]

    async def event_generator():
        sent = 0
        while True:
            new_lines = job.get_lines_from(sent)
            for line in new_lines:
                sent += 1
                yield f"data: {json.dumps({'line': line})}\n\n"

            if job.done and sent >= len(job.lines):
                yield f"data: {json.dumps({'event': 'done', 'returncode': job.returncode})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/training/result/{job_id}")
def get_training_result(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")

    job = JOBS[job_id]

    if not job.done:
        return {"done": False}

    new_run_path = PROJECT_ROOT / "data" / "new_runs" / "new_run_demo.csv"
    validation_path = PROJECT_ROOT / "reports" / "validation" / "validation_result.json"
    evaluation_path = PROJECT_ROOT / "reports" / "evaluation" / "evaluation_result.json"

    result = {"done": True, "returncode": job.returncode}

    run_id = None
    try:
        import pandas as pd
        new_run = pd.read_csv(new_run_path).iloc[0].to_dict()
        result["metrics"] = new_run
        run_id = new_run.get("run_id")
    except Exception:
        result["metrics"] = None

    try:
        with open(validation_path, "r", encoding="utf-8") as file:
            result["validation"] = json.load(file)
    except Exception:
        result["validation"] = None

    try:
        with open(evaluation_path, "r", encoding="utf-8") as file:
            result["evaluation"] = json.load(file)
    except Exception:
        result["evaluation"] = None

    result["figures"] = {}
    if run_id:
        figure_dir = FRIENDLY_REPORTS_DIR / run_id
        for figure_name, file_name in [
            ("episode_reward", "episode_reward.png"),
            ("success_rate", "success_rate_moving.png"),
            ("collisions", "collisions.png"),
        ]:
            if (figure_dir / file_name).exists():
                result["figures"][figure_name] = f"/reports/{run_id}/{file_name}"

    return result

@app.get("/mlflow-url")
def get_mlflow_url():
    return {"url": "http://127.0.0.1:5000"}


@app.get("/history")
def get_history():

    import pandas as pd

    sources = [
        PROJECT_ROOT / "data" / "processed" / "historical_results_clean.csv",
        PROJECT_ROOT / "data" / "raw" / "historical_results_paper.csv",
    ]

    frames = []
    for path in sources:
        if path.exists():
            try:
                df = pd.read_csv(path)
                df["source_type"] = "paper" if "paper" in path.name else "pipeline"
                frames.append(df)
            except Exception:
                pass

    if not frames:
        return {"runs": [], "total": 0}

    combined = pd.concat(frames, ignore_index=True)
    ombined = combined.fillna("").replace({float("nan"): None})

    numeric_cols = ["success_rate", "avg_reward", "collisions", "first_reach_step", "avg_steps", "training_time_sec", "episodes"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.fillna("").infer_objects(copy=False)

    return {
        "runs": combined.to_dict(orient="records"),
        "total": len(combined),
    }

@app.get("/history/summary")
def get_history_summary():
    """
    Devuelve estadísticas agregadas por algoritmo y escenario
    para la pestaña de comparación de historial.
    """
    import pandas as pd

    sources = [
        PROJECT_ROOT / "data" / "processed" / "historical_results_clean.csv",
        PROJECT_ROOT / "data" / "raw" / "historical_results_paper.csv",
    ]

    frames = []
    for path in sources:
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass

    if not frames:
        return {"by_algorithm": [], "by_scenario": []}

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["run_id"], keep="first")

    for col in ["success_rate", "avg_reward", "collisions"]:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    by_algo = (
        combined.groupby("algorithm")[["success_rate", "avg_reward", "collisions"]]
        .agg(["mean", "count"])
        .reset_index()
    )
    by_algo.columns = ["_".join(c).strip("_") for c in by_algo.columns]
    by_algo = by_algo.rename(columns={"algorithm_": "algorithm"})

    by_scenario = (
        combined.groupby("scenario")[["success_rate", "avg_reward"]]
        .mean()
        .reset_index()
    )

    return {
        "by_algorithm": by_algo.where(by_algo.notna(), None).to_dict(orient="records"),
        "by_scenario": by_scenario.where(by_scenario.notna(), None).to_dict(orient="records"),
    }

@app.get("/mlflow/status")
def get_mlflow_status():
    import urllib.request
    try:
        urllib.request.urlopen("http://localhost:5000", timeout=1)
        return {"running": True, "url": "http://localhost:5000"}
    except Exception:
        return {"running": False, "url": "http://localhost:5000"}


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend-app" / "dist"

if FRONTEND_DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST_DIR), html=True), name="frontend")

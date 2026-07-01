import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

const PALETTE = {
  bg: "#FBF8FC",
  card: "#FFFFFF",
  ink: "#2D2438",
  inkSoft: "#6E6280",
  border: "#E9E1F2",
  accent: "#9B7FD4",
  accentSoft: "#EFE7FB",
  accent2: "#D4729A",
  accent2Soft: "#FCEAF1",
  consoleBg: "#1A1625",
  consoleText: "#C9BFE0",
  approvedFg: "#2F7A49",
  approvedBg: "#E7F5EC",
  approvedBorder: "#BFE3CC",
  warningFg: "#9C7415",
  warningBg: "#FBF1DE",
  warningBorder: "#F0DBA8",
  rejectedFg: "#A8415C",
  rejectedBg: "#FBE8ED",
  rejectedBorder: "#F0BFCD",
};

const TRAINER_OPTIONS = [
  { value: "demo", label: "Demo (simulated, fast)" },
  { value: "q_learning", label: "Q-Learning (simple grid)" },
  { value: "q_learning_full", label: "Q-Learning (full environment)" },
  { value: "dqn", label: "DQN" },
  { value: "sac", label: "Discrete SAC" },
];

const REWARD_PARAM_FIELDS = [
  { name: "step_penalty", default: 1.0, label: "Step penalty" },
  { name: "obstacle_hit_penalty", default: 100.0, label: "Obstacle collision penalty" },
  { name: "stay_outside_penalty", default: 60.0, label: "Penalty for staying outside the goal" },
  { name: "arrival_reward", default: 150.0, label: "Reward for reaching the goal" },
  { name: "arrival_bonus_multiplier", default: 100.0, label: "Arrival bonus multiplier" },
  { name: "goal_stay_reward", default: 50.0, label: "Reward for staying at the goal" },
  { name: "goal_stay_out_penalty", default: 100.0, label: "Penalty for leaving the goal" },
  { name: "goal_position_scale", default: 2.0, label: "Goal-progress reward scale" },
  { name: "obstacle_position_scale", default: 2.0, label: "Obstacle-avoidance reward scale" },
];

const TRAINER_EXTRA_PARAMS = {
  demo: [],
  q_learning: [],
  q_learning_full: [
    { name: "alpha", default: 0.5, label: "Learning rate" },
    { name: "gamma", default: 0.95, label: "Discount factor (gamma)" },
    { name: "epsilon_end", default: 0.05, label: "Epsilon decay target" },
  ],
  dqn: [
    { name: "learning_rate", default: 0.0001, label: "Learning rate" },
    { name: "gamma", default: 0.95, label: "Discount factor (gamma)" },
    { name: "epsilon_decay_fraction", default: 0.9, label: "Epsilon decay fraction" },
  ],
  sac: [
    { name: "learning_rate", default: 0.0001, label: "Learning rate" },
    { name: "gamma", default: 0.95, label: "Discount factor (gamma)" },
  ],
};

const TRAINERS_WITH_REWARD_PARAMS = new Set(["q_learning_full", "dqn", "sac"]);

const SCENARIO_OPTIONS = [
  { value: "wall", label: "Wall" },
  { value: "l_shape", label: "L-shape" },
  { value: "u_shape", label: "U-shape" },
];

function statusColors(status) {
  const s = (status || "").toLowerCase();
  if (s === "approved" || s === "comparable" || s === "converged") {
    return { fg: PALETTE.approvedFg, bg: PALETTE.approvedBg, border: PALETTE.approvedBorder };
  }
  if (s === "warning" || s === "likely_insufficient" || s === "partially_converged" || s === "exceeds_history") {
    return { fg: PALETTE.warningFg, bg: PALETTE.warningBg, border: PALETTE.warningBorder };
  }
  if (s === "rejected" || s === "not_converged") {
    return { fg: PALETTE.rejectedFg, bg: PALETTE.rejectedBg, border: PALETTE.rejectedBorder };
  }
  return { fg: PALETTE.inkSoft, bg: PALETTE.accentSoft, border: PALETTE.border };
}

function StatusChip({ status, size = "md" }) {
  if (!status) return null;
  const { fg, bg, border } = statusColors(status);
  const label = String(status).toUpperCase().replace(/_/g, " ");
  return (
    <span
      style={{
        display: "inline-block",
        background: bg,
        color: fg,
        border: `1px solid ${border}`,
        padding: size === "sm" ? "3px 10px" : "5px 14px",
        borderRadius: 999,
        fontWeight: 700,
        fontSize: size === "sm" ? 11 : 12.5,
        letterSpacing: "0.04em",
      }}
    >
      {label}
    </span>
  );
}

function SeverityDot({ severity }) {
  const colors = { high: "#D4607A", medium: "#D9A23B", low: "#9B7FD4" };
  return (
    <span
      style={{
        display: "inline-block",
        width: 8,
        height: 8,
        borderRadius: "50%",
        background: colors[severity] || PALETTE.inkSoft,
        marginRight: 8,
        flexShrink: 0,
        marginTop: 5,
      }}
    />
  );
}

export default function TrainingConsoleApp() {
  const [form, setForm] = useState({
    trainer: "q_learning_full",
    scenario: "wall",
    episodes: 5000,
    checkpoint_interval: 1000,
    seed: 42,
  });

  const [extraParams, setExtraParams] = useState({});
  const [rewardParams, setRewardParams] = useState({});

  useEffect(() => {
    const defaults = {};
    (TRAINER_EXTRA_PARAMS[form.trainer] || []).forEach((p) => {
      defaults[p.name] = p.default;
    });
    setExtraParams(defaults);

    if (TRAINERS_WITH_REWARD_PARAMS.has(form.trainer)) {
      const rewardDefaults = {};
      REWARD_PARAM_FIELDS.forEach((f) => {
        rewardDefaults[f.name] = f.default;
      });
      setRewardParams(rewardDefaults);
    } else {
      setRewardParams({});
    }
  }, [form.trainer]);

  const [advisor, setAdvisor] = useState(null);
  const [advisorLoading, setAdvisorLoading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState(null);

  const consoleEndRef = useRef(null);
  const debounceRef = useRef(null);
  const eventSourceRef = useRef(null);

  const fetchAdvisor = useCallback(async (formState, rewardParamsState) => {
    setAdvisorLoading(true);
    setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/advisor/check`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trainer: formState.trainer,
          scenario: formState.scenario,
          episodes: Number(formState.episodes),
          reward_params: rewardParamsState,
        }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      setAdvisor(data);
    } catch (err) {
      setApiError(
        "Could not reach the training API at localhost:8000. Make sure the FastAPI server is running."
      );
      setAdvisor(null);
    } finally {
      setAdvisorLoading(false);
    }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      fetchAdvisor(form, rewardParams);
    }, 350);
    return () => clearTimeout(debounceRef.current);
  }, [form, rewardParams, fetchAdvisor]);

  useEffect(() => {
    consoleEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logLines]);

  function updateField(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
  }

  function updateExtraParam(name, value) {
    setExtraParams((prev) => ({ ...prev, [name]: value }));
  }

  function updateRewardParam(name, value) {
    setRewardParams((prev) => ({ ...prev, [name]: value }));
  }

  async function handleStartTraining() {
    setLogLines([]);
    setResult(null);
    setTraining(true);
    setApiError(null);

    try {
      const res = await fetch(`${API_BASE}/training/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trainer: form.trainer,
          scenario: form.scenario,
          episodes: Number(form.episodes),
          checkpoint_interval: Number(form.checkpoint_interval),
          seed: Number(form.seed),
          extra_params: extraParams,
          reward_params: rewardParams,
        }),
      });
      if (!res.ok) throw new Error(`API returned ${res.status}`);
      const data = await res.json();
      setJobId(data.job_id);

      const es = new EventSource(`${API_BASE}/training/stream/${data.job_id}`);
      eventSourceRef.current = es;

      es.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.line !== undefined) {
          setLogLines((prev) => [...prev, payload.line]);
        }
        if (payload.event === "done") {
          es.close();
          setTraining(false);
          fetchResult(data.job_id);
        }
      };

      es.onerror = () => {
        es.close();
        setTraining(false);
      };
    } catch (err) {
      setApiError(
        "Could not reach the training API at localhost:8000. Make sure the FastAPI server is running."
      );
      setTraining(false);
    }
  }

  async function fetchResult(id) {
    try {
      const res = await fetch(`${API_BASE}/training/result/${id}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      // silent: the console already shows the raw log
    }
  }

  const historicalAdvice = advisor?.historical;
  const rewardWarnings = advisor?.reward_shaping || [];

  return (
    <div
      style={{
        fontFamily: "'Plus Jakarta Sans', sans-serif",
        background: PALETTE.bg,
        color: PALETTE.ink,
        minHeight: "100vh",
        padding: "40px 24px 80px",
      }}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,700;9..144,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        @keyframes blink { 0%, 50% { opacity: 1; } 50.01%, 100% { opacity: 0; } }
        .cursor-blink { animation: blink 1s step-start infinite; }
        select, input { font-family: 'Plus Jakarta Sans', sans-serif; }
      `}</style>

      <div style={{ maxWidth: 1080, margin: "0 auto" }}>
        <div
          style={{
            paddingBottom: 24,
            borderBottom: `2px solid ${PALETTE.ink}`,
            marginBottom: 28,
          }}
        >
          <p
            style={{
              margin: "0 0 8px",
              fontSize: 12,
              letterSpacing: "0.1em",
              textTransform: "uppercase",
              color: PALETTE.accent,
              fontWeight: 700,
            }}
          >
            RL MLOps · training console
          </p>
          <h1
            style={{
              fontFamily: "'Fraunces', serif",
              fontSize: 32,
              fontWeight: 800,
              margin: 0,
            }}
          >
            Plan a training run
          </h1>
          <p style={{ color: PALETTE.inkSoft, fontSize: 14, margin: "8px 0 0" }}>
            Choose your setup. The advisor checks it against real historical runs as you type.
          </p>
        </div>

        {apiError && (
          <div
            style={{
              background: PALETTE.rejectedBg,
              border: `1px solid ${PALETTE.rejectedBorder}`,
              borderRadius: 12,
              padding: "14px 18px",
              marginBottom: 24,
              color: PALETTE.rejectedFg,
              fontSize: 13.5,
            }}
          >
            {apiError}
          </div>
        )}

        <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
          {/* Form card */}
          <div
            style={{
              flex: "1 1 380px",
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              borderRadius: 18,
              padding: 28,
            }}
          >
            <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 18, margin: "0 0 18px" }}>
              Configuration
            </h2>

            <Field label="Trainer">
              <select
                value={form.trainer}
                onChange={(e) => updateField("trainer", e.target.value)}
                style={selectStyle}
              >
                {TRAINER_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Scenario">
              <select
                value={form.scenario}
                onChange={(e) => updateField("scenario", e.target.value)}
                style={selectStyle}
              >
                {SCENARIO_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            </Field>

            <Field label="Episodes">
              <input
                type="number"
                min={1}
                value={form.episodes}
                onChange={(e) => updateField("episodes", e.target.value)}
                style={inputStyle}
              />
            </Field>

            <Field label="Checkpoint interval">
              <input
                type="number"
                min={1}
                value={form.checkpoint_interval}
                onChange={(e) => updateField("checkpoint_interval", e.target.value)}
                style={inputStyle}
              />
            </Field>

            <Field label="Seed">
              <input
                type="number"
                value={form.seed}
                onChange={(e) => updateField("seed", e.target.value)}
                style={inputStyle}
              />
            </Field>

            {(TRAINER_EXTRA_PARAMS[form.trainer] || []).length > 0 && (
              <>
                <Divider label="Algorithm parameters" />
                {TRAINER_EXTRA_PARAMS[form.trainer].map((p) => (
                  <Field label={p.label} key={p.name}>
                    <input
                      type="number"
                      step="any"
                      value={extraParams[p.name] ?? p.default}
                      onChange={(e) => updateExtraParam(p.name, e.target.value)}
                      style={inputStyle}
                    />
                  </Field>
                ))}
              </>
            )}

            {TRAINERS_WITH_REWARD_PARAMS.has(form.trainer) && (
              <>
                <Divider label="Reward function" />
                {REWARD_PARAM_FIELDS.map((f) => (
                  <Field label={f.label} key={f.name}>
                    <input
                      type="number"
                      step="any"
                      value={rewardParams[f.name] ?? f.default}
                      onChange={(e) => updateRewardParam(f.name, e.target.value)}
                      style={inputStyle}
                    />
                  </Field>
                ))}
              </>
            )}

            <button
              onClick={handleStartTraining}
              disabled={training}
              style={{
                width: "100%",
                marginTop: 8,
                padding: "13px 0",
                borderRadius: 12,
                border: "none",
                background: training ? PALETTE.border : PALETTE.ink,
                color: training ? PALETTE.inkSoft : "#fff",
                fontWeight: 700,
                fontSize: 14.5,
                cursor: training ? "default" : "pointer",
                transition: "background 0.15s ease",
              }}
            >
              {training ? "Training in progress…" : "Start training"}
            </button>
          </div>

          {/* Advisor card */}
          <div
            style={{
              flex: "1 1 380px",
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              borderRadius: 18,
              padding: 28,
            }}
          >
            <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 18, margin: "0 0 18px" }}>
              Advisor
              {advisorLoading && (
                <span style={{ fontSize: 12, color: PALETTE.inkSoft, marginLeft: 10, fontWeight: 400 }}>
                  checking…
                </span>
              )}
            </h2>

            {historicalAdvice && (
              <div style={{ marginBottom: 18 }}>
                <p
                  style={{
                    margin: "0 0 8px",
                    fontSize: 11,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: PALETTE.inkSoft,
                    fontWeight: 700,
                  }}
                >
                  Historical comparison
                </p>
                <div style={{ marginBottom: 8 }}>
                  <StatusChip status={historicalAdvice.verdict} />
                </div>
                <p style={{ fontSize: 13, lineHeight: 1.6, color: PALETTE.ink, margin: 0 }}>
                  {historicalAdvice.message}
                </p>
              </div>
            )}

            <div>
              <p
                style={{
                  margin: "0 0 8px",
                  fontSize: 11,
                  letterSpacing: "0.06em",
                  textTransform: "uppercase",
                  color: PALETTE.inkSoft,
                  fontWeight: 700,
                }}
              >
                Reward shaping
              </p>
              {rewardWarnings.length === 0 ? (
                <p style={{ fontSize: 13, color: PALETTE.inkSoft, margin: 0 }}>
                  {advisor ? "No obvious imbalance detected." : "—"}
                </p>
              ) : (
                rewardWarnings.map((w, i) => (
                  <div key={i} style={{ display: "flex", marginBottom: 10 }}>
                    <SeverityDot severity={w.severity} />
                    <p style={{ fontSize: 12.5, lineHeight: 1.55, margin: 0, color: PALETTE.ink }}>
                      {w.message}
                    </p>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Console */}
        {(logLines.length > 0 || training) && (
          <div
            style={{
              marginTop: 24,
              background: PALETTE.consoleBg,
              borderRadius: 18,
              padding: "20px 22px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12.5,
              lineHeight: 1.7,
              color: PALETTE.consoleText,
              maxHeight: 360,
              overflowY: "auto",
            }}
          >
            {logLines.map((line, i) => (
              <div key={i} style={{ whiteSpace: "pre-wrap" }}>
                {line}
              </div>
            ))}
            {training && (
              <span className="cursor-blink" style={{ color: PALETTE.accent }}>
                ▋
              </span>
            )}
            <div ref={consoleEndRef} />
          </div>
        )}

        {/* Result */}
        {result && result.evaluation && (
          <div
            style={{
              marginTop: 24,
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              borderRadius: 18,
              padding: 28,
            }}
          >
            <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 18, margin: "0 0 18px" }}>
              Result
            </h2>
            <div style={{ display: "flex", gap: 10, marginBottom: 18, flexWrap: "wrap" }}>
              <StatusChip status={result.evaluation.training_status} size="sm" />
              <StatusChip status={result.evaluation.historical_validation_status} size="sm" />
              <StatusChip status={result.evaluation.final_recommendation} />
            </div>
            {result.metrics && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))",
                  gap: 14,
                }}
              >
                <Metric label="Success rate" value={result.metrics.success_rate} />
                <Metric label="Avg reward" value={result.metrics.avg_reward} />
                <Metric label="Collisions" value={result.metrics.collisions} />
                <Metric label="Episodes" value={result.metrics.episodes} />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function Divider({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "20px 0 14px" }}>
      <span
        style={{
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: PALETTE.accent2,
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </span>
      <div style={{ flex: 1, height: 1, background: PALETTE.border }} />
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 16 }}>
      <label
        style={{
          display: "block",
          fontSize: 12,
          fontWeight: 600,
          color: PALETTE.inkSoft,
          marginBottom: 6,
        }}
      >
        {label}
      </label>
      {children}
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div
      style={{
        background: PALETTE.bg,
        border: `1px solid ${PALETTE.border}`,
        borderRadius: 12,
        padding: "14px 16px",
      }}
    >
      <p style={{ margin: "0 0 4px", fontSize: 11, color: PALETTE.inkSoft, fontWeight: 600 }}>
        {label}
      </p>
      <p style={{ margin: 0, fontSize: 20, fontWeight: 800, fontFamily: "'Fraunces', serif" }}>
        {typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : value}
      </p>
    </div>
  );
}

const selectStyle = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 10,
  border: `1px solid ${PALETTE.border}`,
  fontSize: 13.5,
  color: PALETTE.ink,
  background: "#fff",
};

const inputStyle = {
  width: "100%",
  padding: "10px 12px",
  borderRadius: 10,
  border: `1px solid ${PALETTE.border}`,
  fontSize: 13.5,
  color: PALETTE.ink,
  boxSizing: "border-box",
};

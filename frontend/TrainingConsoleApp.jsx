import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8000";

// ─── PALETTE ────────────────────────────────────────────────────────────────
const P = {
  bg: "#FBF8FC", card: "#FFFFFF", ink: "#2D2438", inkSoft: "#6E6280",
  border: "#E9E1F2", accent: "#9B7FD4", accentSoft: "#EFE7FB",
  accent2: "#D4729A", accent2Soft: "#FCEAF1",
  consoleBg: "#1A1625", consoleText: "#C9BFE0",
  approvedFg: "#2F7A49", approvedBg: "#E7F5EC", approvedBorder: "#BFE3CC",
  warningFg: "#9C7415", warningBg: "#FBF1DE", warningBorder: "#F0DBA8",
  rejectedFg: "#A8415C", rejectedBg: "#FBE8ED", rejectedBorder: "#F0BFCD",
};

// ─── STATIC DATA ────────────────────────────────────────────────────────────
const TRAINER_OPTIONS = [
  { value: "demo", label: "Demo (simulated)" },
  { value: "q_learning", label: "Q-Learning (simple grid)" },
  { value: "q_learning_full", label: "Q-Learning (full env)" },
  { value: "dqn", label: "DQN" },
  { value: "sac", label: "Discrete SAC" },
];

const SCENARIO_OPTIONS = [
  { value: "wall", label: "Wall" },
  { value: "l_shape", label: "L-shape" },
  { value: "u_shape", label: "U-shape" },
];

const TRAINER_EXTRA_PARAMS = {
  demo: [], q_learning: [],
  q_learning_full: [
    { name: "alpha", default: 0.5, label: "Learning rate (α)" },
    { name: "gamma", default: 0.95, label: "Discount factor (γ)" },
    { name: "epsilon_end", default: 0.05, label: "Epsilon decay target" },
  ],
  dqn: [
    { name: "learning_rate", default: 0.0001, label: "Learning rate" },
    { name: "gamma", default: 0.95, label: "Discount factor (γ)" },
    { name: "epsilon_decay_fraction", default: 0.9, label: "Epsilon decay fraction" },
  ],
  sac: [
    { name: "learning_rate", default: 0.0001, label: "Learning rate" },
    { name: "gamma", default: 0.95, label: "Discount factor (γ)" },
  ],
};

const REWARD_FIELDS = [
  { name: "step_penalty", default: 1.0, label: "Step penalty" },
  { name: "obstacle_hit_penalty", default: 100.0, label: "Collision penalty" },
  { name: "stay_outside_penalty", default: 60.0, label: "Stay-outside penalty" },
  { name: "arrival_reward", default: 150.0, label: "Arrival reward" },
  { name: "arrival_bonus_multiplier", default: 100.0, label: "Arrival bonus ×" },
  { name: "goal_stay_reward", default: 50.0, label: "Goal-stay reward" },
  { name: "goal_stay_out_penalty", default: 100.0, label: "Leave-goal penalty" },
  { name: "goal_position_scale", default: 2.0, label: "Goal-progress scale" },
  { name: "obstacle_position_scale", default: 2.0, label: "Obstacle-avoid scale" },
];

const REWARD_TRAINERS = new Set(["q_learning_full", "dqn", "sac"]);

const ALGO_COLORS = {
  q_learning: "#9B7FD4", dqn: "#D4729A", sac: "#5FA875",
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
function statusColors(s) {
  s = (s || "").toLowerCase();
  if (["approved","comparable","converged"].includes(s))
    return { fg: P.approvedFg, bg: P.approvedBg, border: P.approvedBorder };
  if (["warning","likely_insufficient","partially_converged","exceeds_history"].includes(s))
    return { fg: P.warningFg, bg: P.warningBg, border: P.warningBorder };
  if (["rejected","not_converged"].includes(s))
    return { fg: P.rejectedFg, bg: P.rejectedBg, border: P.rejectedBorder };
  return { fg: P.inkSoft, bg: P.accentSoft, border: P.border };
}

function fmt(v, decimals = 3) {
  if (v == null || v === "") return "—";
  const n = parseFloat(v);
  return isNaN(n) ? String(v) : n.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

// ─── TINY COMPONENTS ─────────────────────────────────────────────────────────
function StatusChip({ status, size = "md" }) {
  if (!status) return null;
  const { fg, bg, border } = statusColors(status);
  return (
    <span style={{
      display: "inline-block", background: bg, color: fg,
      border: `1px solid ${border}`,
      padding: size === "sm" ? "2px 8px" : "5px 14px",
      borderRadius: 999, fontWeight: 700,
      fontSize: size === "sm" ? 10.5 : 12,
      letterSpacing: "0.04em",
    }}>
      {String(status).toUpperCase().replace(/_/g, " ")}
    </span>
  );
}

function Card({ children, style }) {
  return (
    <div style={{
      background: P.card, border: `1px solid ${P.border}`,
      borderRadius: 18, padding: 24, ...style,
    }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{
      fontFamily: "'Fraunces',serif", fontSize: 17,
      fontWeight: 700, margin: "0 0 16px", color: P.ink,
    }}>
      {children}
    </h2>
  );
}

function Divider({ label }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "18px 0 12px" }}>
      <span style={{
        fontSize: 10.5, fontWeight: 700, letterSpacing: "0.07em",
        textTransform: "uppercase", color: P.accent2, whiteSpace: "nowrap",
      }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 1, background: P.border }} />
    </div>
  );
}

function Field({ label, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <label style={{ display: "block", fontSize: 11.5, fontWeight: 600, color: P.inkSoft, marginBottom: 5 }}>
        {label}
      </label>
      {children}
    </div>
  );
}

function Metric({ label, value, color }) {
  return (
    <div style={{
      background: P.bg, border: `1px solid ${P.border}`,
      borderRadius: 12, padding: "12px 14px",
    }}>
      <p style={{ margin: "0 0 3px", fontSize: 10.5, color: P.inkSoft, fontWeight: 600 }}>{label}</p>
      <p style={{ margin: 0, fontSize: 19, fontWeight: 800, fontFamily: "'Fraunces',serif", color: color || P.ink }}>
        {fmt(value)}
      </p>
    </div>
  );
}

const selectStyle = {
  width: "100%", padding: "9px 11px", borderRadius: 10,
  border: `1px solid ${P.border}`, fontSize: 13, color: P.ink, background: "#fff",
  fontFamily: "'Plus Jakarta Sans',sans-serif",
};
const inputStyle = {
  width: "100%", padding: "9px 11px", borderRadius: 10,
  border: `1px solid ${P.border}`, fontSize: 13, color: P.ink,
  boxSizing: "border-box", fontFamily: "'Plus Jakarta Sans',sans-serif",
};

// ─── MINI BAR CHART ─────────────────────────────────────────────────────────
function MiniBar({ data, valueKey, labelKey, colorFn, maxVal }) {
  if (!data || data.length === 0) return <p style={{ color: P.inkSoft, fontSize: 13 }}>No data available</p>;
  const max = maxVal || Math.max(...data.map(d => d[valueKey] || 0));
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {data.map((d, i) => {
        const val = d[valueKey] || 0;
        const pct = max > 0 ? (val / max) * 100 : 0;
        const color = colorFn ? colorFn(d[labelKey]) : P.accent;
        return (
          <div key={i}>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}>
              <span style={{ fontSize: 12, fontWeight: 600, color: P.ink }}>{d[labelKey]}</span>
              <span style={{ fontSize: 12, color: P.inkSoft }}>{fmt(val, 3)}</span>
            </div>
            <div style={{ background: P.accentSoft, borderRadius: 6, height: 10, overflow: "hidden" }}>
              <div style={{ width: `${pct}%`, background: color, height: "100%", borderRadius: 6, transition: "width 0.5s ease" }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── TAB: TRAIN ──────────────────────────────────────────────────────────────
function TrainTab() {
  const [form, setForm] = useState({ trainer: "q_learning_full", scenario: "wall", episodes: 5000, checkpoint_interval: 1000, seed: 42 });
  const [extraParams, setExtraParams] = useState({});
  const [rewardParams, setRewardParams] = useState({});
  const [advisor, setAdvisor] = useState(null);
  const [advisorLoading, setAdvisorLoading] = useState(false);
  const [jobId, setJobId] = useState(null);
  const [logLines, setLogLines] = useState([]);
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [apiError, setApiError] = useState(null);
  const consoleEndRef = useRef(null);
  const debounceRef = useRef(null);

  useEffect(() => {
    const ep = {};
    (TRAINER_EXTRA_PARAMS[form.trainer] || []).forEach(p => { ep[p.name] = p.default; });
    setExtraParams(ep);
    if (REWARD_TRAINERS.has(form.trainer)) {
      const rp = {};
      REWARD_FIELDS.forEach(f => { rp[f.name] = f.default; });
      setRewardParams(rp);
    } else { setRewardParams({}); }
  }, [form.trainer]);

  const fetchAdvisor = useCallback(async (f, rp) => {
    setAdvisorLoading(true); setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/advisor/check`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trainer: f.trainer, scenario: f.scenario, episodes: Number(f.episodes), reward_params: rp }),
      });
      if (!res.ok) throw new Error();
      setAdvisor(await res.json());
    } catch {
      setApiError("Cannot reach localhost:8000. Is the FastAPI server running?");
      setAdvisor(null);
    } finally { setAdvisorLoading(false); }
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => fetchAdvisor(form, rewardParams), 350);
    return () => clearTimeout(debounceRef.current);
  }, [form, rewardParams, fetchAdvisor]);

  useEffect(() => { consoleEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [logLines]);

  async function handleStart() {
    setLogLines([]); setResult(null); setTraining(true); setApiError(null);
    try {
      const res = await fetch(`${API_BASE}/training/start`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trainer: form.trainer, scenario: form.scenario, episodes: Number(form.episodes), checkpoint_interval: Number(form.checkpoint_interval), seed: Number(form.seed), extra_params: extraParams, reward_params: rewardParams }),
      });
      if (!res.ok) throw new Error();
      const { job_id } = await res.json();
      setJobId(job_id);
      const es = new EventSource(`${API_BASE}/training/stream/${job_id}`);
      es.onmessage = (e) => {
        const p = JSON.parse(e.data);
        if (p.line !== undefined) setLogLines(prev => [...prev, p.line]);
        if (p.event === "done") {
          es.close(); setTraining(false);
          fetch(`${API_BASE}/training/result/${job_id}`).then(r => r.json()).then(setResult);
        }
      };
      es.onerror = () => { es.close(); setTraining(false); };
    } catch {
      setApiError("Cannot reach localhost:8000."); setTraining(false);
    }
  }

  const hist = advisor?.historical;
  const rewardWarnings = advisor?.reward_shaping || [];
  const hasHighWarning = rewardWarnings.some(w => w.severity === "high");

  return (
    <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
      {/* LEFT: Form */}
      <Card style={{ flex: "1 1 360px", minWidth: 0 }}>
        <SectionTitle>Configuration</SectionTitle>

        <Field label="Algorithm">
          <select value={form.trainer} onChange={e => setForm(p => ({ ...p, trainer: e.target.value }))} style={selectStyle}>
            {TRAINER_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </Field>
        <Field label="Scenario">
          <select value={form.scenario} onChange={e => setForm(p => ({ ...p, scenario: e.target.value }))} style={selectStyle}>
            {SCENARIO_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </Field>
        <Field label="Episodes">
          <input type="number" min={1} value={form.episodes} onChange={e => setForm(p => ({ ...p, episodes: e.target.value }))} style={inputStyle} />
        </Field>
        <Field label="Checkpoint every">
          <input type="number" min={1} value={form.checkpoint_interval} onChange={e => setForm(p => ({ ...p, checkpoint_interval: e.target.value }))} style={inputStyle} />
        </Field>
        <Field label="Seed">
          <input type="number" value={form.seed} onChange={e => setForm(p => ({ ...p, seed: e.target.value }))} style={inputStyle} />
        </Field>

        {(TRAINER_EXTRA_PARAMS[form.trainer] || []).length > 0 && (
          <>
            <Divider label="Algorithm parameters" />
            {(TRAINER_EXTRA_PARAMS[form.trainer] || []).map(p => (
              <Field label={p.label} key={p.name}>
                <input type="number" step="any" value={extraParams[p.name] ?? p.default} onChange={e => setExtraParams(prev => ({ ...prev, [p.name]: e.target.value }))} style={inputStyle} />
              </Field>
            ))}
          </>
        )}

        {REWARD_TRAINERS.has(form.trainer) && (
          <>
            <Divider label="Reward function" />
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 14px" }}>
              {REWARD_FIELDS.map(f => (
                <Field label={f.label} key={f.name}>
                  <input type="number" step="any" value={rewardParams[f.name] ?? f.default} onChange={e => setRewardParams(prev => ({ ...prev, [f.name]: e.target.value }))} style={inputStyle} />
                </Field>
              ))}
            </div>
          </>
        )}

        <button onClick={handleStart} disabled={training} style={{
          width: "100%", marginTop: 6, padding: "13px 0", borderRadius: 12, border: "none",
          background: training ? P.border : P.ink, color: training ? P.inkSoft : "#fff",
          fontWeight: 700, fontSize: 14, cursor: training ? "default" : "pointer",
          fontFamily: "'Plus Jakarta Sans',sans-serif",
        }}>
          {training ? "Training…" : "▶ Start training"}
        </button>
      </Card>

      {/* RIGHT: Advisor + Console + Result */}
      <div style={{ flex: "1 1 380px", minWidth: 0, display: "flex", flexDirection: "column", gap: 20 }}>

        {apiError && (
          <div style={{ background: P.rejectedBg, border: `1px solid ${P.rejectedBorder}`, borderRadius: 14, padding: "12px 16px", color: P.rejectedFg, fontSize: 13 }}>
            {apiError}
          </div>
        )}

        {/* Advisor */}
        <Card>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", marginBottom: 14 }}>
            <SectionTitle>Advisor</SectionTitle>
            {advisorLoading && <span style={{ fontSize: 11, color: P.inkSoft }}>checking…</span>}
          </div>

          {hist && (
            <div style={{ marginBottom: 14, paddingBottom: 14, borderBottom: `1px solid ${P.border}` }}>
              <p style={{ margin: "0 0 8px", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: P.inkSoft }}>Historical</p>
              <div style={{ marginBottom: 6 }}><StatusChip status={hist.verdict} /></div>
              <p style={{ fontSize: 12.5, lineHeight: 1.6, color: P.ink, margin: 0 }}>{hist.message}</p>
              {hist.n_matches > 0 && (
                <p style={{ fontSize: 11, color: P.inkSoft, margin: "6px 0 0" }}>
                  {hist.n_matches} comparable run(s) · avg success rate: {fmt(hist.historical_success_rate_mean)}
                </p>
              )}
            </div>
          )}

          <p style={{ margin: "0 0 8px", fontSize: 10.5, fontWeight: 700, letterSpacing: "0.06em", textTransform: "uppercase", color: P.inkSoft }}>Reward shaping</p>
          {rewardWarnings.length === 0 ? (
            <p style={{ fontSize: 12.5, color: P.inkSoft, margin: 0 }}>
              {advisor ? "✓ No imbalances detected." : "—"}
            </p>
          ) : rewardWarnings.map((w, i) => (
            <div key={i} style={{ display: "flex", gap: 8, marginBottom: 8 }}>
              <span style={{ fontSize: 13, flexShrink: 0 }}>
                {w.severity === "high" ? "🔴" : w.severity === "medium" ? "🟡" : "🟢"}
              </span>
              <p style={{ fontSize: 12, lineHeight: 1.55, margin: 0, color: P.ink }}>{w.message}</p>
            </div>
          ))}

          {hasHighWarning && (
            <div style={{ marginTop: 12, padding: "10px 12px", background: P.warningBg, borderRadius: 10, border: `1px solid ${P.warningBorder}` }}>
              <p style={{ margin: 0, fontSize: 12, color: P.warningFg, fontWeight: 600 }}>
                ⚠ High-severity imbalance detected. The agent may learn an undesired shortcut.
              </p>
            </div>
          )}
        </Card>

        {/* Console */}
        {(logLines.length > 0 || training) && (
          <Card style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "12px 16px 8px", borderBottom: `1px solid ${P.border}`, display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ fontSize: 11.5, fontWeight: 700, color: P.inkSoft, letterSpacing: "0.04em", textTransform: "uppercase" }}>Training output</span>
              {training && <span style={{ fontSize: 11, color: P.accent2 }}>● live</span>}
            </div>
            <div style={{
              background: P.consoleBg, padding: "14px 18px",
              fontFamily: "'JetBrains Mono',monospace", fontSize: 12, lineHeight: 1.7,
              color: P.consoleText, maxHeight: 320, overflowY: "auto",
            }}>
              {logLines.map((line, i) => <div key={i} style={{ whiteSpace: "pre-wrap" }}>{line}</div>)}
              {training && <span style={{ color: P.accent, animation: "blink 1s step-start infinite" }}>▋</span>}
              <div ref={consoleEndRef} />
            </div>
          </Card>
        )}

        {/* Result */}
        {result?.evaluation && (
          <Card>
            <SectionTitle>Result</SectionTitle>
            <div style={{ display: "flex", gap: 8, marginBottom: 16, flexWrap: "wrap" }}>
              <StatusChip status={result.evaluation.training_status} size="sm" />
              <StatusChip status={result.evaluation.historical_validation_status} size="sm" />
              <StatusChip status={result.evaluation.final_recommendation} />
            </div>
            {result.metrics && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(130px,1fr))", gap: 12 }}>
                <Metric label="Success rate" value={result.metrics.success_rate} color={P.approvedFg} />
                <Metric label="Avg reward" value={result.metrics.avg_reward} />
                <Metric label="Collisions" value={result.metrics.collisions} color={P.rejectedFg} />
                <Metric label="Episodes" value={result.metrics.episodes} />
              </div>
            )}
          </Card>
        )}
      </div>
    </div>
  );
}

// ─── TAB: HISTORY ─────────────────────────────────────────────────────────────
function HistoryTab() {
  const [summary, setSummary] = useState(null);
  const [runs, setRuns] = useState([]);
  const [filter, setFilter] = useState({ algorithm: "all", scenario: "all" });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API_BASE}/history/summary`).then(r => r.json()),
      fetch(`${API_BASE}/history`).then(r => r.json()),
    ]).then(([s, h]) => {
      setSummary(s);
      setRuns(h.runs || []);
      setLoading(false);
    }).catch(() => {
      setError("Cannot reach localhost:8000.");
      setLoading(false);
    });
  }, []);

  const filteredRuns = runs.filter(r => {
    if (filter.algorithm !== "all" && r.algorithm !== filter.algorithm) return false;
    if (filter.scenario !== "all" && r.scenario !== filter.scenario) return false;
    return true;
  });

  const algorithms = [...new Set(runs.map(r => r.algorithm).filter(Boolean))].sort();
  const scenarios = [...new Set(runs.map(r => r.scenario).filter(Boolean))].sort();

  if (loading) return <div style={{ padding: 40, textAlign: "center", color: P.inkSoft }}>Loading history…</div>;
  if (error) return <div style={{ padding: 40, textAlign: "center", color: P.rejectedFg }}>{error}</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      {/* Summary charts */}
      {summary && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 20 }}>
          <Card>
            <SectionTitle>Success rate by algorithm</SectionTitle>
            <MiniBar
              data={(summary.by_algorithm || []).filter(d => d.success_rate_mean != null)}
              valueKey="success_rate_mean"
              labelKey="algorithm"
              colorFn={a => ALGO_COLORS[a] || P.accent}
              maxVal={1}
            />
          </Card>
          <Card>
            <SectionTitle>Avg reward by algorithm</SectionTitle>
            <MiniBar
              data={(summary.by_algorithm || []).filter(d => d.avg_reward_mean != null).map(d => ({
                ...d, avg_reward_mean_abs: Math.abs(d.avg_reward_mean)
              }))}
              valueKey="avg_reward_mean_abs"
              labelKey="algorithm"
              colorFn={a => ALGO_COLORS[a] || P.accent}
            />
          </Card>
        </div>
      )}

      {/* Filter bar */}
      <Card style={{ padding: "14px 20px" }}>
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: P.inkSoft }}>Filter</span>
          <select value={filter.algorithm} onChange={e => setFilter(p => ({ ...p, algorithm: e.target.value }))} style={{ ...selectStyle, width: "auto", padding: "6px 10px" }}>
            <option value="all">All algorithms</option>
            {algorithms.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
          <select value={filter.scenario} onChange={e => setFilter(p => ({ ...p, scenario: e.target.value }))} style={{ ...selectStyle, width: "auto", padding: "6px 10px" }}>
            <option value="all">All scenarios</option>
            {scenarios.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
          <span style={{ fontSize: 12, color: P.inkSoft, marginLeft: "auto" }}>
            {filteredRuns.length} run(s)
          </span>
        </div>
      </Card>

      {/* Runs table */}
      <Card style={{ padding: 0, overflow: "hidden" }}>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12.5 }}>
            <thead>
              <tr style={{ borderBottom: `2px solid ${P.ink}` }}>
                {["Algorithm", "Scenario", "Success rate", "Avg reward", "Collisions", "Episodes"].map(h => (
                  <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: P.inkSoft, fontWeight: 700, fontSize: 11, textTransform: "uppercase", letterSpacing: "0.04em", background: P.bg }}>
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRuns.slice(0, 50).map((r, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${P.border}`, background: i % 2 === 0 ? P.card : P.bg }}>
                  <td style={{ padding: "10px 16px", fontWeight: 600 }}>
                    <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: ALGO_COLORS[r.algorithm] || P.inkSoft, marginRight: 8 }} />
                    {r.algorithm || "—"}
                  </td>
                  <td style={{ padding: "10px 16px", color: P.inkSoft }}>{r.scenario || "—"}</td>
                  <td style={{ padding: "10px 16px" }}>
                    <span style={{ color: parseFloat(r.success_rate) > 0.7 ? P.approvedFg : parseFloat(r.success_rate) > 0.3 ? P.warningFg : P.rejectedFg, fontWeight: 700 }}>
                      {fmt(r.success_rate)}
                    </span>
                  </td>
                  <td style={{ padding: "10px 16px", color: P.inkSoft }}>{fmt(r.avg_reward)}</td>
                  <td style={{ padding: "10px 16px", color: P.inkSoft }}>{fmt(r.collisions)}</td>
                  <td style={{ padding: "10px 16px", color: P.inkSoft }}>{fmt(r.episodes, 0)}</td>
                </tr>
              ))}
              {filteredRuns.length === 0 && (
                <tr><td colSpan={6} style={{ padding: 32, textAlign: "center", color: P.inkSoft }}>No runs found.</td></tr>
              )}
            </tbody>
          </table>
        </div>
        {filteredRuns.length > 50 && (
          <div style={{ padding: "10px 16px", borderTop: `1px solid ${P.border}`, fontSize: 12, color: P.inkSoft, textAlign: "center" }}>
            Showing 50 of {filteredRuns.length} runs
          </div>
        )}
      </Card>
    </div>
  );
}

// ─── TAB: MLFLOW ─────────────────────────────────────────────────────────────
function MLflowTab() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/mlflow/status`).then(r => r.json()).then(setStatus).catch(() => setStatus({ running: false }));
  }, []);

  const handleOpen = () => window.open("http://localhost:5000", "_blank");
  const handleRefresh = () => {
    setStatus(null);
    fetch(`${API_BASE}/mlflow/status`).then(r => r.json()).then(setStatus).catch(() => setStatus({ running: false }));
  };

  return (
    <div style={{ maxWidth: 600, margin: "0 auto" }}>
      <Card>
        <SectionTitle>MLflow Experiment Tracker</SectionTitle>
        <p style={{ fontSize: 13.5, lineHeight: 1.7, color: P.inkSoft, margin: "0 0 20px" }}>
          MLflow tracks every training run automatically — metrics, parameters, artifacts, and model versions. Open the MLflow UI to explore and compare runs visually.
        </p>

        <div style={{
          background: status?.running ? P.approvedBg : P.warningBg,
          border: `1px solid ${status?.running ? P.approvedBorder : P.warningBorder}`,
          borderRadius: 14, padding: "16px 20px", marginBottom: 20,
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          <div>
            <p style={{ margin: "0 0 4px", fontWeight: 700, fontSize: 14, color: status?.running ? P.approvedFg : P.warningFg }}>
              {status == null ? "Checking…" : status.running ? "✓ MLflow is running" : "MLflow is not running"}
            </p>
            <p style={{ margin: 0, fontSize: 12, color: P.inkSoft }}>
              {status?.running ? "Ready at http://localhost:5000" : "Start it with the command below"}
            </p>
          </div>
          <button onClick={handleRefresh} style={{
            background: "transparent", border: `1px solid ${P.border}`, borderRadius: 8,
            padding: "6px 12px", fontSize: 12, color: P.inkSoft, cursor: "pointer",
          }}>
            Refresh
          </button>
        </div>

        {status?.running ? (
          <button onClick={handleOpen} style={{
            width: "100%", padding: "13px 0", borderRadius: 12, border: "none",
            background: P.ink, color: "#fff", fontWeight: 700, fontSize: 14,
            cursor: "pointer", fontFamily: "'Plus Jakarta Sans',sans-serif",
          }}>
            Open MLflow UI ↗
          </button>
        ) : (
          <>
            <p style={{ fontSize: 12.5, fontWeight: 600, color: P.inkSoft, margin: "0 0 8px" }}>
              Run this in a terminal to start MLflow:
            </p>
            <div style={{
              background: P.consoleBg, borderRadius: 12, padding: "14px 18px",
              fontFamily: "'JetBrains Mono',monospace", fontSize: 13,
              color: P.consoleText, marginBottom: 16,
            }}>
              python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
            </div>
            <button onClick={handleOpen} style={{
              width: "100%", padding: "13px 0", borderRadius: 12, border: `1px solid ${P.border}`,
              background: P.bg, color: P.ink, fontWeight: 700, fontSize: 14,
              cursor: "pointer", fontFamily: "'Plus Jakarta Sans',sans-serif",
            }}>
              Try opening anyway ↗
            </button>
          </>
        )}

        <div style={{ marginTop: 24, paddingTop: 20, borderTop: `1px solid ${P.border}` }}>
          <p style={{ margin: "0 0 8px", fontSize: 11.5, fontWeight: 700, color: P.inkSoft, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            What MLflow tracks
          </p>
          {["Success rate, avg reward, collisions, steps per episode", "Algorithm, scenario, seed, and all hyperparameters", "Trained model artifacts and checkpoints", "Historical validation status and convergence verdict"].map((item, i) => (
            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 6, alignItems: "flex-start" }}>
              <span style={{ color: P.accent, fontSize: 14, lineHeight: 1.5 }}>·</span>
              <span style={{ fontSize: 13, color: P.ink, lineHeight: 1.5 }}>{item}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ─── APP ROOT ─────────────────────────────────────────────────────────────────
const TABS = [
  { id: "train", label: "🚀  Train" },
  { id: "history", label: "📊  History" },
  { id: "mlflow", label: "🔬  MLflow" },
];

export default function App() {
  const [activeTab, setActiveTab] = useState("train");

  return (
    <div style={{ fontFamily: "'Plus Jakarta Sans',sans-serif", background: P.bg, color: P.ink, minHeight: "100vh" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,800&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
        @keyframes blink { 0%,50% { opacity:1; } 50.01%,100% { opacity:0; } }
        * { box-sizing: border-box; }
        select, input, button { font-family: 'Plus Jakarta Sans',sans-serif; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: ${P.bg}; }
        ::-webkit-scrollbar-thumb { background: ${P.border}; border-radius: 3px; }
      `}</style>

      {/* Header */}
      <div style={{ borderBottom: `1px solid ${P.border}`, background: P.card, position: "sticky", top: 0, zIndex: 100 }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <div style={{ padding: "16px 0" }}>
            <span style={{ fontFamily: "'Fraunces',serif", fontSize: 20, fontWeight: 800, color: P.ink }}>RL MLOps</span>
            <span style={{ fontSize: 12, color: P.inkSoft, marginLeft: 12 }}>Training Console</span>
          </div>
          <nav style={{ display: "flex", gap: 4 }}>
            {TABS.map(t => (
              <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
                padding: "8px 18px", borderRadius: 10, border: "none",
                background: activeTab === t.id ? P.accentSoft : "transparent",
                color: activeTab === t.id ? P.accent : P.inkSoft,
                fontWeight: activeTab === t.id ? 700 : 500,
                fontSize: 13.5, cursor: "pointer",
                borderBottom: activeTab === t.id ? `2px solid ${P.accent}` : "2px solid transparent",
                transition: "all 0.15s ease",
              }}>
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </div>

      {/* Content */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "28px 24px 80px" }}>
        {activeTab === "train" && <TrainTab />}
        {activeTab === "history" && <HistoryTab />}
        {activeTab === "mlflow" && <MLflowTab />}
      </div>
    </div>
  );
}

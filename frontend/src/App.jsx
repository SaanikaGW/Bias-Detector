import { useState, useEffect, useRef } from "react";

const API = import.meta?.env?.VITE_API_BASE_URL || "http://localhost:5001";

// ── Design tokens ────────────────────────────────────────────────────────────
const C = {
  ink:      "#0B1120",
  slate:    "#1E293B",
  mist:     "#475569",
  silver:   "#94A3B8",
  ghost:    "#E2E8F0",
  snow:     "#F8FAFC",
  teal:     "#0EA5E9",
  tealDark: "#0284C7",
  emerald:  "#10B981",
  amber:    "#F59E0B",
  rose:     "#F43F5E",
  card:     "rgba(255,255,255,0.85)",
};

const fontLink = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;1,9..144,400&family=DM+Sans:wght@300;400;500;600&display=swap');
`;

const globalStyles = `
  ${fontLink}
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: ${C.snow};
    color: ${C.ink};
    min-height: 100vh;
  }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: ${C.ghost}; }
  ::-webkit-scrollbar-thumb { background: ${C.teal}; border-radius: 3px; }

  @keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(14,165,233,0.4); }
    70%  { box-shadow: 0 0 0 12px rgba(14,165,233,0); }
    100% { box-shadow: 0 0 0 0 rgba(14,165,233,0); }
  }
  @keyframes shimmer {
    0%   { background-position: -400px 0; }
    100% { background-position: 400px 0; }
  }
  .fade-up { animation: fadeUp 0.5s ease both; }
  .fade-up-2 { animation: fadeUp 0.5s 0.1s ease both; }
  .fade-up-3 { animation: fadeUp 0.5s 0.2s ease both; }

  textarea, input, select {
    font-family: 'DM Sans', sans-serif;
  }
  button { cursor: pointer; }
  @keyframes spin { to { transform: rotate(360deg); } }
`;

// ── Reusable components ───────────────────────────────────────────────────────

function Pill({ color = C.teal, children }) {
  return (
    <span style={{
      display: "inline-block",
      padding: "2px 10px",
      borderRadius: 99,
      fontSize: 11,
      fontWeight: 600,
      letterSpacing: "0.05em",
      textTransform: "uppercase",
      background: color + "22",
      color,
      border: `1px solid ${color}44`,
    }}>
      {children}
    </span>
  );
}

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: C.card,
      border: `1px solid ${C.ghost}`,
      borderRadius: 16,
      padding: "28px 32px",
      backdropFilter: "blur(8px)",
      boxShadow: "0 2px 16px rgba(11,17,32,0.06)",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Btn({ children, onClick, variant = "primary", disabled, style = {} }) {
  const base = {
    display: "inline-flex",
    alignItems: "center",
    gap: 8,
    padding: "10px 22px",
    borderRadius: 10,
    fontFamily: "'DM Sans', sans-serif",
    fontWeight: 600,
    fontSize: 14,
    border: "none",
    transition: "all 0.18s",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.55 : 1,
  };
  const variants = {
    primary: {
      background: C.teal,
      color: "#fff",
      boxShadow: "0 2px 12px rgba(14,165,233,0.28)",
    },
    outline: {
      background: "transparent",
      color: C.teal,
      border: `1.5px solid ${C.teal}`,
    },
    ghost: {
      background: "transparent",
      color: C.mist,
      border: `1px solid ${C.ghost}`,
    },
    danger: {
      background: C.rose,
      color: "#fff",
    },
  };
  return (
    <button style={{ ...base, ...variants[variant], ...style }} onClick={onClick} disabled={disabled}>
      {children}
    </button>
  );
}

function ScoreMeter({ score, label }) {
  const pct   = Math.round(score * 100);
  const color = pct < 25 ? C.emerald : pct < 60 ? C.amber : C.rose;
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ position: "relative", width: 120, height: 120, margin: "0 auto 12px" }}>
        <svg width="120" height="120" style={{ transform: "rotate(-90deg)" }}>
          <circle cx="60" cy="60" r="50" fill="none" stroke={C.ghost} strokeWidth="10" />
          <circle
            cx="60" cy="60" r="50"
            fill="none"
            stroke={color}
            strokeWidth="10"
            strokeDasharray={`${2 * Math.PI * 50}`}
            strokeDashoffset={`${2 * Math.PI * 50 * (1 - pct / 100)}`}
            strokeLinecap="round"
            style={{ transition: "stroke-dashoffset 0.8s ease" }}
          />
        </svg>
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontFamily: "'Fraunces', serif", fontSize: 28, fontWeight: 700, color }}>{pct}</span>
          <span style={{ fontSize: 11, color: C.silver }}>/ 100</span>
        </div>
      </div>
      <span style={{ fontSize: 13, color: C.mist, fontWeight: 500 }}>{label}</span>
    </div>
  );
}

function Spinner() {
  return (
    <div style={{
      width: 20, height: 20,
      border: `2px solid ${C.ghost}`,
      borderTopColor: C.teal,
      borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
    }} />
  );
}

function HighlightChip({ span, type }) {
  const colors = {
    explicit_gender:   C.rose,
    stereotype:        C.amber,
    requirements_bias: "#8B5CF6",
    age_bias:          "#F97316",
  };
  return (
    <span style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 4,
      padding: "3px 10px",
      borderRadius: 6,
      fontSize: 13,
      background: (colors[type] || C.teal) + "18",
      border: `1px solid ${(colors[type] || C.teal)}44`,
      color: colors[type] || C.teal,
      fontWeight: 500,
    }}>
      ⚑ {span}
    </span>
  );
}

function EmptyState({ icon, title, body }) {
  return (
    <div style={{ textAlign: "center", padding: "48px 24px", color: C.silver }}>
      <div style={{ fontSize: 40, marginBottom: 12 }}>{icon}</div>
      <div style={{ fontFamily: "'Fraunces', serif", fontSize: 18, color: C.mist, marginBottom: 6 }}>{title}</div>
      <div style={{ fontSize: 14 }}>{body}</div>
    </div>
  );
}

// ── Nav ───────────────────────────────────────────────────────────────────────

function Nav({ page, setPage }) {
  const items = [
    { id: "home",     label: "Home" },
    { id: "reducer",  label: "JD Bias Reducer" },
    { id: "hiring",   label: "Hiring AI" },
    { id: "fairindex",label: "Fair Index" },
    { id: "about",    label: "About" },
    { id: "contact",  label: "Contact" },
  ];
  return (
    <nav style={{
      position: "sticky", top: 0, zIndex: 100,
      background: "rgba(248,250,252,0.92)",
      backdropFilter: "blur(12px)",
      borderBottom: `1px solid ${C.ghost}`,
      display: "flex", alignItems: "center",
      padding: "0 40px",
      height: 58,
      gap: 4,
    }}>
      <div
        style={{ cursor: "pointer", marginRight: "auto", display: "flex", alignItems: "center", gap: 8 }}
        onClick={() => setPage("home")}
      >
        <div style={{
          width: 30, height: 30,
          borderRadius: 8,
          background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 14,
        }}>⚖️</div>
        <span style={{
          fontFamily: "'Fraunces', serif",
          fontWeight: 700,
          fontSize: 17,
          color: C.ink,
          letterSpacing: "-0.01em",
        }}>BIOS Check</span>
      </div>
      {items.map(it => (
        <button
          key={it.id}
          onClick={() => setPage(it.id)}
          style={{
            background: page === it.id ? `${C.teal}14` : "transparent",
            color: page === it.id ? C.teal : C.mist,
            border: "none",
            borderRadius: 8,
            padding: "6px 14px",
            fontSize: 13,
            fontWeight: page === it.id ? 600 : 400,
            transition: "all 0.15s",
            fontFamily: "'DM Sans', sans-serif",
          }}
        >
          {it.label}
        </button>
      ))}
    </nav>
  );
}

// ── Home page ─────────────────────────────────────────────────────────────────

function HomePage({ setPage }) {
  const features = [
    { icon: "🔍", title: "JD Bias Reducer", desc: "Paste any job description and instantly detect explicit, stereotype, age, and requirements bias — with inclusive rewrites.", page: "reducer" },
    { icon: "🤖", title: "Bias-Aware Hiring AI", desc: "Evaluate candidate-job fit against PII-stripped resumes and bias-removed JDs. See how traditional AI compares.", page: "hiring" },
    { icon: "📊", title: "Fair Hiring Index", desc: "A quantitative 0–100 score measuring the fairness of a company's hiring language over time.", page: "fairindex" },
  ];

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 32px" }}>
      {/* Hero */}
      <div className="fade-up" style={{
        textAlign: "center",
        padding: "88px 40px 64px",
        position: "relative",
      }}>
        <div style={{
          position: "absolute", inset: 0,
          background: `radial-gradient(ellipse 60% 40% at 50% 30%, ${C.teal}10, transparent)`,
          pointerEvents: "none",
        }} />
        <Pill color={C.emerald}>Open Research Project</Pill>
        <h1 style={{
          fontFamily: "'Fraunces', serif",
          fontSize: "clamp(38px, 5vw, 60px)",
          fontWeight: 700,
          lineHeight: 1.12,
          marginTop: 20,
          marginBottom: 20,
          letterSpacing: "-0.02em",
          color: C.ink,
        }}>
          Fair hiring starts<br />
          <span style={{
            background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>with fair language.</span>
        </h1>
        <p style={{
          fontSize: 18,
          color: C.mist,
          maxWidth: 580,
          margin: "0 auto 40px",
          lineHeight: 1.65,
          fontWeight: 300,
        }}>
          AI hiring tools inherit gender bias from their training data. BIOS Check makes that
          bias visible, measurable, and fixable — at every step of the process.
        </p>
        <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
          <Btn onClick={() => setPage("reducer")} style={{ padding: "13px 32px", fontSize: 15, animation: "pulse-ring 2.5s infinite" }}>
            ✦ Analyze a Job Description
          </Btn>
          <Btn variant="outline" onClick={() => setPage("about")} style={{ padding: "13px 32px", fontSize: 15 }}>
            Learn More
          </Btn>
        </div>
      </div>

      {/* Feature cards */}
      <div className="fade-up-2" style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
        gap: 20,
        paddingBottom: 80,
      }}>
        {features.map(f => (
          <Card key={f.page} style={{
            cursor: "pointer",
            transition: "transform 0.18s, box-shadow 0.18s",
          }}
            onClick={() => setPage(f.page)}
          >
            <div style={{ fontSize: 30, marginBottom: 12 }}>{f.icon}</div>
            <div style={{
              fontFamily: "'Fraunces', serif",
              fontSize: 20,
              fontWeight: 600,
              color: C.ink,
              marginBottom: 8,
            }}>{f.title}</div>
            <p style={{ fontSize: 14, color: C.mist, lineHeight: 1.6 }}>{f.desc}</p>
            <div style={{ marginTop: 16 }}>
              <span style={{ color: C.teal, fontSize: 13, fontWeight: 600 }}>Open tool →</span>
            </div>
          </Card>
        ))}
      </div>

      {/* Research callout */}
      <Card className="fade-up-3" style={{
        background: `linear-gradient(135deg, ${C.ink}, ${C.slate})`,
        border: "none",
        color: "#fff",
        marginBottom: 80,
        display: "flex",
        alignItems: "center",
        gap: 32,
        flexWrap: "wrap",
      }}>
        <div style={{ flex: 1, minWidth: 240 }}>
          <Pill color={C.teal}>Research Insight</Pill>
          <h2 style={{
            fontFamily: "'Fraunces', serif",
            fontSize: 26,
            fontWeight: 600,
            marginTop: 12,
            marginBottom: 8,
            color: "#fff",
          }}>Why does this matter?</h2>
          <p style={{ fontSize: 14, color: "#94A3B8", lineHeight: 1.7 }}>
            Studies show that gendered language in job postings reduces the applicant pool by up to 42% 
            for underrepresented groups. Embedding-based AI screeners amplify this further — a bias 
            present in training data becomes a structural disadvantage at scale.
          </p>
        </div>
        <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
          {[["42%", "fewer applicants from biased JDs"], ["87%", "of hiring AIs trained on biased data"], ["3×", "more likely to screen out qualified candidates"]].map(([stat, desc]) => (
            <div key={stat} style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 700, color: C.teal }}>{stat}</div>
              <div style={{ fontSize: 12, color: "#64748B", maxWidth: 100 }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

// ── JD Bias Reducer ───────────────────────────────────────────────────────────

function ReducerPage() {
  const [text, setText]         = useState("");
  const [result, setResult]     = useState(null);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState("");
  const [tab, setTab]           = useState("suggestions");
  const MAX = 3000;

  async function handleAnalyze() {
    if (!text.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res  = await fetch(`${API}/api/bias-reducer/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Analysis failed");
      setResult(data);
      setTab("suggestions");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const levelColor = { low: C.emerald, medium: C.amber, high: C.rose };
  const levelIcon  = { low: "✓", medium: "⚠", high: "✕" };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 32 }}>
        <Pill>Phase 1</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 34, fontWeight: 700, marginTop: 10, marginBottom: 6 }}>
          JD Bias Reducer
        </h1>
        <p style={{ color: C.mist, fontSize: 15 }}>
          Paste a job description. We'll detect bias, explain it, and rewrite it — inclusively.
        </p>
        <p style={{ fontSize: 12, color: C.silver, marginTop: 6 }}>
          ⚠ Do not input personal or identifying information.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Left — Input */}
        <div className="fade-up">
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <label style={{ fontWeight: 600, fontSize: 14, color: C.ink }}>Job Description</label>
              <span style={{ fontSize: 12, color: text.length > MAX * 0.9 ? C.rose : C.silver }}>
                {text.length} / {MAX}
              </span>
            </div>
            <textarea
              value={text}
              onChange={e => setText(e.target.value.slice(0, MAX))}
              placeholder="Paste job description here…

Example:
We are looking for a young, energetic salesman to join our team. The ideal candidate should be available 24/7 and have a dominant personality..."
              style={{
                width: "100%",
                minHeight: 320,
                border: `1.5px solid ${C.ghost}`,
                borderRadius: 10,
                padding: "14px 16px",
                fontSize: 14,
                lineHeight: 1.65,
                color: C.ink,
                resize: "vertical",
                outline: "none",
                background: C.snow,
              }}
            />
            {error && (
              <div style={{ marginTop: 10, padding: "8px 12px", background: `${C.rose}12`, border: `1px solid ${C.rose}33`, borderRadius: 8, color: C.rose, fontSize: 13 }}>
                {error}
              </div>
            )}
            <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
              <Btn onClick={handleAnalyze} disabled={loading || !text.trim()}>
                {loading ? <><Spinner /> Analyzing…</> : "✦ Analyze"}
              </Btn>
              <Btn variant="ghost" onClick={() => { setText(""); setResult(null); setError(""); }}>
                Clear
              </Btn>
            </div>
          </Card>
        </div>

        {/* Right — Results */}
        <div className="fade-up-2">
          {!result && !loading && (
            <Card style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyState icon="🔍" title="Analysis will appear here" body="Paste a job description on the left and click Analyze." />
            </Card>
          )}
          {loading && (
            <Card style={{ height: "100%", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16 }}>
              <div style={{ width: 40, height: 40, borderRadius: "50%", border: `3px solid ${C.ghost}`, borderTopColor: C.teal, animation: "spin 0.8s linear infinite" }} />
              <p style={{ color: C.mist, fontSize: 14 }}>Detecting bias patterns…</p>
            </Card>
          )}
          {result && (
            <Card>
              {/* Score summary */}
              <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 20, padding: "16px", background: C.snow, borderRadius: 10 }}>
                <ScoreMeter score={result.bias_score} label="Bias Score" />
                <div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <span style={{
                      fontSize: 13,
                      fontWeight: 700,
                      color: levelColor[result.bias_level] || C.mist,
                      background: (levelColor[result.bias_level] || C.mist) + "18",
                      padding: "3px 10px",
                      borderRadius: 6,
                    }}>
                      {levelIcon[result.bias_level]} {result.bias_level?.toUpperCase()} BIAS
                    </span>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {(result.categories || []).map(cat => (
                      <Pill key={cat} color={
                        cat === "explicit_gender" ? C.rose :
                        cat === "stereotype"       ? C.amber :
                        cat === "age_bias"         ? "#F97316" : "#8B5CF6"
                      }>{cat.replace("_", " ")}</Pill>
                    ))}
                    {result.categories?.length === 0 && <Pill color={C.emerald}>No bias detected</Pill>}
                  </div>
                </div>
              </div>

              {/* Flagged spans */}
              {result.highlights?.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.07em", textTransform: "uppercase", marginBottom: 8 }}>Flagged Phrases</div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                    {result.highlights.map((h, i) => <HighlightChip key={i} span={h.span} type={h.type} />)}
                  </div>
                </div>
              )}

              {/* Tabs */}
              <div style={{ display: "flex", gap: 2, borderBottom: `1px solid ${C.ghost}`, marginBottom: 16 }}>
                {["suggestions", "rewrite"].map(t => (
                  <button key={t} onClick={() => setTab(t)} style={{
                    padding: "7px 16px",
                    fontSize: 13,
                    fontWeight: tab === t ? 600 : 400,
                    color: tab === t ? C.teal : C.mist,
                    background: "transparent",
                    border: "none",
                    borderBottom: tab === t ? `2px solid ${C.teal}` : "2px solid transparent",
                    marginBottom: -1,
                    transition: "all 0.15s",
                    fontFamily: "'DM Sans', sans-serif",
                  }}>
                    {t === "suggestions" ? "💡 Suggestions" : "✏️ Rewritten JD"}
                  </button>
                ))}
              </div>

              <div style={{ maxHeight: 280, overflowY: "auto", fontSize: 14, lineHeight: 1.7, color: C.slate }}>
                {tab === "suggestions" && (
                  result.suggestions?.length > 0
                    ? <ul style={{ paddingLeft: 0, listStyle: "none" }}>
                        {result.suggestions.map((s, i) => (
                          <li key={i} style={{ padding: "7px 0", borderBottom: `1px solid ${C.ghost}`, display: "flex", gap: 8 }}>
                            <span style={{ color: C.teal, fontWeight: 700, minWidth: 18 }}>→</span>
                            <span>{s}</span>
                          </li>
                        ))}
                      </ul>
                    : <EmptyState icon="✓" title="No suggestions needed" body="This job description appears inclusive." />
                )}
                {tab === "rewrite" && (
                  result.rewritten_jd
                    ? <div style={{
                        background: C.snow,
                        borderRadius: 8,
                        padding: "14px 16px",
                        whiteSpace: "pre-wrap",
                        fontFamily: "'DM Sans', sans-serif",
                        fontSize: 13.5,
                      }}>{result.rewritten_jd}</div>
                    : <EmptyState icon="✏️" title="Rewrite pending" body="Rewritten JD will appear here." />
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Hiring AI page ────────────────────────────────────────────────────────────

function HiringAIPage() {
  const [step, setStep]           = useState(1);
  const [jd, setJd]               = useState("");
  const [resume, setResume]       = useState("");
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState("");

  async function handleEvaluate() {
    if (!jd.trim() || !resume.trim()) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res  = await fetch(`${API}/api/hiring-ai/compare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ original_jd: jd, rewritten_jd: jd, original_resume: resume }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Evaluation failed");
      setResult(data);
      setStep(3);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const matchColor = { full: C.emerald, partial: C.amber, none: C.rose };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "40px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 32 }}>
        <Pill color="#8B5CF6">Phase 3</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 34, fontWeight: 700, marginTop: 10, marginBottom: 6 }}>
          Bias-Aware Hiring AI
        </h1>
        <p style={{ color: C.mist, fontSize: 15, maxWidth: 600 }}>
          Evaluate candidate fit using PII-stripped resumes and bias-reduced JDs.
          See how traditional AI compares.
        </p>
        <div style={{ marginTop: 8, padding: "8px 14px", background: `${C.teal}10`, border: `1px solid ${C.teal}30`, borderRadius: 8, fontSize: 13, color: C.teal, display: "inline-flex", gap: 6, alignItems: "center" }}>
          🔒 Resumes are never stored. PII is stripped before any evaluation.
        </div>
      </div>

      {/* Stepper */}
      <div style={{ display: "flex", gap: 0, marginBottom: 32, background: C.ghost, borderRadius: 12, padding: 4 }}>
        {[1, 2, 3].map(s => (
          <button key={s} onClick={() => s < step && setStep(s)} style={{
            flex: 1,
            padding: "10px 16px",
            borderRadius: 10,
            border: "none",
            background: step === s ? "#fff" : "transparent",
            color: step === s ? C.teal : step > s ? C.emerald : C.silver,
            fontWeight: step === s ? 600 : 400,
            fontSize: 13,
            fontFamily: "'DM Sans', sans-serif",
            boxShadow: step === s ? "0 1px 8px rgba(0,0,0,0.08)" : "none",
            cursor: s < step ? "pointer" : "default",
            transition: "all 0.15s",
          }}>
            {step > s ? "✓ " : `${s}. `}
            {s === 1 ? "Job Description" : s === 2 ? "Resume" : "Evaluation Results"}
          </button>
        ))}
      </div>

      {step === 1 && (
        <Card className="fade-up">
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 20, marginBottom: 6 }}>Step 1 — Paste Job Description</h2>
          <p style={{ fontSize: 13, color: C.mist, marginBottom: 16 }}>Use a bias-reduced JD from the Bias Reducer for best results.</p>
          <textarea
            value={jd}
            onChange={e => setJd(e.target.value)}
            placeholder="Paste the (rewritten, bias-reduced) job description here…"
            style={{
              width: "100%", minHeight: 260,
              border: `1.5px solid ${C.ghost}`, borderRadius: 10,
              padding: "14px 16px", fontSize: 14, lineHeight: 1.65,
              color: C.ink, resize: "vertical", outline: "none", background: C.snow,
            }}
          />
          <div style={{ marginTop: 14 }}>
            <Btn onClick={() => setStep(2)} disabled={!jd.trim()}>Next: Add Resume →</Btn>
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card className="fade-up">
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 20, marginBottom: 6 }}>Step 2 — Paste Resume</h2>
          <p style={{ fontSize: 13, color: C.mist, marginBottom: 16 }}>
            Paste the candidate's resume text. PII will be automatically stripped before evaluation.
          </p>
          <textarea
            value={resume}
            onChange={e => setResume(e.target.value)}
            placeholder="Paste resume text here…"
            style={{
              width: "100%", minHeight: 320,
              border: `1.5px solid ${C.ghost}`, borderRadius: 10,
              padding: "14px 16px", fontSize: 14, lineHeight: 1.65,
              color: C.ink, resize: "vertical", outline: "none", background: C.snow,
            }}
          />
          {error && (
            <div style={{ marginTop: 10, padding: "8px 12px", background: `${C.rose}12`, borderRadius: 8, color: C.rose, fontSize: 13 }}>
              {error}
            </div>
          )}
          <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
            <Btn variant="ghost" onClick={() => setStep(1)}>← Back</Btn>
            <Btn onClick={handleEvaluate} disabled={loading || !resume.trim()}>
              {loading ? <><Spinner /> Evaluating…</> : "✦ Evaluate Fit"}
            </Btn>
          </div>
        </Card>
      )}

      {step === 3 && result && (
        <div className="fade-up">
          {/* Comparison panel */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginBottom: 24 }}>
            {[
              { label: "Traditional AI", color: C.rose, data: result.traditional, icon: "📉" },
              { label: "Bias-Aware AI (BIOS Check)", color: C.emerald, data: result.bias_aware, icon: "✦" },
            ].map(({ label, color, data, icon }) => (
              <Card key={label} style={{ borderTop: `3px solid ${color}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                  <span style={{ fontWeight: 600, fontSize: 14, color: C.ink }}>{icon} {label}</span>
                  <span style={{
                    fontFamily: "'Fraunces', serif",
                    fontSize: 26,
                    fontWeight: 700,
                    color,
                  }}>{Math.round((data?.fit_score || 0) * 100)}%</span>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <Pill color={
                    data?.fit_level === "strong" ? C.emerald :
                    data?.fit_level === "moderate" ? C.amber : C.rose
                  }>{data?.fit_level || "—"} fit</Pill>
                </div>
                <p style={{ fontSize: 13, color: C.mist, lineHeight: 1.6 }}>{data?.explanation || "—"}</p>
              </Card>
            ))}
          </div>

          {/* Delta callout */}
          <Card style={{ background: `linear-gradient(135deg, ${C.ink}, ${C.slate})`, border: "none", marginBottom: 24, display: "flex", alignItems: "center", gap: 24 }}>
            <div style={{ fontFamily: "'Fraunces', serif", fontSize: 48, fontWeight: 700, color: C.teal }}>
              {result.score_delta}
            </div>
            <div>
              <div style={{ color: "#fff", fontWeight: 600, marginBottom: 4 }}>Score Delta</div>
              <div style={{ color: "#64748B", fontSize: 14 }}>
                {parseFloat(result.score_delta) >= 0
                  ? "Our bias-aware system ranked this candidate higher than traditional AI would have."
                  : "Traditional AI may have over-fitted on biased signals for this candidate."}
              </div>
            </div>
          </Card>

          {/* Skill matches */}
          {result.bias_aware?.skill_matches?.length > 0 && (
            <Card style={{ marginBottom: 24 }}>
              <h3 style={{ fontFamily: "'Fraunces', serif", fontSize: 18, marginBottom: 16 }}>Skill Match Breakdown</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {result.bias_aware.skill_matches.map((m, i) => (
                  <div key={i} style={{
                    display: "flex", gap: 12, alignItems: "flex-start",
                    padding: "10px 14px",
                    background: C.snow,
                    borderRadius: 8,
                    borderLeft: `3px solid ${matchColor[m.match] || C.silver}`,
                  }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, fontSize: 13, color: C.ink, marginBottom: 3 }}>{m.requirement}</div>
                      <div style={{ fontSize: 12, color: C.mist }}>{m.evidence}</div>
                    </div>
                    <Pill color={matchColor[m.match] || C.silver}>{m.match}</Pill>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* PII suppressed */}
          {result.bias_aware?.bias_signals_suppressed?.length > 0 && (
            <Card>
              <h3 style={{ fontFamily: "'Fraunces', serif", fontSize: 16, marginBottom: 10 }}>🔒 Bias Signals Suppressed</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {result.bias_aware.bias_signals_suppressed.map((s, i) => (
                  <span key={i} style={{
                    padding: "3px 10px",
                    borderRadius: 6,
                    background: `${C.teal}12`,
                    color: C.teal,
                    fontSize: 12,
                    fontWeight: 500,
                  }}>{s}</span>
                ))}
              </div>
            </Card>
          )}

          <div style={{ marginTop: 20 }}>
            <Btn variant="outline" onClick={() => { setStep(1); setResult(null); setJd(""); setResume(""); }}>
              ← Start New Evaluation
            </Btn>
          </div>
        </div>
      )}
    </div>
  );
}

// ── Fair Index page ───────────────────────────────────────────────────────────

function FairIndexPage() {
  const [jds, setJds]   = useState([{ text: "", id: Date.now() }]);
  const [scores, setScores] = useState(null);
  const [loading, setLoading] = useState(false);

  const WEIGHTS = { explicit_gender: 1.0, stereotype: 0.8, age_bias: 0.7, requirements_bias: 0.6 };

  function addJd() {
    setJds(prev => [...prev, { text: "", id: Date.now() }]);
  }

  async function calculateFHI() {
    const texts = jds.map(j => j.text.trim()).filter(Boolean);
    if (!texts.length) return;
    setLoading(true);
    try {
      const results = await Promise.all(texts.map(text =>
        fetch(`${API}/api/bias-reducer/analyze`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text }),
        }).then(r => r.json())
      ));
      const N = results.length;
      const avg = results.reduce((acc, r) => {
        const w = r.categories?.reduce((s, c) => s + (WEIGHTS[c] || 0.5), 0) || 0;
        return acc + (r.bias_score * (w || 1));
      }, 0) / N;
      const fhi = Math.round(100 * (1 - Math.min(1, avg)));
      setScores({ fhi, results, avg: round2(avg) });
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  function round2(n) { return Math.round(n * 100) / 100; }

  const fhiColor = scores
    ? scores.fhi >= 75 ? C.emerald : scores.fhi >= 50 ? C.amber : C.rose
    : C.silver;

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "40px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 32 }}>
        <Pill color={C.amber}>Phase 2</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 34, fontWeight: 700, marginTop: 10, marginBottom: 6 }}>
          Fair Hiring Index
        </h1>
        <p style={{ color: C.mist, fontSize: 15, maxWidth: 560 }}>
          A 0–100 score measuring the fairness of your hiring language across multiple job descriptions.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Input */}
        <Card className="fade-up">
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 18, marginBottom: 16 }}>
            Batch Job Descriptions
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {jds.map((jd, i) => (
              <div key={jd.id}>
                <label style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.06em", textTransform: "uppercase" }}>
                  JD #{i + 1}
                </label>
                <textarea
                  value={jd.text}
                  onChange={e => setJds(prev => prev.map(j => j.id === jd.id ? { ...j, text: e.target.value } : j))}
                  placeholder="Paste job description…"
                  style={{
                    width: "100%", minHeight: 100, marginTop: 4,
                    border: `1.5px solid ${C.ghost}`, borderRadius: 8,
                    padding: "10px 12px", fontSize: 13, lineHeight: 1.6,
                    color: C.ink, resize: "vertical", outline: "none", background: C.snow,
                  }}
                />
              </div>
            ))}
          </div>
          <div style={{ marginTop: 14, display: "flex", gap: 10 }}>
            <Btn variant="ghost" onClick={addJd}>+ Add JD</Btn>
            <Btn onClick={calculateFHI} disabled={loading || !jds.some(j => j.text.trim())}>
              {loading ? <><Spinner /> Calculating…</> : "Calculate FHI"}
            </Btn>
          </div>

          {/* Formula card */}
          <div style={{ marginTop: 20, padding: "14px", background: C.snow, borderRadius: 8 }}>
            <div style={{ fontSize: 12, fontWeight: 600, color: C.silver, marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.06em" }}>FHI Formula</div>
            <div style={{ fontFamily: "monospace", fontSize: 13, color: C.slate, lineHeight: 1.8 }}>
              FHI = 100 × (1 − (1/N) × Σ(Bᵢ × Cᵢ))<br />
              <span style={{ color: C.silver, fontSize: 11 }}>Bᵢ = bias score · Cᵢ = category weight</span>
            </div>
            <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Object.entries(WEIGHTS).map(([k, v]) => (
                <span key={k} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: C.ghost, color: C.mist }}>
                  {k.replace("_", " ")}: {v}×
                </span>
              ))}
            </div>
          </div>
        </Card>

        {/* Results */}
        <div className="fade-up-2">
          {!scores && (
            <Card style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center" }}>
              <EmptyState icon="📊" title="FHI will appear here" body="Add job descriptions and click Calculate FHI." />
            </Card>
          )}
          {scores && (
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <Card style={{ textAlign: "center", borderTop: `3px solid ${fhiColor}` }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.08em", textTransform: "uppercase", marginBottom: 10 }}>Fair Hiring Index</div>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 72, fontWeight: 700, color: fhiColor, lineHeight: 1 }}>
                  {scores.fhi}
                </div>
                <div style={{ fontSize: 14, color: C.mist, marginTop: 6 }}>out of 100</div>
                <div style={{ marginTop: 10 }}>
                  <Pill color={fhiColor}>
                    {scores.fhi >= 75 ? "High Fairness" : scores.fhi >= 50 ? "Moderate Fairness" : "Low Fairness"}
                  </Pill>
                </div>
              </Card>

              {scores.results.map((r, i) => (
                <Card key={i} style={{ borderLeft: `3px solid ${r.bias_level === "low" ? C.emerald : r.bias_level === "medium" ? C.amber : C.rose}` }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontWeight: 600, fontSize: 13, color: C.ink }}>JD #{i + 1}</span>
                    <span style={{ fontFamily: "'Fraunces', serif", fontWeight: 700, color: r.bias_score > 0.5 ? C.rose : r.bias_score > 0.25 ? C.amber : C.emerald }}>
                      {Math.round(r.bias_score * 100)}% bias
                    </span>
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                    {r.categories?.length ? r.categories.map(c => <Pill key={c} color={C.silver}>{c.replace("_", " ")}</Pill>) : <Pill color={C.emerald}>clean</Pill>}
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── About ─────────────────────────────────────────────────────────────────────

function AboutPage() {
  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: "60px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 48 }}>
        <Pill color={C.teal}>About the Project</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 700, marginTop: 16, marginBottom: 20, letterSpacing: "-0.02em" }}>
          Measuring what matters.
        </h1>
        <p style={{ fontSize: 16, color: C.mist, lineHeight: 1.8, marginBottom: 16 }}>
          BIOS Check Careers is a research project investigating how gender bias in language
          propagates through AI hiring systems — and how to stop it.
        </p>
        <p style={{ fontSize: 15, color: C.mist, lineHeight: 1.8 }}>
          Most bias-detection tools work retroactively. This project is designed differently:
          by stripping demographic signals before evaluation and using bias-reduced job
          descriptions as rubrics, the hiring AI is structurally incapable of acting on
          bias signals — not just instructed to ignore them.
        </p>
      </div>

      <div style={{ display: "grid", gap: 20, marginBottom: 48 }}>
        {[
          { title: "Phase 1 — Bias Detector", status: "Complete", desc: "Rule-based + semantic detection of explicit, stereotype, age, and requirements bias in job descriptions." },
          { title: "Phase 2 — Fair Hiring Index", status: "In Progress", desc: "A weighted, quantitative fairness score across a company's full JD history." },
          { title: "Phase 3 — Bias-Aware Hiring AI", status: "In Progress", desc: "PII stripping + constrained fit evaluation compared against traditional AI scoring." },
        ].map(p => (
          <Card key={p.title} style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
            <div>
              <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 6 }}>
                <span style={{ fontFamily: "'Fraunces', serif", fontSize: 17, fontWeight: 600 }}>{p.title}</span>
                <Pill color={p.status === "Complete" ? C.emerald : C.amber}>{p.status}</Pill>
              </div>
              <p style={{ fontSize: 14, color: C.mist, lineHeight: 1.6 }}>{p.desc}</p>
            </div>
          </Card>
        ))}
      </div>

      <Card style={{ background: `linear-gradient(135deg, ${C.ink}, ${C.slate})`, border: "none" }}>
        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
          <div style={{
            width: 56, height: 56, borderRadius: "50%",
            background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 22, flexShrink: 0,
          }}>👩‍💻</div>
          <div>
            <div style={{ color: "#fff", fontFamily: "'Fraunces', serif", fontSize: 18, fontWeight: 600, marginBottom: 4 }}>
              Saanika
            </div>
            <div style={{ color: "#64748B", fontSize: 13, lineHeight: 1.6 }}>
              Builder of BIOS Check Careers · Researching fairness in AI hiring systems ·
              Mentored by Jason
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ── Contact ───────────────────────────────────────────────────────────────────

function ContactPage() {
  const [form, setForm] = useState({ name: "", email: "", category: "General", message: "", consent: false });
  const [sent, setSent] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSubmit() {
    if (!form.name || !form.email || !form.message || !form.consent) return;
    setLoading(true); setError("");
    try {
      const res  = await fetch(`${API}/api/contact/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Submission failed");
      setSent(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const inputStyle = {
    width: "100%",
    border: `1.5px solid ${C.ghost}`,
    borderRadius: 10,
    padding: "11px 14px",
    fontSize: 14,
    color: C.ink,
    outline: "none",
    background: C.snow,
    fontFamily: "'DM Sans', sans-serif",
    transition: "border-color 0.15s",
  };

  return (
    <div style={{ maxWidth: 560, margin: "0 auto", padding: "60px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 36 }}>
        <Pill>Get in Touch</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 34, fontWeight: 700, marginTop: 12, marginBottom: 8 }}>
          Contact Us
        </h1>
        <p style={{ color: C.mist, fontSize: 15 }}>Feedback, research questions, or collaboration inquiries.</p>
      </div>

      {sent ? (
        <Card className="fade-up" style={{ textAlign: "center", padding: "48px 32px" }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>✓</div>
          <div style={{ fontFamily: "'Fraunces', serif", fontSize: 22, marginBottom: 8 }}>Message Sent</div>
          <p style={{ color: C.mist, fontSize: 14 }}>Thank you! We'll be in touch soon.</p>
        </Card>
      ) : (
        <Card className="fade-up">
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.06em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>Name</label>
                <input style={inputStyle} value={form.name} onChange={e => setForm(p => ({ ...p, name: e.target.value }))} placeholder="Your name" />
              </div>
              <div>
                <label style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.06em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>Email</label>
                <input type="email" style={inputStyle} value={form.email} onChange={e => setForm(p => ({ ...p, email: e.target.value }))} placeholder="you@example.com" />
              </div>
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.06em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>Category</label>
              <select style={inputStyle} value={form.category} onChange={e => setForm(p => ({ ...p, category: e.target.value }))}>
                {["General", "Research", "Bug Report", "Collaboration", "Other"].map(c => <option key={c}>{c}</option>)}
              </select>
            </div>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: C.silver, letterSpacing: "0.06em", textTransform: "uppercase", display: "block", marginBottom: 6 }}>Message</label>
              <textarea style={{ ...inputStyle, minHeight: 130, resize: "vertical" }} value={form.message} onChange={e => setForm(p => ({ ...p, message: e.target.value }))} placeholder="Your message…" />
            </div>
            <label style={{ display: "flex", gap: 10, alignItems: "flex-start", cursor: "pointer" }}>
              <input
                type="checkbox"
                checked={form.consent}
                onChange={e => setForm(p => ({ ...p, consent: e.target.checked }))}
                style={{ marginTop: 3, accentColor: C.teal }}
              />
              <span style={{ fontSize: 13, color: C.mist, lineHeight: 1.5 }}>
                I consent to having my contact information used to respond to this inquiry.
                I understand this data is not stored beyond the initial response.
              </span>
            </label>
            {error && <div style={{ padding: "8px 12px", background: `${C.rose}12`, borderRadius: 8, color: C.rose, fontSize: 13 }}>{error}</div>}
            <Btn
              onClick={handleSubmit}
              disabled={loading || !form.name || !form.email || !form.message || !form.consent}
            >
              {loading ? <><Spinner /> Sending…</> : "Send Message"}
            </Btn>
          </div>
        </Card>
      )}
    </div>
  );
}

// ── App root ──────────────────────────────────────────────────────────────────

export default function App() {
  const [page, setPage] = useState("home");

  const pages = {
    home:      <HomePage setPage={setPage} />,
    reducer:   <ReducerPage />,
    hiring:    <HiringAIPage />,
    fairindex: <FairIndexPage />,
    about:     <AboutPage />,
    contact:   <ContactPage />,
  };

  return (
    <>
      <style>{globalStyles}</style>
      <Nav page={page} setPage={setPage} />
      <main style={{ minHeight: "calc(100vh - 58px)" }}>
        {pages[page] || pages.home}
      </main>
      <footer style={{
        borderTop: `1px solid ${C.ghost}`,
        padding: "24px 40px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 12,
        fontSize: 13,
        color: C.silver,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span>⚖️</span>
          <span style={{ fontFamily: "'Fraunces', serif", color: C.mist }}>BIOS Check Careers</span>
          <span>— Making fair hiring measurable.</span>
        </div>
        <div style={{ display: "flex", gap: 16 }}>
          {["Home", "Bias Reducer", "Hiring AI", "Fair Index", "About"].map(l => (
            <button key={l} style={{ background: "none", border: "none", color: C.silver, fontSize: 13, cursor: "pointer", fontFamily: "'DM Sans', sans-serif" }}>
              {l}
            </button>
          ))}
        </div>
      </footer>
    </>
  );
}

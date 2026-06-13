import { useState, useEffect, useRef } from "react";

const API = import.meta?.env?.VITE_API_BASE_URL || "http://localhost:5001";

// ── Design tokens (dark theme) ────────────────────────────────────────────────
const C = {
  ink:      "#FFFFFF",
  slate:    "#CBD5E1",
  mist:     "#94A3B8",
  silver:   "#64748B",
  ghost:    "#1E293B",
  snow:     "#060D1F",
  surface:  "#0B1525",
  teal:     "#0EA5E9",
  tealDark: "#0284C7",
  emerald:  "#10B981",
  amber:    "#F59E0B",
  rose:     "#F43F5E",
  card:     "rgba(11,20,40,0.92)",
};

const fontLink = `
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;0,9..144,700;0,9..144,800;1,9..144,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');
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
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
  }
  @keyframes pulse-ring {
    0%   { box-shadow: 0 0 0 0 rgba(14,165,233,0.5); }
    70%  { box-shadow: 0 0 0 14px rgba(14,165,233,0); }
    100% { box-shadow: 0 0 0 0 rgba(14,165,233,0); }
  }
  @keyframes glow-pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
  }
  .fade-up   { animation: fadeUp 0.55s ease both; }
  .fade-up-2 { animation: fadeUp 0.55s 0.12s ease both; }
  .fade-up-3 { animation: fadeUp 0.55s 0.24s ease both; }

  textarea, input, select { font-family: 'DM Sans', sans-serif; }
  button { cursor: pointer; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .card-hover {
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
  }
  .card-hover:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 40px rgba(14,165,233,0.18);
    border-color: rgba(14,165,233,0.4) !important;
  }
`;

// ── Reusable components ───────────────────────────────────────────────────────

function Pill({ color = C.teal, children }) {
  return (
    <span style={{
      display: "inline-block",
      padding: "3px 12px",
      borderRadius: 99,
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: "0.07em",
      textTransform: "uppercase",
      background: color + "20",
      color,
      border: `1px solid ${color}50`,
    }}>
      {children}
    </span>
  );
}

function Card({ children, style = {}, className = "", onClick }) {
  return (
    <div className={className} onClick={onClick} style={{
      background: C.card,
      border: `1px solid ${C.ghost}`,
      borderRadius: 18,
      padding: "28px 32px",
      backdropFilter: "blur(12px)",
      boxShadow: "0 4px 24px rgba(0,0,0,0.4)",
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
    padding: "11px 24px",
    borderRadius: 10,
    fontFamily: "'DM Sans', sans-serif",
    fontWeight: 700,
    fontSize: 14,
    border: "none",
    transition: "all 0.18s",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
  };
  const variants = {
    primary: {
      background: `linear-gradient(135deg, ${C.teal}, ${C.tealDark})`,
      color: "#fff",
      boxShadow: "0 4px 20px rgba(14,165,233,0.4)",
    },
    outline: {
      background: "transparent",
      color: C.teal,
      border: `1.5px solid ${C.teal}`,
    },
    ghost: {
      background: "rgba(255,255,255,0.06)",
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
            style={{ transition: "stroke-dashoffset 0.8s ease", filter: `drop-shadow(0 0 6px ${color}80)` }}
          />
        </svg>
        <div style={{
          position: "absolute", inset: 0,
          display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
        }}>
          <span style={{ fontFamily: "'Fraunces', serif", fontSize: 28, fontWeight: 800, color }}>{pct}</span>
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
      <div style={{ fontSize: 14, color: C.silver }}>{body}</div>
    </div>
  );
}

// ── Section label ─────────────────────────────────────────────────────────────
function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 11,
      fontWeight: 700,
      letterSpacing: "0.1em",
      textTransform: "uppercase",
      color: C.silver,
      marginBottom: 8,
    }}>{children}</div>
  );
}

// ── Nav ───────────────────────────────────────────────────────────────────────

function Nav({ page, setPage }) {
  const items = [
    { id: "home",      label: "Home" },
    { id: "reducer",   label: "JD Bias Reducer" },
    { id: "hiring",    label: "Hiring AI" },
    { id: "fairindex", label: "Fair Index" },
    { id: "about",     label: "About" },
    { id: "contact",   label: "Contact" },
  ];
  return (
    <nav style={{
      position: "sticky", top: 0, zIndex: 100,
      background: "rgba(6,13,31,0.92)",
      backdropFilter: "blur(16px)",
      borderBottom: `1px solid ${C.ghost}`,
      display: "flex", alignItems: "center",
      padding: "0 40px",
      height: 60,
      gap: 2,
    }}>
      <div
        style={{ cursor: "pointer", marginRight: "auto", display: "flex", alignItems: "center", gap: 10 }}
        onClick={() => setPage("home")}
      >
        <div style={{
          width: 32, height: 32,
          borderRadius: 9,
          background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
          display: "flex", alignItems: "center", justifyContent: "center",
          fontSize: 15,
          boxShadow: `0 0 16px ${C.teal}50`,
        }}>⚖️</div>
        <span style={{
          fontFamily: "'Fraunces', serif",
          fontWeight: 700,
          fontSize: 17,
          color: "#fff",
          letterSpacing: "-0.01em",
        }}>BIOS Check</span>
      </div>
      {items.map(it => (
        <button
          key={it.id}
          onClick={() => setPage(it.id)}
          style={{
            background: page === it.id ? `${C.teal}18` : "transparent",
            color: page === it.id ? C.teal : C.mist,
            border: "none",
            borderRadius: 8,
            padding: "6px 14px",
            fontSize: 13,
            fontWeight: page === it.id ? 700 : 400,
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
    {
      icon: "🔍",
      title: "JD Bias Reducer",
      desc: "Paste any job description and get an instant bias score, flagged phrases, and an inclusively rewritten version — ready to post.",
      page: "reducer",
      accent: C.teal,
    },
    {
      icon: "🤖",
      title: "Bias-Aware Hiring AI",
      desc: "Compare how a bias-aware AI and a traditional AI rank the same candidate — see the gap that bias creates, quantified.",
      page: "hiring",
      accent: "#8B5CF6",
    },
    {
      icon: "📊",
      title: "Fair Hiring Index",
      desc: "Track fairness across your full library of job descriptions. One score that tells you where your language stands — and where to improve.",
      page: "fairindex",
      accent: C.emerald,
    },
  ];

  return (
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "0 32px" }}>
      {/* Hero */}
      <div className="fade-up" style={{
        textAlign: "center",
        padding: "100px 40px 72px",
        position: "relative",
      }}>
        {/* Background glow */}
        <div style={{
          position: "absolute", inset: 0,
          background: `radial-gradient(ellipse 70% 50% at 50% 0%, ${C.teal}18, transparent 70%)`,
          pointerEvents: "none",
        }} />
        <div style={{
          position: "absolute", inset: 0,
          background: `radial-gradient(ellipse 40% 30% at 50% 0%, ${C.emerald}10, transparent 60%)`,
          pointerEvents: "none",
        }} />

        <Pill color={C.emerald}>Open Research Project</Pill>
        <h1 style={{
          fontFamily: "'Fraunces', serif",
          fontSize: "clamp(42px, 6vw, 72px)",
          fontWeight: 800,
          lineHeight: 1.08,
          marginTop: 24,
          marginBottom: 24,
          letterSpacing: "-0.03em",
          color: "#fff",
        }}>
          Fair hiring starts<br />
          <span style={{
            background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}>with fair language.</span>
        </h1>
        <p style={{
          fontSize: 19,
          color: C.slate,
          maxWidth: 560,
          margin: "0 auto 44px",
          lineHeight: 1.7,
          fontWeight: 300,
        }}>
          AI hiring tools inherit gender bias from their training data. BIOS Check makes that
          bias <em>visible</em>, measurable, and fixable — at every step of the process.
        </p>
        <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap" }}>
          <Btn
            onClick={() => setPage("reducer")}
            style={{ padding: "14px 36px", fontSize: 15, animation: "pulse-ring 2.5s infinite" }}
          >
            ✦ Analyze a Job Description
          </Btn>
          <Btn variant="outline" onClick={() => setPage("about")} style={{ padding: "14px 36px", fontSize: 15 }}>
            Learn More
          </Btn>
        </div>
      </div>

      {/* Feature cards */}
      <div className="fade-up-2" style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
        gap: 20,
        marginBottom: 56,
      }}>
        {features.map(f => (
          <Card
            key={f.page}
            className="card-hover"
            style={{ cursor: "pointer", borderTop: `3px solid ${f.accent}` }}
            onClick={() => setPage(f.page)}
          >
            <div style={{
              width: 48, height: 48, borderRadius: 12,
              background: f.accent + "18",
              border: `1px solid ${f.accent}30`,
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 22, marginBottom: 16,
            }}>{f.icon}</div>
            <div style={{
              fontFamily: "'Fraunces', serif",
              fontSize: 21,
              fontWeight: 700,
              color: "#fff",
              marginBottom: 10,
            }}>{f.title}</div>
            <p style={{ fontSize: 14, color: C.slate, lineHeight: 1.7, marginBottom: 20 }}>{f.desc}</p>
            <span style={{ color: f.accent, fontSize: 13, fontWeight: 700, letterSpacing: "0.01em" }}>
              Open tool →
            </span>
          </Card>
        ))}
      </div>

      {/* How to navigate */}
      <div className="fade-up-3" style={{
        marginBottom: 100,
        borderRadius: 24,
        border: `1px solid ${C.ghost}`,
        background: "linear-gradient(135deg, rgba(14,165,233,0.06) 0%, rgba(11,20,40,0.6) 50%, rgba(16,185,129,0.06) 100%)",
        padding: "48px 40px",
        position: "relative",
        overflow: "hidden",
      }}>
        {/* Subtle corner glows */}
        <div style={{ position: "absolute", top: -60, left: -60, width: 200, height: 200, borderRadius: "50%", background: `${C.teal}0C`, pointerEvents: "none" }} />
        <div style={{ position: "absolute", bottom: -60, right: -60, width: 200, height: 200, borderRadius: "50%", background: `${C.emerald}0C`, pointerEvents: "none" }} />

        <div style={{ textAlign: "center", marginBottom: 40, position: "relative" }}>
          <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", color: C.teal, marginBottom: 10 }}>
            Start here
          </div>
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 28, fontWeight: 800, color: "#fff", marginBottom: 8, letterSpacing: "-0.02em" }}>
            Your path to fairer hiring
          </h2>
          <p style={{ color: C.mist, fontSize: 15, maxWidth: 440, margin: "0 auto" }}>
            Each tool builds on the last. Follow the steps below or jump in anywhere.
          </p>
        </div>

        <div style={{ display: "flex", alignItems: "stretch", gap: 0, position: "relative", justifyContent: "center" }}>
          {/* Gradient connector line */}
          <div style={{
            position: "absolute",
            top: 36, left: "15%", right: "15%",
            height: 2,
            background: `linear-gradient(90deg, ${C.teal}, #8B5CF6, ${C.emerald})`,
            opacity: 0.35,
            pointerEvents: "none",
          }} />

          {[
            { num: "01", icon: "🔍", label: "Fix your JD", sub: "Detect & rewrite biased language before it shrinks your candidate pool.", page: "reducer", accent: C.teal },
            { num: "02", icon: "🤖", label: "Compare AI outcomes", sub: "See how bias shifts rankings — traditional AI vs. our bias-aware pipeline.", page: "hiring", accent: "#8B5CF6" },
            { num: "03", icon: "📊", label: "Score your org", sub: "Aggregate fairness across all your JDs into one benchmark score.", page: "fairindex", accent: C.emerald },
          ].map((s, i) => (
            <div
              key={s.num}
              onClick={() => setPage(s.page)}
              onMouseEnter={e => {
                e.currentTarget.style.background = s.accent + "14";
                e.currentTarget.style.borderColor = s.accent + "60";
                e.currentTarget.style.transform = "translateY(-4px)";
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.borderColor = C.ghost;
                e.currentTarget.style.transform = "translateY(0)";
              }}
              style={{
                cursor: "pointer",
                flex: 1,
                maxWidth: 280,
                textAlign: "center",
                padding: "28px 24px 24px",
                borderRadius: 16,
                border: `1px solid ${C.ghost}`,
                background: "transparent",
                position: "relative",
                transition: "all 0.2s ease",
                margin: "0 8px",
              }}
            >
              {/* Big step number */}
              <div style={{
                fontFamily: "'Fraunces', serif",
                fontSize: 48,
                fontWeight: 800,
                lineHeight: 1,
                marginBottom: 16,
                background: `linear-gradient(135deg, ${s.accent}, ${s.accent}66)`,
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
              }}>{s.num}</div>

              {/* Icon */}
              <div style={{
                width: 52, height: 52, borderRadius: 14,
                background: s.accent + "18",
                border: `1px solid ${s.accent}40`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 24, margin: "0 auto 16px",
                boxShadow: `0 0 20px ${s.accent}20`,
              }}>{s.icon}</div>

              <div style={{ fontSize: 16, fontWeight: 700, color: "#fff", marginBottom: 8 }}>{s.label}</div>
              <p style={{ fontSize: 13, color: C.mist, lineHeight: 1.6, marginBottom: 16 }}>{s.sub}</p>
              <span style={{ fontSize: 12, fontWeight: 700, color: s.accent, letterSpacing: "0.04em" }}>
                Go →
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── JD Bias Reducer ───────────────────────────────────────────────────────────

function ReducerPage() {
  const [text, setText]       = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [tab, setTab]         = useState("suggestions");
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
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 36, textAlign: "center" }}>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 800, marginBottom: 10, color: "#fff", letterSpacing: "-0.02em" }}>
          JD Bias Reducer
        </h1>
        <p style={{ color: C.slate, fontSize: 16, maxWidth: 520, margin: "0 auto 8px" }}>
          Paste a job description. We'll detect bias, explain it, and rewrite it — inclusively.
        </p>
        <p style={{ fontSize: 12, color: C.silver }}>
          ⚠ Do not input personal or identifying information.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Left — Input */}
        <div className="fade-up">
          <Card>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <label style={{ fontWeight: 700, fontSize: 14, color: "#fff" }}>Job Description</label>
              <span style={{ fontSize: 12, color: text.length > MAX * 0.9 ? C.rose : C.silver }}>
                {text.length} / {MAX}
              </span>
            </div>
            <textarea
              value={text}
              onChange={e => setText(e.target.value.slice(0, MAX))}
              placeholder={`Paste job description here…\n\nExample:\nWe are looking for a young, energetic salesman to join our team. The ideal candidate should be available 24/7 and have a dominant personality...`}
              style={{
                width: "100%",
                minHeight: 320,
                border: `1.5px solid ${C.ghost}`,
                borderRadius: 10,
                padding: "14px 16px",
                fontSize: 14,
                lineHeight: 1.65,
                color: C.slate,
                resize: "vertical",
                outline: "none",
                background: C.surface,
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
              <div style={{ width: 44, height: 44, borderRadius: "50%", border: `3px solid ${C.ghost}`, borderTopColor: C.teal, animation: "spin 0.8s linear infinite" }} />
              <p style={{ color: C.mist, fontSize: 14 }}>Detecting bias patterns…</p>
            </Card>
          )}
          {result && (
            <Card>
              {/* Score summary */}
              <div style={{ display: "flex", alignItems: "center", gap: 20, marginBottom: 20, padding: "16px", background: C.surface, borderRadius: 10, border: `1px solid ${C.ghost}` }}>
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
                      }>{cat.replace(/_/g, " ")}</Pill>
                    ))}
                    {result.categories?.length === 0 && <Pill color={C.emerald}>No bias detected</Pill>}
                  </div>
                </div>
              </div>

              {/* Flagged spans */}
              {result.highlights?.length > 0 && (
                <div style={{ marginBottom: 16 }}>
                  <SectionLabel>Flagged Phrases</SectionLabel>
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
                    fontWeight: tab === t ? 700 : 400,
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
                        background: C.surface,
                        borderRadius: 8,
                        padding: "14px 16px",
                        whiteSpace: "pre-wrap",
                        fontFamily: "'DM Sans', sans-serif",
                        fontSize: 13.5,
                        border: `1px solid ${C.ghost}`,
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
  const [step, setStep]       = useState(1);
  const [jd, setJd]           = useState("");
  const [resume, setResume]   = useState("");
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

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
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: "48px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 36, textAlign: "center" }}>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 800, marginBottom: 10, color: "#fff", letterSpacing: "-0.02em" }}>
          Bias-Aware Hiring AI
        </h1>
        <p style={{ color: C.slate, fontSize: 16, maxWidth: 560, margin: "0 auto 14px" }}>
          Evaluate candidate fit using PII-stripped resumes and bias-reduced JDs.
          See how traditional AI compares side-by-side.
        </p>
        <div style={{ display: "inline-flex", padding: "8px 16px", background: `${C.teal}12`, border: `1px solid ${C.teal}30`, borderRadius: 8, fontSize: 13, color: C.teal, gap: 6, alignItems: "center" }}>
          🔒 Resumes are never stored. PII is stripped before any evaluation.
        </div>
      </div>

      {/* Stepper */}
      <div style={{ display: "flex", gap: 0, marginBottom: 32, background: C.surface, borderRadius: 12, padding: 4, border: `1px solid ${C.ghost}` }}>
        {[1, 2, 3].map(s => (
          <button key={s} onClick={() => s < step && setStep(s)} style={{
            flex: 1,
            padding: "10px 16px",
            borderRadius: 10,
            border: "none",
            background: step === s ? C.ghost : "transparent",
            color: step === s ? C.teal : step > s ? C.emerald : C.silver,
            fontWeight: step === s ? 700 : 400,
            fontSize: 13,
            fontFamily: "'DM Sans', sans-serif",
            boxShadow: step === s ? "0 1px 8px rgba(0,0,0,0.3)" : "none",
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
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 22, fontWeight: 700, marginBottom: 6, color: "#fff" }}>Step 1 — Paste Job Description</h2>
          <p style={{ fontSize: 13, color: C.mist, marginBottom: 16 }}>Use a bias-reduced JD from the Bias Reducer for best results.</p>
          <textarea
            value={jd}
            onChange={e => setJd(e.target.value)}
            placeholder="Paste the (rewritten, bias-reduced) job description here…"
            style={{
              width: "100%", minHeight: 260,
              border: `1.5px solid ${C.ghost}`, borderRadius: 10,
              padding: "14px 16px", fontSize: 14, lineHeight: 1.65,
              color: C.slate, resize: "vertical", outline: "none", background: C.surface,
            }}
          />
          <div style={{ marginTop: 14 }}>
            <Btn onClick={() => setStep(2)} disabled={!jd.trim()}>Next: Add Resume →</Btn>
          </div>
        </Card>
      )}

      {step === 2 && (
        <Card className="fade-up">
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 22, fontWeight: 700, marginBottom: 6, color: "#fff" }}>Step 2 — Paste Resume</h2>
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
              color: C.slate, resize: "vertical", outline: "none", background: C.surface,
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
                  <span style={{ fontWeight: 700, fontSize: 14, color: "#fff" }}>{icon} {label}</span>
                  <span style={{
                    fontFamily: "'Fraunces', serif",
                    fontSize: 28,
                    fontWeight: 800,
                    color,
                  }}>{Math.round((data?.fit_score || 0) * 100)}%</span>
                </div>
                <div style={{ marginBottom: 8 }}>
                  <Pill color={
                    data?.fit_level === "strong" ? C.emerald :
                    data?.fit_level === "moderate" ? C.amber : C.rose
                  }>{data?.fit_level || "—"} fit</Pill>
                </div>
                <p style={{ fontSize: 13, color: C.slate, lineHeight: 1.6 }}>{data?.explanation || "—"}</p>
              </Card>
            ))}
          </div>

          {/* Delta callout */}
          <Card style={{
            background: `linear-gradient(135deg, rgba(14,165,233,0.12), rgba(16,185,129,0.08))`,
            border: `1px solid ${C.teal}30`,
            marginBottom: 24,
            display: "flex",
            alignItems: "center",
            gap: 24,
          }}>
            <div style={{ fontFamily: "'Fraunces', serif", fontSize: 52, fontWeight: 800, color: C.teal }}>
              {result.score_delta}
            </div>
            <div>
              <div style={{ color: "#fff", fontWeight: 700, fontSize: 16, marginBottom: 4 }}>Score Delta</div>
              <div style={{ color: C.slate, fontSize: 14 }}>
                {parseFloat(result.score_delta) >= 0
                  ? "Our bias-aware system ranked this candidate higher than traditional AI would have."
                  : "Traditional AI may have over-fitted on biased signals for this candidate."}
              </div>
            </div>
          </Card>

          {/* Skill matches */}
          {result.bias_aware?.skill_matches?.length > 0 && (
            <Card style={{ marginBottom: 24 }}>
              <h3 style={{ fontFamily: "'Fraunces', serif", fontSize: 20, fontWeight: 700, marginBottom: 16, color: "#fff" }}>Skill Match Breakdown</h3>
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                {result.bias_aware.skill_matches.map((m, i) => (
                  <div key={i} style={{
                    display: "flex", gap: 12, alignItems: "flex-start",
                    padding: "10px 14px",
                    background: C.surface,
                    borderRadius: 8,
                    borderLeft: `3px solid ${matchColor[m.match] || C.silver}`,
                    border: `1px solid ${C.ghost}`,
                  }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 700, fontSize: 13, color: "#fff", marginBottom: 3 }}>{m.requirement}</div>
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
              <h3 style={{ fontFamily: "'Fraunces', serif", fontSize: 16, fontWeight: 700, marginBottom: 10, color: "#fff" }}>🔒 Bias Signals Suppressed</h3>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {result.bias_aware.bias_signals_suppressed.map((s, i) => (
                  <span key={i} style={{
                    padding: "3px 10px",
                    borderRadius: 6,
                    background: `${C.teal}12`,
                    color: C.teal,
                    fontSize: 12,
                    fontWeight: 600,
                  }}>{s}</span>
                ))}
              </div>
            </Card>
          )}

          <div style={{ marginTop: 20 }}>
            <Btn variant="ghost" onClick={() => { setStep(1); setResult(null); setJd(""); setResume(""); }}>
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
  const [jds, setJds]         = useState([{ text: "", id: Date.now() }]);
  const [scores, setScores]   = useState(null);
  const [loading, setLoading] = useState(false);

  const WEIGHTS = { explicit_gender: 1.0, stereotype: 0.8, age_bias: 0.7, requirements_bias: 0.6 };
  const MAX_JD = 3000;

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
    <div style={{ maxWidth: 1000, margin: "0 auto", padding: "48px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 36, textAlign: "center" }}>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 800, marginBottom: 10, color: "#fff", letterSpacing: "-0.02em" }}>
          Fair Hiring Index
        </h1>
        <p style={{ color: C.slate, fontSize: 16, maxWidth: 540, margin: "0 auto" }}>
          Paste one or more job descriptions to get a single 0–100 fairness score.
          Higher is fairer — scores are weighted by the severity of each bias type detected.
        </p>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
        {/* Input */}
        <Card className="fade-up">
          <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 20, fontWeight: 700, marginBottom: 16, color: "#fff" }}>
            Batch Job Descriptions
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {jds.map((jd, i) => (
              <div key={jd.id}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <SectionLabel>JD #{i + 1}</SectionLabel>
                  <span style={{ fontSize: 11, color: jd.text.length > MAX_JD * 0.9 ? C.rose : C.silver }}>
                    {jd.text.length} / {MAX_JD}
                  </span>
                </div>
                <textarea
                  value={jd.text}
                  onChange={e => setJds(prev => prev.map(j => j.id === jd.id ? { ...j, text: e.target.value.slice(0, MAX_JD) } : j))}
                  placeholder="Paste job description…"
                  style={{
                    width: "100%", minHeight: 100, marginTop: 4,
                    border: `1.5px solid ${C.ghost}`, borderRadius: 8,
                    padding: "10px 12px", fontSize: 13, lineHeight: 1.6,
                    color: C.slate, resize: "vertical", outline: "none", background: C.surface,
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
          <div style={{ marginTop: 20, padding: "16px", background: C.surface, borderRadius: 8, border: `1px solid ${C.ghost}` }}>
            <SectionLabel>How the score is calculated</SectionLabel>
            <div style={{ fontFamily: "monospace", fontSize: 13, color: C.slate, lineHeight: 1.9 }}>
              FHI = 100 × (1 − (1/N) × Σ(Bᵢ × Cᵢ))<br />
              <span style={{ color: C.mist, fontSize: 11 }}>Bᵢ = bias score per JD · Cᵢ = category severity weight</span>
            </div>
            <div style={{ marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6 }}>
              {Object.entries(WEIGHTS).map(([k, v]) => (
                <span key={k} style={{ fontSize: 11, padding: "2px 8px", borderRadius: 4, background: C.ghost, color: C.slate }}>
                  {k.replace(/_/g, " ")}: {v}×
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
              {/* FHI Score */}
              <Card style={{ textAlign: "center", borderTop: `3px solid ${fhiColor}` }}>
                <SectionLabel>Fair Hiring Index</SectionLabel>
                <div style={{ fontFamily: "'Fraunces', serif", fontSize: 80, fontWeight: 800, color: fhiColor, lineHeight: 1, marginTop: 8, textShadow: `0 0 40px ${fhiColor}60` }}>
                  {scores.fhi}
                </div>
                <div style={{ fontSize: 14, color: C.mist, marginTop: 6 }}>out of 100</div>
                <div style={{ marginTop: 10 }}>
                  <Pill color={fhiColor}>
                    {scores.fhi >= 75 ? "High Fairness" : scores.fhi >= 50 ? "Moderate Fairness" : "Low Fairness"}
                  </Pill>
                </div>
              </Card>

              {/* Interpretation */}
              <Card style={{ padding: "16px 20px" }}>
                <SectionLabel>What this means</SectionLabel>
                <p style={{ fontSize: 13, color: C.slate, lineHeight: 1.7, marginTop: 6 }}>
                  {scores.fhi >= 75
                    ? "Your hiring language is largely inclusive. Small refinements to the flagged phrases below may help attract an even broader candidate pool."
                    : scores.fhi >= 50
                    ? "Some patterns of exclusionary language detected. Companies at this level typically see 15–30% fewer applications from underrepresented groups. Prioritize the flagged phrases below."
                    : "Multiple bias patterns detected across your job descriptions. This significantly reduces applicant diversity. Use the JD Bias Reducer to rewrite each affected description."}
                </p>
                {scores.fhi < 75 && (
                  <div style={{ marginTop: 10, fontSize: 12, color: C.teal, fontWeight: 700 }}>
                    → Run each flagged JD through the Bias Reducer for an inclusive rewrite.
                  </div>
                )}
              </Card>

              {/* Per-JD breakdown */}
              {scores.results.map((r, i) => {
                const levelColor = r.bias_level === "low" ? C.emerald : r.bias_level === "medium" ? C.amber : C.rose;
                return (
                  <Card key={i} style={{ borderLeft: `3px solid ${levelColor}` }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                      <span style={{ fontWeight: 700, fontSize: 13, color: "#fff" }}>JD #{i + 1}</span>
                      <span style={{ fontFamily: "'Fraunces', serif", fontWeight: 700, color: levelColor }}>
                        {Math.round(r.bias_score * 100)}% bias
                      </span>
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginBottom: r.highlights?.length ? 10 : 0 }}>
                      {r.categories?.length ? r.categories.map(c => <Pill key={c} color={C.mist}>{c.replace(/_/g, " ")}</Pill>) : <Pill color={C.emerald}>clean</Pill>}
                    </div>
                    {r.highlights?.length > 0 && (
                      <div style={{ marginBottom: r.suggestions?.length ? 8 : 0 }}>
                        <SectionLabel>Flagged Phrases</SectionLabel>
                        <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                          {r.highlights.slice(0, 4).map((h, j) => <HighlightChip key={j} span={h.span} type={h.type} />)}
                          {r.highlights.length > 4 && (
                            <span style={{ fontSize: 12, color: C.silver, alignSelf: "center" }}>+{r.highlights.length - 4} more</span>
                          )}
                        </div>
                      </div>
                    )}
                    {r.suggestions?.length > 0 && (
                      <div style={{ padding: "8px 10px", background: `${C.teal}08`, border: `1px solid ${C.teal}20`, borderRadius: 6, fontSize: 12, color: C.slate, lineHeight: 1.55 }}>
                        <span style={{ fontWeight: 700, color: C.teal }}>Suggestion: </span>{r.suggestions[0]}
                      </div>
                    )}
                  </Card>
                );
              })}

              {/* Priority improvements */}
              {(() => {
                const ADVICE = {
                  explicit_gender: "Avoid gendered job titles and pronouns. Use 'they/them' or gender-neutral terms like 'engineer' instead of 'salesman'.",
                  stereotype: "Remove culturally coded language (e.g. 'rockstar', 'ninja', 'culture fit') that skews toward specific demographics.",
                  age_bias: "Avoid phrases implying age preference (e.g. 'young', 'recent graduate', 'digital native'). Focus on experience, not tenure.",
                  requirements_bias: "Audit degree and years-of-experience requirements. Use skills-based criteria where a degree isn't strictly necessary.",
                };
                const catCounts = {};
                scores.results.forEach(r => r.categories?.forEach(c => { catCounts[c] = (catCounts[c] || 0) + 1; }));
                const sorted = Object.entries(catCounts).sort((a, b) => (WEIGHTS[b[0]] || 0.5) - (WEIGHTS[a[0]] || 0.5));
                if (!sorted.length) return null;
                return (
                  <Card>
                    <SectionLabel>Priority Improvements</SectionLabel>
                    <div style={{ display: "flex", flexDirection: "column", gap: 10, marginTop: 8 }}>
                      {sorted.map(([cat, count]) => {
                        const dotColor = cat === "explicit_gender" ? C.rose : cat === "stereotype" ? C.amber : cat === "age_bias" ? "#F97316" : "#8B5CF6";
                        return (
                          <div key={cat} style={{ display: "flex", gap: 10, alignItems: "flex-start", padding: "10px 12px", background: C.surface, borderRadius: 8, border: `1px solid ${C.ghost}` }}>
                            <div style={{ minWidth: 8, height: 8, borderRadius: "50%", background: dotColor, marginTop: 4, flexShrink: 0 }} />
                            <div>
                              <div style={{ fontWeight: 700, fontSize: 12, color: "#fff", marginBottom: 3 }}>
                                {cat.replace(/_/g, " ")} · {count} JD{count > 1 ? "s" : ""}
                              </div>
                              <div style={{ fontSize: 12, color: C.mist, lineHeight: 1.55 }}>{ADVICE[cat] || "Review flagged phrases above."}</div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card>
                );
              })()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── About ─────────────────────────────────────────────────────────────────────

function AboutPage() {
  const features = [
    {
      icon: "🔍",
      title: "JD Bias Reducer",
      status: "Live",
      statusColor: C.emerald,
      accent: C.teal,
      what: "Detects four categories of bias in job descriptions: explicit gender language (e.g. \"salesman\", \"he/she\"), stereotyping (language that implicitly favors one gender's traits), age bias (phrases that exclude older or younger candidates), and requirements bias (credentials that are not genuinely necessary for the role).",
      how: "Each flagged phrase is explained, scored, and replaced with an inclusive alternative. The full JD is rewritten and ready to post.",
      implications: "Biased language causes qualified candidates to self-screen before applying. By intervening at the source — the job description — this tool reduces structural exclusion before the pipeline even starts.",
    },
    {
      icon: "📊",
      title: "Fair Hiring Index",
      status: "Live",
      statusColor: C.emerald,
      accent: C.emerald,
      what: "Aggregates bias scores across one or many job descriptions into a single 0–100 fairness score. Each bias category is weighted by severity: explicit gender bias counts most, requirements bias least. The score reflects how equitable your hiring language is as a whole.",
      how: "Paste multiple JDs at once. The index runs each through the bias detector, applies category weights, and produces a composite score with a per-JD breakdown.",
      implications: "Individual JDs can look fine in isolation but reveal a pattern at scale. The Fair Hiring Index makes that pattern visible — giving teams a benchmark to track over time and compare across departments.",
    },
    {
      icon: "🤖",
      title: "Bias-Aware Hiring AI",
      status: "In Progress",
      statusColor: C.amber,
      accent: "#8B5CF6",
      what: "Evaluates a candidate's resume against a job description using two separate AI pipelines: a traditional AI (which sees the original, potentially biased JD and unredacted resume) and a bias-aware AI (which uses a bias-reduced JD and a PII-stripped resume). Both return a fit score and skill match breakdown.",
      how: "The two scores are compared side-by-side. The delta between them reveals exactly how much bias in the original pipeline disadvantaged — or advantaged — the candidate.",
      implications: "Most hiring AI systems learn from historical data that reflects past discrimination. By racing the two systems, BIOS Check shows not just that bias exists, but how much it changes outcomes for real candidates.",
    },
  ];

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "60px 32px" }}>
      {/* Hero */}
      <div className="fade-up" style={{ marginBottom: 56, textAlign: "center" }}>
        <Pill color={C.teal}>About the Project</Pill>
        <h1 style={{
          fontFamily: "'Fraunces', serif",
          fontSize: "clamp(36px, 5vw, 52px)",
          fontWeight: 800,
          marginTop: 20,
          marginBottom: 24,
          letterSpacing: "-0.03em",
          color: "#fff",
          lineHeight: 1.1,
        }}>
          Measuring what matters.
        </h1>
        <p style={{ fontSize: 17, color: C.slate, lineHeight: 1.8, marginBottom: 16, maxWidth: 640, margin: "0 auto 16px" }}>
          BIOS Check is a research project investigating how gender bias in language propagates through AI hiring systems — and how to stop it.
        </p>
        <p style={{ fontSize: 15, color: C.mist, lineHeight: 1.8, maxWidth: 640, margin: "0 auto" }}>
          Most bias-detection tools work retroactively, flagging decisions already made. This project is designed differently: by stripping demographic signals before evaluation and using bias-reduced job descriptions as rubrics, the hiring AI is structurally incapable of acting on bias signals — not just instructed to ignore them.
        </p>
      </div>

      {/* Research stats banner */}
      <Card className="fade-up-2" style={{
        background: `linear-gradient(135deg, rgba(14,165,233,0.14), rgba(16,185,129,0.07))`,
        border: `1px solid ${C.teal}30`,
        marginBottom: 56,
        display: "flex",
        alignItems: "center",
        gap: 32,
        flexWrap: "wrap",
      }}>
        <div style={{ flex: 1, minWidth: 240 }}>
          <Pill color={C.teal}>Research Insight</Pill>
          <h2 style={{
            fontFamily: "'Fraunces', serif",
            fontSize: 24,
            fontWeight: 700,
            marginTop: 14,
            marginBottom: 10,
            color: "#fff",
          }}>Why does this matter?</h2>
          <p style={{ fontSize: 14, color: C.mist, lineHeight: 1.75 }}>
            Studies show that gendered language in job postings reduces the applicant pool by up to 42%
            for underrepresented groups. Embedding-based AI screeners amplify this further — a bias
            present in training data becomes a structural disadvantage at scale.
          </p>
        </div>
        <div style={{ display: "flex", gap: 28, flexWrap: "wrap" }}>
          {[["42%", "fewer applicants from biased JDs"], ["87%", "of hiring AIs trained on biased data"], ["3×", "more likely to screen out qualified candidates"]].map(([stat, desc]) => (
            <div key={stat} style={{ textAlign: "center" }}>
              <div style={{ fontFamily: "'Fraunces', serif", fontSize: 38, fontWeight: 800, color: C.teal, textShadow: `0 0 20px ${C.teal}60` }}>{stat}</div>
              <div style={{ fontSize: 12, color: C.mist, maxWidth: 100, marginTop: 4 }}>{desc}</div>
            </div>
          ))}
        </div>
      </Card>

      {/* Feature deep-dives */}
      <div className="fade-up-3" style={{ marginBottom: 56 }}>
        <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 28, fontWeight: 800, marginBottom: 28, color: "#fff", textAlign: "center" }}>
          What each tool does
        </h2>
        <div style={{ display: "grid", gap: 24 }}>
          {features.map(f => (
            <Card key={f.title} style={{ borderLeft: `4px solid ${f.accent}` }}>
              <div style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 20 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 11,
                  background: f.accent + "18",
                  border: `1px solid ${f.accent}30`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 20,
                }}>{f.icon}</div>
                <span style={{ fontFamily: "'Fraunces', serif", fontSize: 22, fontWeight: 700, color: "#fff" }}>{f.title}</span>
                <Pill color={f.statusColor}>{f.status}</Pill>
              </div>

              <div style={{ display: "grid", gap: 18 }}>
                {[
                  { label: "What it does", text: f.what },
                  { label: "How it works", text: f.how },
                  { label: "Why it matters", text: f.implications },
                ].map(({ label, text }) => (
                  <div key={label}>
                    <SectionLabel>{label}</SectionLabel>
                    <p style={{ fontSize: 14, color: C.slate, lineHeight: 1.75 }}>{text}</p>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
      </div>

      {/* GitHub CTA */}
      <Card style={{
        marginBottom: 32,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        flexWrap: "wrap",
        gap: 16,
        background: `linear-gradient(135deg, rgba(14,165,233,0.08), rgba(16,185,129,0.04))`,
        border: `1px solid ${C.teal}20`,
      }}>
        <div>
          <div style={{ fontFamily: "'Fraunces', serif", fontSize: 18, fontWeight: 700, color: "#fff", marginBottom: 6 }}>
            Curious about the methodology?
          </div>
          <p style={{ fontSize: 14, color: C.mist, lineHeight: 1.6 }}>
            The full technical implementation — models, training data, bias detection logic, and evaluation pipeline — is open source.
          </p>
        </div>
        <Btn variant="outline" onClick={() => window.open("https://github.com/SaanikaGW/Bias-Detector", "_blank")}>
          View Source →
        </Btn>
      </Card>

      {/* Author */}
      <Card style={{
        background: `linear-gradient(135deg, rgba(14,165,233,0.1), rgba(16,185,129,0.06))`,
        border: `1px solid ${C.teal}25`,
      }}>
        <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
          <div style={{
            width: 58, height: 58, borderRadius: "50%",
            background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 24, flexShrink: 0,
            boxShadow: `0 0 24px ${C.teal}50`,
          }}>👩‍💻</div>
          <div>
            <div style={{ color: "#fff", fontFamily: "'Fraunces', serif", fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
              Saanika
            </div>
            <div style={{ color: C.mist, fontSize: 13, lineHeight: 1.6 }}>
              Builder of BIOS Check · Researching fairness in AI hiring systems · Mentored by Jason
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
}

// ── Contact ───────────────────────────────────────────────────────────────────

function ContactPage() {
  // STEP: Go to forms.google.com → create form → Send → Embed (</>) → copy the src="..." URL and paste below
  const GOOGLE_FORM_SRC = "https://docs.google.com/forms/d/e/1FAIpQLSfYfRGCYAwyZtF0HSkqgf-0eF8P6emtfpYXqBwXEf0_MXKj5w/viewform?embedded=true";

  return (
    <div style={{ maxWidth: 660, margin: "0 auto", padding: "60px 32px" }}>
      <div className="fade-up" style={{ marginBottom: 36, textAlign: "center" }}>
        <Pill>Get in Touch</Pill>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 800, marginTop: 16, marginBottom: 10, color: "#fff", letterSpacing: "-0.02em" }}>
          Contact Us
        </h1>
        <p style={{ color: C.mist, fontSize: 15 }}>Feedback, research questions, or collaboration inquiries.</p>
      </div>

      <Card className="fade-up" style={{ padding: 0, overflow: "hidden", border: `1px solid ${C.ghost}` }}>
        <iframe
          src={GOOGLE_FORM_SRC}
          width="100%"
          height="1008"
          frameBorder="0"
          marginHeight="0"
          marginWidth="0"
          title="Contact Form"
          style={{ display: "block" }}
        >
          Loading form…
        </iframe>
      </Card>
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
      <main style={{ minHeight: "calc(100vh - 60px)" }}>
        {pages[page] || pages.home}
      </main>
      <footer style={{
        borderTop: `1px solid ${C.ghost}`,
        padding: "28px 40px",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 12,
        fontSize: 13,
        color: C.silver,
        background: "rgba(6,13,31,0.6)",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 7,
            background: `linear-gradient(135deg, ${C.teal}, ${C.emerald})`,
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 13,
          }}>⚖️</div>
          <span style={{ fontFamily: "'Fraunces', serif", color: C.mist, fontWeight: 600 }}>BIOS Check</span>
          <span style={{ color: C.silver }}>— Making fair hiring measurable.</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          {[
            ["Home", "home"], ["Bias Reducer", "reducer"], ["Hiring AI", "hiring"],
            ["Fair Index", "fairindex"], ["About", "about"],
          ].map(([l, id]) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              style={{ background: "none", border: "none", color: C.silver, fontSize: 13, cursor: "pointer", fontFamily: "'DM Sans', sans-serif", padding: "4px 10px", borderRadius: 6, transition: "color 0.15s" }}
            >
              {l}
            </button>
          ))}
        </div>
      </footer>
    </>
  );
}

import { useState, useEffect } from "react";
import { getUserId, getUsageType } from "./shared/usageMetrics.js";
import {
  submitFeedback, fetchTestimonials, adminPending, adminApprove, adminDelete,
  getFeedbackState, setFeedbackState,
} from "./shared/feedbackClient.js";

// Self-contained UI for testimonials/feedback. App.jsx passes its live theme
// palette (C) and API base so this file matches the rest of the site.

const RUN_EVENT = "bios:run-complete";

function Stars({ value, onChange, C, size = 22 }) {
  return (
    <span role={onChange ? "radiogroup" : undefined} aria-label={`${value} out of 5 stars`}>
      {[1, 2, 3, 4, 5].map(n => (
        <span
          key={n}
          role={onChange ? "radio" : undefined}
          aria-checked={onChange ? value === n : undefined}
          aria-label={`${n} star${n > 1 ? "s" : ""}`}
          tabIndex={onChange ? 0 : undefined}
          onClick={onChange ? () => onChange(n) : undefined}
          onKeyDown={onChange ? e => { if (e.key === "Enter" || e.key === " ") onChange(n); } : undefined}
          style={{
            fontSize: size, cursor: onChange ? "pointer" : "default",
            color: n <= value ? C.amber : C.ghost, marginRight: 2,
          }}
        >★</span>
      ))}
    </span>
  );
}

function FeedbackForm({ C, api }) {
  const stored = getUsageType();
  const [userType, setUserType] = useState(stored.userType === "company" ? "company" : "individual");
  const [rating, setRating] = useState(0);
  const [message, setMessage] = useState("");
  const [showName, setShowName] = useState(false);
  const [name, setName] = useState(stored.companyName && stored.userType === "company" ? stored.companyName : "");
  const [role, setRole] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [done, setDone] = useState(false);

  const input = {
    width: "100%", padding: "10px 12px", borderRadius: 10, fontSize: 14,
    background: C.snow, color: C.ink, border: `1px solid ${C.ghost}`,
  };
  const label = { display: "block", fontSize: 13, fontWeight: 600, color: C.slate, marginBottom: 6 };
  const isCompany = userType === "company";

  async function send() {
    setError("");
    if (!rating) return setError("Please choose a star rating.");
    if (message.trim().length < 10) return setError("Please write at least a sentence of feedback.");
    if (showName && !name.trim()) return setError(isCompany ? "Add your organization's name, or choose to stay anonymous." : "Add your name, or choose to stay anonymous.");
    setBusy(true);
    try {
      await submitFeedback(api, {
        user_id: getUserId(), user_type: userType, rating,
        message: message.trim(), show_name: showName,
        display_name: showName ? name.trim() : null, role: role.trim() || null,
      });
      setFeedbackState("submitted");
      setDone(true);
    } catch (e) {
      setError(e.message || "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  if (done) {
    return (
      <div style={{ textAlign: "center", padding: "24px 8px" }}>
        <div style={{ fontSize: 36, marginBottom: 8 }}>🙏</div>
        <div style={{ fontFamily: "'Fraunces', serif", fontSize: 22, fontWeight: 700, color: C.ink, marginBottom: 6 }}>
          Thank you!
        </div>
        <p style={{ color: C.mist, fontSize: 14 }}>
          Your feedback was received. It will appear on this page once it has been reviewed.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", gap: 8, marginBottom: 18 }}>
        {[["individual", "I'm an individual"], ["company", "I'm with an organization"]].map(([id, text]) => (
          <button
            key={id} type="button" onClick={() => setUserType(id)}
            style={{
              flex: 1, padding: "9px 10px", borderRadius: 10, fontSize: 13, fontWeight: 700,
              background: userType === id ? `${C.teal}20` : "transparent",
              color: userType === id ? C.teal : C.mist,
              border: `1.5px solid ${userType === id ? C.teal : C.ghost}`,
            }}
          >{text}</button>
        ))}
      </div>

      <div style={{ marginBottom: 16 }}>
        <span style={label}>How would you rate BIOS Check?</span>
        <Stars value={rating} onChange={setRating} C={C} size={30} />
      </div>

      <div style={{ marginBottom: 16 }}>
        <label style={label} htmlFor="fb-message">Your feedback</label>
        <textarea
          id="fb-message" rows={5} maxLength={1000} value={message}
          onChange={e => setMessage(e.target.value)}
          placeholder="What did you use it for, and how did it help?"
          style={{ ...input, resize: "vertical" }}
        />
        <div style={{ fontSize: 11, color: C.silver, textAlign: "right" }}>{message.length} / 1000</div>
      </div>

      <div style={{ marginBottom: 16 }}>
        <span style={label}>{isCompany ? "Would you like your organization named?" : "Would you like to be named?"}</span>
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", fontSize: 14, color: C.slate }}>
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="radio" checked={!showName} onChange={() => setShowName(false)} /> Keep me anonymous
          </label>
          <label style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <input type="radio" checked={showName} onChange={() => setShowName(true)} />
            {isCompany ? "Show our name" : "Show my name"}
          </label>
        </div>
      </div>

      {showName && (
        <div style={{ marginBottom: 16 }}>
          <label style={label} htmlFor="fb-name">{isCompany ? "Organization name" : "Your name"}</label>
          <input id="fb-name" value={name} maxLength={80} onChange={e => setName(e.target.value)} style={input} />
        </div>
      )}

      <div style={{ marginBottom: 18 }}>
        <label style={label} htmlFor="fb-role">
          {isCompany ? "Your role (optional)" : "Role or background (optional)"}
        </label>
        <input
          id="fb-role" value={role} maxLength={80} onChange={e => setRole(e.target.value)}
          placeholder={isCompany ? "e.g. Head of Talent" : "e.g. College counselor, job seeker"}
          style={input}
        />
      </div>

      {error && <div role="alert" style={{ color: C.rose, fontSize: 13, marginBottom: 12 }}>{error}</div>}

      <button
        type="button" onClick={send} disabled={busy}
        style={{
          padding: "12px 26px", borderRadius: 10, border: "none", fontWeight: 700, fontSize: 14,
          background: `linear-gradient(135deg, ${C.teal}, ${C.tealDark})`, color: "#fff",
          opacity: busy ? 0.6 : 1, cursor: busy ? "not-allowed" : "pointer",
        }}
      >{busy ? "Sending…" : "Submit feedback"}</button>
    </div>
  );
}

function TestimonialCard({ t, C }) {
  const who = t.display_name || "Anonymous";
  const sub = [t.role, t.user_type === "company" ? "Organization" : "Individual"].filter(Boolean).join(" · ");
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.ghost}`, borderRadius: 16, padding: 22,
      display: "flex", flexDirection: "column", gap: 10,
    }}>
      <Stars value={t.rating} C={C} size={16} />
      <p style={{ color: C.slate, fontSize: 14.5, lineHeight: 1.6 }}>“{t.message}”</p>
      <div style={{ marginTop: "auto" }}>
        <div style={{ fontWeight: 700, color: C.ink, fontSize: 14 }}>{who}</div>
        <div style={{ color: C.silver, fontSize: 12 }}>{sub}</div>
      </div>
    </div>
  );
}

function AdminPanel({ C, api, adminKey }) {
  const [items, setItems] = useState(null);
  const [err, setErr] = useState("");

  const load = () => adminPending(api, adminKey)
    .then(d => { setItems(d.items); setErr(""); })
    .catch(e => setErr(e.message));
  useEffect(() => { load(); /* eslint-disable-next-line */ }, []);

  async function act(fn, id) {
    try { await fn(api, adminKey, id); await load(); } catch (e) { setErr(e.message); }
  }

  const btn = { padding: "6px 14px", borderRadius: 8, fontSize: 13, fontWeight: 700, border: "none" };
  return (
    <div style={{ marginTop: 40, padding: 22, border: `1px dashed ${C.amber}`, borderRadius: 16 }}>
      <div style={{ fontWeight: 800, color: C.amber, marginBottom: 12 }}>Moderation — pending feedback</div>
      {err && <div style={{ color: C.rose, fontSize: 13, marginBottom: 8 }}>{err}</div>}
      {items && items.length === 0 && <div style={{ color: C.mist, fontSize: 14 }}>Nothing waiting for review.</div>}
      {(items || []).map(t => (
        <div key={t.id} style={{ borderTop: `1px solid ${C.ghost}`, padding: "14px 0" }}>
          <Stars value={t.rating} C={C} size={14} />
          <p style={{ color: C.slate, fontSize: 14, margin: "6px 0" }}>{t.message}</p>
          <div style={{ color: C.silver, fontSize: 12, marginBottom: 8 }}>
            {t.display_name || "Anonymous"} · {t.role || "no role"} · {t.user_type}
          </div>
          <button style={{ ...btn, background: C.emerald, color: "#fff", marginRight: 8 }} onClick={() => act(adminApprove, t.id)}>Approve</button>
          <button style={{ ...btn, background: C.rose, color: "#fff" }} onClick={() => act(adminDelete, t.id)}>Delete</button>
        </div>
      ))}
    </div>
  );
}

export function FeedbackPage({ C, api }) {
  const [data, setData] = useState(null);
  const [loadErr, setLoadErr] = useState("");
  const adminKey = new URLSearchParams(window.location.search).get("admin");

  useEffect(() => {
    fetchTestimonials(api).then(setData).catch(e => setLoadErr(e.message));
  }, [api]);

  return (
    <div style={{ maxWidth: 980, margin: "0 auto", padding: "60px 32px" }}>
      <div className="fade-up" style={{ textAlign: "center", marginBottom: 40 }}>
        <h1 style={{ fontFamily: "'Fraunces', serif", fontSize: 36, fontWeight: 800, color: C.ink, letterSpacing: "-0.02em", marginBottom: 10 }}>
          Testimonials &amp; Feedback
        </h1>
        <p style={{ color: C.mist, fontSize: 15, maxWidth: 560, margin: "0 auto" }}>
          What individuals and organizations say about BIOS Check. Used it? Tell us how it went.
        </p>
        {data && data.count > 0 && (
          <div style={{ marginTop: 14, display: "inline-flex", alignItems: "center", gap: 8, color: C.slate, fontSize: 14 }}>
            <Stars value={Math.round(data.avg_rating)} C={C} size={18} />
            <strong>{data.avg_rating}</strong> from {data.count} review{data.count > 1 ? "s" : ""}
          </div>
        )}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16, marginBottom: 48 }}>
        {loadErr && <div style={{ color: C.rose, fontSize: 14 }}>{loadErr}</div>}
        {data && data.items.length === 0 && (
          <div style={{ color: C.mist, fontSize: 15, gridColumn: "1 / -1", textAlign: "center" }}>
            No testimonials yet. Be the first to leave one below.
          </div>
        )}
        {(data?.items || []).map(t => <TestimonialCard key={t.id} t={t} C={C} />)}
      </div>

      <div id="leave-feedback" style={{
        background: C.card, border: `1px solid ${C.ghost}`, borderRadius: 18,
        padding: "28px 32px", maxWidth: 620, margin: "0 auto",
      }}>
        <h2 style={{ fontFamily: "'Fraunces', serif", fontSize: 24, fontWeight: 700, color: C.ink, marginBottom: 16 }}>
          Leave feedback
        </h2>
        <FeedbackForm C={C} api={api} />
      </div>

      {adminKey && <AdminPanel C={C} api={api} adminKey={adminKey} />}
    </div>
  );
}

/** Small prompt shown once, after a visitor finishes a tool run. */
export function FeedbackPrompt({ C, setPage }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    let timer;
    const onRun = () => {
      if (getFeedbackState()) return; // already submitted or dismissed
      clearTimeout(timer);
      timer = setTimeout(() => setVisible(true), 3500);
    };
    window.addEventListener(RUN_EVENT, onRun);
    return () => { window.removeEventListener(RUN_EVENT, onRun); clearTimeout(timer); };
  }, []);

  if (!visible) return null;

  const close = () => { setFeedbackState("dismissed"); setVisible(false); };
  const go = () => {
    setVisible(false);
    setPage("feedback");
    setTimeout(() => document.getElementById("leave-feedback")?.scrollIntoView({ behavior: "smooth" }), 150);
  };

  return (
    <div role="dialog" aria-label="Leave feedback" style={{
      position: "fixed", bottom: 20, left: 20, zIndex: 200, maxWidth: 320,
      background: C.surface, border: `1px solid ${C.ghost}`, borderRadius: 14,
      padding: 18, boxShadow: "0 12px 32px rgba(0,0,0,0.25)",
    }}>
      <button onClick={close} aria-label="Dismiss" style={{
        position: "absolute", top: 10, right: 12, background: "none", border: "none", color: C.silver, fontSize: 14,
      }}>✕</button>
      <div style={{ fontSize: 14, color: C.ink, fontWeight: 700, marginBottom: 6 }}>How did that go?</div>
      <p style={{ fontSize: 13, color: C.mist, marginBottom: 12, lineHeight: 1.5 }}>
        Your feedback helps others decide to try BIOS Check. It takes under a minute, and naming yourself is optional.
      </p>
      <button onClick={go} style={{
        padding: "8px 16px", borderRadius: 9, border: "none", fontWeight: 700, fontSize: 13,
        background: `linear-gradient(135deg, ${C.teal}, ${C.tealDark})`, color: "#fff", marginRight: 8,
      }}>Leave feedback</button>
      <button onClick={close} style={{
        padding: "8px 12px", borderRadius: 9, background: "transparent", color: C.mist,
        border: `1px solid ${C.ghost}`, fontSize: 13,
      }}>Not now</button>
    </div>
  );
}

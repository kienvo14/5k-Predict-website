import { useEffect, useState } from "react";
import { Routes, Route, NavLink, useNavigate } from "react-router-dom";
import { api, getToken, setToken } from "./api";
import Auth from "./Auth";
import History, { HistoryRow } from "./History";
import ProgressPage from "./ProgressPage";
import Models from "./Models";
import { paceToStr, paceToDec, kmToMi, miToKm, paceKmToMi, paceMiToKm, paceKmToMiStr, kmToMiStr } from "./format";

type Prediction = {
  id?: number;
  predicted_time: string;
  range_low: string;
  range_high: string;
  note: string;
  model?: string;
  model_note?: string;
  detected?: { runs_used: number; typical_pace: number; avg_hr: number; longest_km: number };
};
type FeedbackResult = { predicted_time: string; actual_time: string; diff_seconds: number; verdict: string };

export default function App() {
  const navigate = useNavigate();

  // auth
  const [user, setUser] = useState<string | null>(null);
  const [showAuth, setShowAuth] = useState(false);
  const [pendingClaimId, setPendingClaimId] = useState<number | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // predict inputs
  const [mode, setMode] = useState<"manual" | "upload">("manual");
  const [gender, setGender] = useState("male");
  const [pace, setPace] = useState("");
  const [easyHr, setEasyHr] = useState("");
  const [maxHr, setMaxHr] = useState("");
  const [longest, setLongest] = useState("");
  const [weeks, setWeeks] = useState<string[]>(["", "", ""]);
  const [uploadGender, setUploadGender] = useState("male");
  const [file, setFile] = useState<File | null>(null);
  const [uploadModel, setUploadModel] = useState<"linear" | "pytorch">("linear");

  // results
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [pr, setPr] = useState("");
  const [feedback, setFeedback] = useState<FeedbackResult | null>(null);
  const [fbError, setFbError] = useState("");

  // Pre-fill the Predict form with the user's most recent prediction.
  const prefillLatest = async () => {
    const d = await api.history();
    if (d.history && d.history.length) {
      const row = d.history[0];
      setMode("manual");
      setGender(row.gender ?? "male");
      // stored values are metric (min/km, km) -> show in the form as min/mile, miles
      setPace(row.typical_pace != null ? paceToStr(paceKmToMi(row.typical_pace)) : "");
      setEasyHr(row.easy_hr != null ? String(row.easy_hr) : "");
      setMaxHr(row.max_hr != null ? String(row.max_hr) : "");
      setLongest(row.longest_km != null ? kmToMi(row.longest_km).toFixed(1) : "");
      if (row.weekly_km && row.weekly_km.length)
        setWeeks(row.weekly_km.map((km: number) => kmToMi(km).toFixed(1)));
    }
  };

  useEffect(() => {
    if (getToken())
      api.me().then((d) => {
        if (d.user) {
          setUser(d.user);
          prefillLatest(); // start from their most recent prediction
        }
      });
  }, []);

  const logout = async () => {
    await api.logout();
    setToken(null);
    setUser(null);
    navigate("/");
  };

  const onAuthed = async (username: string) => {
    setUser(username);
    setShowAuth(false);
    if (pendingClaimId) {
      await api.claim(pendingClaimId);
      setPendingClaimId(null);
    } else {
      prefillLatest(); // no pending claim -> start from their most recent prediction
    }
    setRefreshKey((k) => k + 1);
  };

  const setWeek = (i: number, v: string) => {
    const next = [...weeks];
    next[i] = v;
    setWeeks(next);
  };

  const resetResult = () => {
    setError("");
    setResult(null);
    setLoading(true);
    setFeedback(null);
    setFbError("");
    setPr("");
  };

  const submitManual = async () => {
    resetResult();
    // UI is in miles + min/mile; convert to the km + min/km the model expects.
    const data = await api.predict({
      gender,
      typical_pace: paceMiToKm(paceToDec(pace)),
      easy_hr: parseFloat(easyHr),
      max_hr: parseFloat(maxHr),
      longest_run_km: miToKm(parseFloat(longest)),
      weekly_mileage_km: weeks
        .map((w) => parseFloat(w))
        .filter((w) => !isNaN(w) && w > 0)
        .map((mi) => miToKm(mi)),
    });
    setLoading(false);
    data.error ? setError(data.error) : (setResult(data), setRefreshKey((k) => k + 1));
  };

  const submitUpload = async () => {
    if (!file) return setError("Please choose your Strava activities file.");
    resetResult();
    const form = new FormData();
    form.append("gender", uploadGender);
    form.append("file", file);
    form.append("model", uploadModel);
    const data = await api.predictFile(form);
    setLoading(false);
    data.error ? setError(data.error) : (setResult(data), setRefreshKey((k) => k + 1));
  };

  const submitFeedback = async () => {
    if (!result?.id) return;
    setFbError("");
    const data = await api.feedback(result.id, pr);
    data.error ? setFbError(data.error) : (setFeedback(data), setRefreshKey((k) => k + 1));
  };

  const askLoginToSave = () => {
    if (result?.id) setPendingClaimId(result.id);
    setShowAuth(true);
  };

  // Click a history item -> reload its inputs into the form, add a blank week, go to Predict.
  const loadPrediction = (row: HistoryRow) => {
    setMode("manual");
    setGender(row.gender ?? "male");
    // stored metric -> shown as min/mile, miles
    setPace(row.typical_pace != null ? paceToStr(paceKmToMi(row.typical_pace)) : "");
    setEasyHr(row.easy_hr != null ? String(row.easy_hr) : "");
    setMaxHr(row.max_hr != null ? String(row.max_hr) : "");
    setLongest(row.longest_km != null ? kmToMi(row.longest_km).toFixed(1) : "");
    setWeeks([...(row.weekly_km ?? []).map((km) => kmToMi(km).toFixed(1)), ""]);
    setResult(null);
    setError("");
    setFeedback(null);
    navigate("/");
  };

  const navClass = ({ isActive }: { isActive: boolean }) => (isActive ? "nav-tab active" : "nav-tab");

  const predictView = (
    <div className="card">
      <div className="tabs">
        <button className={mode === "manual" ? "tab active" : "tab"} onClick={() => setMode("manual")}>
          Enter manually
        </button>
        <button className={mode === "upload" ? "tab active" : "tab"} onClick={() => setMode("upload")}>
          Upload Strava export
        </button>
      </div>

      {mode === "manual" ? (
        <div className="form">
          <div className="cols">
            <div className="col">
              <div className="grid">
                <div className="field">
                  <label>Gender</label>
                  <select value={gender} onChange={(e) => setGender(e.target.value)}>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                  </select>
                </div>
                <div className="field">
                  <label>Typical pace (mm:ss / mile)</label>
                  <input value={pace} onChange={(e) => setPace(e.target.value)} placeholder="8:50" />
                </div>
                <div className="field">
                  <label>Easy-run avg HR</label>
                  <input value={easyHr} onChange={(e) => setEasyHr(e.target.value)} placeholder="145" />
                </div>
                <div className="field">
                  <label>Max HR seen</label>
                  <input value={maxHr} onChange={(e) => setMaxHr(e.target.value)} placeholder="185" />
                </div>
                <div className="field wide">
                  <label>Longest recent run (miles)</label>
                  <input value={longest} onChange={(e) => setLongest(e.target.value)} placeholder="9" />
                </div>
              </div>
            </div>

            <div className="col">
              <label className="section-label">Weekly mileage (miles per week)</label>
              <div className="weeks">
                {weeks.map((w, i) => (
                  <div className="week-row" key={i}>
                    <span className="week-num">Week {i + 1}</span>
                    <input value={w} onChange={(e) => setWeek(i, e.target.value)} placeholder="25" />
                    {weeks.length > 1 && (
                      <button className="rm" onClick={() => setWeeks(weeks.filter((_, j) => j !== i))} type="button">×</button>
                    )}
                  </div>
                ))}
              </div>
              {weeks.length < 16 && (
                <button className="add" onClick={() => setWeeks([...weeks, ""])} type="button">+ Add week</button>
              )}
            </div>
          </div>

          <button className="predict" onClick={submitManual} disabled={loading}>
            {loading ? "Predicting…" : "Predict my 5K"}
          </button>
        </div>
      ) : (
        <div className="form">
          <p className="upload-help">
            Export from Strava → Settings → “Download or Delete Your Account” → Request Archive.
            Upload the <code>activities.csv</code> — we use your <strong>most recent 16 weeks</strong> of runs.
          </p>
          <div className="field">
            <label>Gender</label>
            <select value={uploadGender} onChange={(e) => setUploadGender(e.target.value)}>
              <option value="male">Male</option>
              <option value="female">Female</option>
            </select>
          </div>
          <label className="section-label">Strava activities file (.csv or .xlsx)</label>
          <label className="file-drop">
            <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <span>{file ? `📄 ${file.name}` : "Click to choose your activities file"}</span>
          </label>

          <label className="section-label">Model</label>
          <div className="modelpick">
            <button
              type="button"
              className={uploadModel === "linear" ? "mp on" : "mp"}
              onClick={() => setUploadModel("linear")}
            >
              <b>Fast</b><span>LinearRegression · instant</span>
            </button>
            <button
              type="button"
              className={uploadModel === "pytorch" ? "mp on" : "mp"}
              onClick={() => setUploadModel("pytorch")}
            >
              <b>Better</b><span>PyTorch · lower error</span>
            </button>
          </div>
          {uploadModel === "pytorch" && (
            <div className="coldstart">
              ⏳ The PyTorch model loads <code>torch</code> on first use, so the very first
              prediction can take ~10–20s (cold start). It's more accurate but runs on your
              per-run data — later requests are fast.
            </div>
          )}

          <button className="predict" onClick={submitUpload} disabled={loading}>
            {loading ? "Analyzing…" : "Predict from my Strava data"}
          </button>
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="result">
          <div className="result-label">
            Estimated 5K
            {result.model && (
              <span className={"modelbadge " + result.model}>
                {result.model === "pytorch" ? "PyTorch" : "LinearRegression"}
              </span>
            )}
          </div>
          <div className="big">{result.predicted_time}</div>
          <div className="range">likely {result.range_low} – {result.range_high}</div>
          {result.model_note && <div className="note warn">{result.model_note}</div>}
          {result.detected && (
            <div className="detected">
              From {result.detected.runs_used} runs · typical pace {paceKmToMiStr(result.detected.typical_pace)}/mi ·
              avg HR {result.detected.avg_hr} · longest {kmToMiStr(result.detected.longest_km)} mi
            </div>
          )}
          <div className="note">{result.note}</div>

          {!feedback ? (
            <div className="feedback">
              <label>Know your real 5K PR? See how close we got:</label>
              <div className="fb-row">
                <input value={pr} onChange={(e) => setPr(e.target.value)} placeholder="22:30" />
                <button onClick={submitFeedback} type="button">Check</button>
              </div>
              {fbError && <div className="fb-error">{fbError}</div>}
            </div>
          ) : (
            <div className="feedback done">
              <div className="fb-verdict">{feedback.verdict}</div>
              <div className="fb-detail">
                Predicted {feedback.predicted_time} · your PR {feedback.actual_time} · off by {feedback.diff_seconds}s
              </div>
            </div>
          )}

          <div className="save-row">
            {user ? (
              <span className="saved">✓ Saved to your history</span>
            ) : (
              <button className="save-btn" onClick={askLoginToSave}>Log in to save this to your history</button>
            )}
          </div>
        </div>
      )}
    </div>
  );

  return (
    <div className="page">
      <div className="topbar">
        <div className="brand">🏃 5K Predictor</div>
        <div className="auth-actions">
          {user ? (
            <>
              <span className="who">{user}</span>
              <button className="ghost" onClick={logout}>Log out</button>
            </>
          ) : (
            <button className="ghost" onClick={() => setShowAuth(true)}>Log in / Sign up</button>
          )}
        </div>
      </div>

      <header className="hero">
        <div className="hero-mark">
          <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="38" fill="none" stroke="#fc4c02" strokeWidth="2" opacity="0.9" />
            <circle cx="50" cy="12" r="5.5" fill="#fc4c02" />
          </svg>
        </div>
        <div className="badge">TRAINED ON 104,000+ RUNS</div>
        <h1>How fast is your 5K right now?</h1>
        <p>A regression model estimates your current 5K from your training — pace, mileage, and heart rate.</p>
      </header>

      <div className="nav">
        <NavLink to="/" end className={navClass}>Predict</NavLink>
        <NavLink to="/history" className={navClass}>History</NavLink>
        <NavLink to="/progress" className={navClass}>Progress</NavLink>
        <NavLink to="/models" className={navClass}>Models</NavLink>
      </div>

      <Routes>
        <Route path="/" element={predictView} />
        <Route
          path="/history"
          element={
            <div className="card">
              <History loggedIn={!!user} refreshKey={refreshKey} onSelect={loadPrediction} />
            </div>
          }
        />
        <Route
          path="/progress"
          element={
            <div className="card">
              <ProgressPage loggedIn={!!user} refreshKey={refreshKey} />
            </div>
          }
        />
        <Route path="/models" element={<div className="card"><Models /></div>} />
      </Routes>

      <div className="stats">
        <div className="stat"><b>104,000+</b><span>clean runs</span></div>
        <div className="stat"><b>700,000</b><span>miles</span></div>
        <div className="stat"><b>±66s</b><span>accuracy</span></div>
        <div className="stat"><b>PostgreSQL</b><span>storage</span></div>
      </div>

      <footer className="foot">React + TypeScript · FastAPI · PyTorch · PostgreSQL · GCP — CV MAE ≈ 66s</footer>

      {showAuth && <Auth onClose={() => setShowAuth(false)} onAuthed={onAuthed} />}
    </div>
  );
}

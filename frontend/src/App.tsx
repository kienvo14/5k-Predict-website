import { useState } from "react";

const API = "http://localhost:8000";

type Prediction = {
  predicted_time: string;
  range_low: string;
  range_high: string;
  note: string;
  weeks_used?: number;
  detected?: {
    runs_used: number;
    typical_pace: number;
    avg_hr: number;
    longest_km: number;
  };
};

type Mode = "manual" | "upload";

export default function App() {
  const [mode, setMode] = useState<Mode>("manual");

  // shared result state
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // manual inputs
  const [gender, setGender] = useState("male");
  const [pace, setPace] = useState("");
  const [easyHr, setEasyHr] = useState("");
  const [maxHr, setMaxHr] = useState("");
  const [longest, setLongest] = useState("");
  const [weeks, setWeeks] = useState<string[]>(["", "", ""]);

  // upload inputs
  const [uploadGender, setUploadGender] = useState("male");
  const [file, setFile] = useState<File | null>(null);

  const setWeek = (i: number, v: string) => {
    const next = [...weeks];
    next[i] = v;
    setWeeks(next);
  };
  const addWeek = () => weeks.length < 16 && setWeeks([...weeks, ""]);
  const removeWeek = (i: number) => setWeeks(weeks.filter((_, j) => j !== i));

  const reset = () => {
    setError("");
    setResult(null);
    setLoading(true);
  };

  const submitManual = async () => {
    reset();
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          gender,
          typical_pace: parseFloat(pace),
          easy_hr: parseFloat(easyHr),
          max_hr: parseFloat(maxHr),
          longest_run_km: parseFloat(longest),
          weekly_mileage_km: weeks
            .map((w) => parseFloat(w))
            .filter((w) => !isNaN(w) && w > 0),
        }),
      });
      const data = await res.json();
      data.error ? setError(data.error) : setResult(data);
    } catch {
      setError("Could not reach the backend. Is it running on :8000?");
    }
    setLoading(false);
  };

  const submitUpload = async () => {
    if (!file) {
      setError("Please choose your Strava activities.csv file.");
      return;
    }
    reset();
    try {
      const form = new FormData();
      form.append("gender", uploadGender);
      form.append("file", file);
      const res = await fetch(`${API}/predict-from-file`, {
        method: "POST",
        body: form,
      });
      const data = await res.json();
      data.error ? setError(data.error) : setResult(data);
    } catch {
      setError("Could not reach the backend. Is it running on :8000?");
    }
    setLoading(false);
  };

  return (
    <div className="page">
      <header className="hero">
        <div className="badge">🏃 5K PREDICTOR</div>
        <h1>How fast is your 5K right now?</h1>
        <p>
          A regression model trained on <strong>117,000+ runs</strong> estimates
          your current 5K from your training — pace, mileage, and heart rate.
        </p>
      </header>

      <div className="card">
        <div className="tabs">
          <button
            className={mode === "manual" ? "tab active" : "tab"}
            onClick={() => setMode("manual")}
          >
            Enter manually
          </button>
          <button
            className={mode === "upload" ? "tab active" : "tab"}
            onClick={() => setMode("upload")}
          >
            Upload Strava export
          </button>
        </div>

        {mode === "manual" ? (
          <div className="form">
            <div className="grid">
              <div className="field">
                <label>Gender</label>
                <select value={gender} onChange={(e) => setGender(e.target.value)}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
              <div className="field">
                <label>Typical pace (min/km)</label>
                <input value={pace} onChange={(e) => setPace(e.target.value)} placeholder="5.5 = 5:30" />
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
                <label>Longest recent run (km)</label>
                <input value={longest} onChange={(e) => setLongest(e.target.value)} placeholder="15" />
              </div>
            </div>

            <label className="section-label">Weekly mileage (km per week)</label>
            <div className="weeks">
              {weeks.map((w, i) => (
                <div className="week-row" key={i}>
                  <span className="week-num">Wk {i + 1}</span>
                  <input value={w} onChange={(e) => setWeek(i, e.target.value)} placeholder="40" />
                  {weeks.length > 1 && (
                    <button className="rm" onClick={() => removeWeek(i)} type="button">×</button>
                  )}
                </div>
              ))}
            </div>
            {weeks.length < 16 && (
              <button className="add" onClick={addWeek} type="button">+ Add week</button>
            )}

            <button className="predict" onClick={submitManual} disabled={loading}>
              {loading ? "Predicting…" : "Predict my 5K"}
            </button>
          </div>
        ) : (
          <div className="form">
            <p className="upload-help">
              Export your data from Strava → Settings → “Download or Delete Your Account” →
              Request Archive. Upload the <code>activities.csv</code> — we’ll use your{" "}
              <strong>most recent 16 weeks</strong> of runs.
            </p>
            <div className="field">
              <label>Gender</label>
              <select value={uploadGender} onChange={(e) => setUploadGender(e.target.value)}>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
            <label className="section-label">Strava activities.csv</label>
            <label className="file-drop">
              <input
                type="file"
                accept=".csv"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span>{file ? `📄 ${file.name}` : "Click to choose your activities.csv"}</span>
            </label>

            <button className="predict" onClick={submitUpload} disabled={loading}>
              {loading ? "Analyzing…" : "Predict from my Strava data"}
            </button>
          </div>
        )}

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <div className="result-label">Estimated 5K</div>
            <div className="big">{result.predicted_time}</div>
            <div className="range">
              likely {result.range_low} – {result.range_high}
            </div>
            {result.detected && (
              <div className="detected">
                From {result.detected.runs_used} runs · typical pace{" "}
                {result.detected.typical_pace} min/km · avg HR {result.detected.avg_hr} ·
                longest {result.detected.longest_km} km
              </div>
            )}
            <div className="note">{result.note}</div>
          </div>
        )}
      </div>

      <footer className="foot">
        React + TypeScript · FastAPI · scikit-learn — CV MAE ≈ 82s
      </footer>
    </div>
  );
}

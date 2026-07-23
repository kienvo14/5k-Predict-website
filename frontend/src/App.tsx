import { useState } from "react";

// Where the FastAPI backend is running.
const API = "http://localhost:8000";

// Shape of the successful response from POST /predict.
type Prediction = {
  predicted_time: string;
  range_low: string;
  range_high: string;
  note: string;
};

export default function App() {
  // --- form state ---
  const [gender, setGender] = useState("male");
  const [pace, setPace] = useState("");
  const [easyHr, setEasyHr] = useState("");
  const [maxHr, setMaxHr] = useState("");
  const [longest, setLongest] = useState("");
  const [weeks, setWeeks] = useState<string[]>(["", "", ""]); // start with 3 week slots

  // --- result / status state ---
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // --- weekly-mileage list helpers ---
  const setWeek = (i: number, value: string) => {
    const next = [...weeks];
    next[i] = value;
    setWeeks(next);
  };
  const addWeek = () => weeks.length < 16 && setWeeks([...weeks, ""]);
  const removeWeek = (i: number) => setWeeks(weeks.filter((_, j) => j !== i));

  // --- submit to backend ---
  const submit = async () => {
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const body = {
        gender,
        typical_pace: parseFloat(pace),
        easy_hr: parseFloat(easyHr),
        max_hr: parseFloat(maxHr),
        longest_run_km: parseFloat(longest),
        weekly_mileage_km: weeks
          .map((w) => parseFloat(w))
          .filter((w) => !isNaN(w) && w > 0),
      };
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (data.error) setError(data.error);
      else setResult(data);
    } catch {
      setError("Could not reach the backend. Is it running on :8000?");
    }
    setLoading(false);
  };

  return (
    <div className="page">
      <div className="card">
        <h1>5K Time Predictor</h1>
        <p className="sub">
          Enter your recent training and get an estimate of your current 5K.
        </p>

        <label>Gender</label>
        <select value={gender} onChange={(e) => setGender(e.target.value)}>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>

        <label>Typical training pace (min/km, e.g. 5.5 = 5:30)</label>
        <input value={pace} onChange={(e) => setPace(e.target.value)} placeholder="5.5" />

        <label>Average heart rate on easy runs (bpm)</label>
        <input value={easyHr} onChange={(e) => setEasyHr(e.target.value)} placeholder="145" />

        <label>Highest heart rate you've seen (bpm)</label>
        <input value={maxHr} onChange={(e) => setMaxHr(e.target.value)} placeholder="185" />

        <label>Longest recent run (km)</label>
        <input value={longest} onChange={(e) => setLongest(e.target.value)} placeholder="15" />

        <label>Weekly mileage (km per week &mdash; add a box per week)</label>
        <div className="weeks">
          {weeks.map((w, i) => (
            <div className="week-row" key={i}>
              <span className="week-num">Wk {i + 1}</span>
              <input
                value={w}
                onChange={(e) => setWeek(i, e.target.value)}
                placeholder="40"
              />
              {weeks.length > 1 && (
                <button className="rm" onClick={() => removeWeek(i)} type="button">
                  &times;
                </button>
              )}
            </div>
          ))}
        </div>
        {weeks.length < 16 && (
          <button className="add" onClick={addWeek} type="button">
            + Add week
          </button>
        )}

        <button className="predict" onClick={submit} disabled={loading}>
          {loading ? "Predicting..." : "Predict my 5K"}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <div className="big">{result.predicted_time}</div>
            <div className="range">
              likely {result.range_low} &ndash; {result.range_high}
            </div>
            <div className="note">{result.note}</div>
          </div>
        )}
      </div>
    </div>
  );
}

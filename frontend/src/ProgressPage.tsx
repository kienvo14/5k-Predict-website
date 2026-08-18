import { useEffect, useState } from "react";
import { api } from "./api";
import { paceToStr, paceToDec, paceMiToKm, miToKm, kmToMi, kmToMiStr, paceKmToMiStr } from "./format";

type Run = { id: number; date: string; dist_km: number; pace: number | null; hr: number | null };
type Week = {
  idx: number;
  week_key: string;      // stable "year-week" id used by /add-run
  year: number;
  week: number;
  label: string;
  mileage_km: number;
  num_runs: number;
  avg_pace: number | null;
  avg_hr: number | null;
  runs: Run[];
};

// One row of the runs table — inline-edit HR and delete.
function RunRow({ run, onChanged }: { run: Run; onChanged: () => void }) {
  const [editing, setEditing] = useState(false);
  const [hr, setHr] = useState(run.hr != null ? String(run.hr) : "");
  const save = async () => {
    const val = hr.trim() === "" ? null : parseFloat(hr);
    const d = await api.editRun(run.id, val);
    if (!d.error) {
      setEditing(false);
      onChanged();
    }
  };
  const del = async () => {
    if (!confirm("Delete this run?")) return;
    const d = await api.deleteRun(run.id);
    if (!d.error) onChanged();
  };
  return (
    <div className="wd-run">
      <span className="wd-date">{run.date}</span>
      <span>{kmToMiStr(run.dist_km)} mi</span>
      <span>{paceKmToMiStr(run.pace)}/mi</span>
      {editing ? (
        <input
          className="hr-input"
          value={hr}
          onChange={(e) => setHr(e.target.value)}
          placeholder="bpm"
          autoFocus
          onBlur={save}
          onKeyDown={(e) => e.key === "Enter" && save()}
        />
      ) : (
        <span className="hr-cell" onClick={() => setEditing(true)}>
          {run.hr ?? "–"} bpm ✏️
        </span>
      )}
      <button className="run-del" onClick={del} type="button" title="Delete run">🗑</button>
    </div>
  );
}

export default function ProgressPage({ loggedIn, refreshKey }: { loggedIn: boolean; refreshKey: number }) {
  const [weeks, setWeeks] = useState<Week[]>([]);
  const [msg, setMsg] = useState("");
  const [sel, setSel] = useState<number | null>(null);

  // add-run form (for manual weeks)
  const [rDist, setRDist] = useState("");
  const [rPace, setRPace] = useState("");
  const [rDate, setRDate] = useState("");
  const [rHr, setRHr] = useState("");
  const [rErr, setRErr] = useState("");

  const load = (keepSel = false) => {
    api.progress().then((d) => {
      if (d.error) {
        setMsg(d.error);
        setWeeks([]);
      } else if (!d.weeks || !d.weeks.length) {
        setMsg("Upload your Strava export on the Predict page — your weekly progress will appear here.");
        setWeeks([]);
      } else {
        setWeeks(d.weeks);
        setMsg("");
        if (!keepSel) setSel(d.weeks.length - 1);
      }
    });
  };

  useEffect(() => {
    if (!loggedIn) {
      setWeeks([]);
      setMsg("Log in and upload your Strava data to see your progress.");
      return;
    }
    load();
  }, [loggedIn, refreshKey]);

  if (msg) return <div className="empty">{msg}</div>;

  const selWeek = sel != null ? weeks[sel] : null;

  const addRun = async () => {
    if (!selWeek) return;
    setRErr("");
    // form is miles + min/mile; backend stores km + min/km
    const d = await api.addRun(
      selWeek.week_key,
      miToKm(parseFloat(rDist)),
      rPace.trim() ? paceToStr(paceMiToKm(paceToDec(rPace))) : rPace,
      rDate || undefined,
      rHr ? parseFloat(rHr) : undefined
    );
    if (d.error) {
      setRErr(d.error);
      return;
    }
    setRDist("");
    setRPace("");
    setRDate("");
    setRHr("");
    load(true); // reload, keep the same week selected
  };

  // ----- chart geometry -----
  const n = weeks.length;
  const W = 720, H = 240, padL = 12, padR = 52, padT = 28, padB = 14;
  const cw = W - padL - padR;
  const ch = H - padT - padB;
  // chart is drawn in miles
  const mi = (w: Week) => kmToMi(w.mileage_km);
  const maxM = Math.max(...weeks.map(mi), 1);
  const niceMax = Math.max(20, Math.ceil(maxM / 10) * 10);
  const xAt = (i: number) => padL + (n <= 1 ? cw / 2 : (i / (n - 1)) * cw);
  const yAt = (m: number) => padT + ch - (m / niceMax) * ch;
  const pts = weeks.map((w, i) => [xAt(i), yAt(mi(w))] as const);
  const line = pts.map((p, i) => (i === 0 ? "M" : "L") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
  const area = `${line} L ${xAt(n - 1).toFixed(1)} ${(padT + ch).toFixed(1)} L ${xAt(0).toFixed(1)} ${(padT + ch).toFixed(1)} Z`;
  const grid = [0, niceMax / 2, niceMax];

  return (
    <div className="progress">
      <div className="chart-title">Weekly mileage · last {n} weeks</div>

      <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`}>
        {grid.map((g, i) => (
          <g key={i}>
            <line x1={padL} x2={padL + cw} y1={yAt(g)} y2={yAt(g)} stroke="#242424" strokeWidth="1" />
            <text x={padL + cw + 10} y={yAt(g) + 4} fill="#8a8a8a" fontSize="13">{g} mi</text>
          </g>
        ))}
        <path d={area} fill="rgba(252,76,2,0.14)" />
        <path d={line} fill="none" stroke="#fc4c02" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        {selWeek && sel != null && (
          <>
            <line x1={xAt(sel)} x2={xAt(sel)} y1={padT} y2={padT + ch} stroke="#fff" strokeWidth="1.5" opacity="0.45" />
            <text x={xAt(sel)} y={yAt(mi(weeks[sel])) - 13} fill="#fff" fontSize="14" fontWeight="700" textAnchor="middle">
              {kmToMiStr(weeks[sel].mileage_km)} mi
            </text>
          </>
        )}
        {pts.map((p, i) => (
          <g key={i} onClick={() => setSel(i)} style={{ cursor: "pointer" }}>
            <circle cx={p[0]} cy={p[1]} r={13} fill="transparent" />
            <circle cx={p[0]} cy={p[1]} r={i === sel ? 6 : 5} fill={i === sel ? "#fc4c02" : "#000"} stroke="#fc4c02" strokeWidth={2} />
          </g>
        ))}
      </svg>

      {selWeek && (
        <div className="week-detail">
          <div className="wd-title">{selWeek.label} · {kmToMiStr(selWeek.mileage_km)} mi</div>
          <div className="wd-sub">
            {selWeek.num_runs} runs · {paceKmToMiStr(selWeek.avg_pace)}/mi · {selWeek.avg_hr ?? "–"} bpm
          </div>

          {selWeek.runs.length > 0 && (
            <div className="wd-runs">
              <div className="wd-run wd-hdr">
                <span>date</span><span>distance</span><span>pace</span><span>HR</span><span></span>
              </div>
              {selWeek.runs.map((r) => (
                <RunRow key={r.id} run={r} onChanged={() => load(true)} />
              ))}
            </div>
          )}

          {/* manual weeks: add runs to build up an avg pace */}
          {selWeek.num_runs === 0 && (
            <div className="wd-manual">Manually entered week — add runs below to give it a pace.</div>
          )}
          <div className="add-run">
            <input type="date" value={rDate} onChange={(e) => setRDate(e.target.value)} />
            <input value={rDist} onChange={(e) => setRDist(e.target.value)} placeholder="distance mi" />
            <input value={rPace} onChange={(e) => setRPace(e.target.value)} placeholder="pace 8:50" />
            <input value={rHr} onChange={(e) => setRHr(e.target.value)} placeholder="HR" />
            <button onClick={addRun} type="button">Add run</button>
          </div>
          {rErr && <div className="fb-error">{rErr}</div>}
        </div>
      )}
    </div>
  );
}

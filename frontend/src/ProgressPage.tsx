import { useEffect, useState } from "react";
import { api } from "./api";
import { paceToStr } from "./format";

type Run = { date: string; dist_km: number; pace: number | null; hr: number | null };
type Week = {
  idx: number;
  year: number;
  week: number;
  label: string;
  mileage_km: number;
  num_runs: number;
  avg_pace: number | null;
  avg_hr: number | null;
  runs: Run[];
};

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
    const d = await api.addRun(
      selWeek.idx,
      parseFloat(rDist),
      rPace,
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
  const maxM = Math.max(...weeks.map((w) => w.mileage_km), 1);
  const niceMax = Math.max(30, Math.ceil(maxM / 30) * 30);
  const xAt = (i: number) => padL + (n <= 1 ? cw / 2 : (i / (n - 1)) * cw);
  const yAt = (m: number) => padT + ch - (m / niceMax) * ch;
  const pts = weeks.map((w, i) => [xAt(i), yAt(w.mileage_km)] as const);
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
            <text x={padL + cw + 10} y={yAt(g) + 4} fill="#8a8a8a" fontSize="13">{g} km</text>
          </g>
        ))}
        <path d={area} fill="rgba(252,76,2,0.14)" />
        <path d={line} fill="none" stroke="#fc4c02" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        {selWeek && sel != null && (
          <>
            <line x1={xAt(sel)} x2={xAt(sel)} y1={padT} y2={padT + ch} stroke="#fff" strokeWidth="1.5" opacity="0.45" />
            <text x={xAt(sel)} y={yAt(weeks[sel].mileage_km) - 13} fill="#fff" fontSize="14" fontWeight="700" textAnchor="middle">
              {weeks[sel].mileage_km} km
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
          <div className="wd-title">{selWeek.label} · {selWeek.mileage_km} km</div>
          <div className="wd-sub">
            {selWeek.num_runs} runs · {paceToStr(selWeek.avg_pace)}/km · {selWeek.avg_hr ?? "–"} bpm
          </div>

          {selWeek.runs.length > 0 && (
            <div className="wd-runs">
              <div className="wd-run wd-hdr">
                <span>date</span><span>distance</span><span>pace</span><span>HR</span>
              </div>
              {selWeek.runs.map((r, j) => (
                <div className="wd-run" key={j}>
                  <span className="wd-date">{r.date}</span>
                  <span>{r.dist_km} km</span>
                  <span>{paceToStr(r.pace)}/km</span>
                  <span>{r.hr ?? "–"} bpm</span>
                </div>
              ))}
            </div>
          )}

          {/* manual weeks: add runs to build up an avg pace */}
          {selWeek.num_runs === 0 && (
            <div className="wd-manual">Manually entered week — add runs below to give it a pace.</div>
          )}
          <div className="add-run">
            <input type="date" value={rDate} onChange={(e) => setRDate(e.target.value)} />
            <input value={rDist} onChange={(e) => setRDist(e.target.value)} placeholder="distance km" />
            <input value={rPace} onChange={(e) => setRPace(e.target.value)} placeholder="pace 5:30" />
            <input value={rHr} onChange={(e) => setRHr(e.target.value)} placeholder="HR" />
            <button onClick={addRun} type="button">Add run</button>
          </div>
          {rErr && <div className="fb-error">{rErr}</div>}
        </div>
      )}
    </div>
  );
}

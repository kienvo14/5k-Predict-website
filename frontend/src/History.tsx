import { useEffect, useState } from "react";
import { api } from "./api";
import { paceToStr } from "./format";

export type HistoryRow = {
  id: number;
  date: string;
  source: string;
  predicted_time: string;
  actual_pr_time: string | null;
  diff_seconds: number | null;
  gender: string;
  typical_pace: number;
  avg_weekly_km: number;
  active_weeks: number;
  longest_km: number;
  easy_hr: number;
  max_hr: number;
  weekly_km: number[];
};

// Shows the logged-in user's past predictions. Clicking one reloads it into
// the form (via onSelect) so the user can add a week and re-predict.
export default function History({
  loggedIn,
  refreshKey,
  onSelect,
}: {
  loggedIn: boolean;
  refreshKey: number;
  onSelect: (row: HistoryRow) => void;
}) {
  const [rows, setRows] = useState<HistoryRow[]>([]);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    if (!loggedIn) {
      setRows([]);
      setMsg("Log in to see your prediction history.");
      return;
    }
    api.history().then((d) => {
      if (d.error) {
        setMsg(d.error);
        setRows([]);
      } else {
        setRows(d.history);
        setMsg(d.history.length ? "" : "No predictions yet — make one on the Predict tab!");
      }
    });
  }, [loggedIn, refreshKey]);

  if (msg) return <div className="empty">{msg}</div>;

  return (
    <div className="history">
      <div className="history-hint">Click a prediction to reuse it, add a week, and re-predict →</div>
      {rows.map((r) => (
        <button className="hrow" key={r.id} onClick={() => onSelect(r)} type="button">
          <div className="hleft">
            <div className="hdate">{r.date}</div>
            <div className="hsrc">{r.source} · {r.active_weeks} wks · {paceToStr(r.typical_pace)}/km</div>
          </div>
          <div className="hmid">
            <div className="hlabel">predicted</div>
            <div className="hpred">{r.predicted_time}</div>
          </div>
          <div className="hright">
            {r.actual_pr_time ? (
              <>
                <div className="hlabel">your PR</div>
                <div className="hactual">{r.actual_pr_time}</div>
                <div className="hdiff">±{r.diff_seconds}s</div>
              </>
            ) : (
              <div className="hlabel">no PR yet</div>
            )}
          </div>
        </button>
      ))}
    </div>
  );
}

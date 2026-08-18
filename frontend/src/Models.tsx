import { useEffect, useState } from "react";
import { api } from "./api";

type Card = {
  key: string;
  name: string;
  family: string;
  deployed: boolean;
  mae_seconds: number;
  input: string;
  blurb: string;
};
type ModelsResp = { note: string; models: Card[]; pytorch_available: boolean };

// Read-only comparison of the models I trained, with their REAL held-out MAE.
// Lower MAE = better. The bar width is scaled so differences are visible.
export default function Models() {
  const [data, setData] = useState<ModelsResp | null>(null);

  useEffect(() => {
    api.models().then(setData).catch(() => setData(null));
  }, []);

  if (!data) return <div className="empty">Loading model comparison…</div>;

  const maxMae = Math.max(...data.models.map((m) => m.mae_seconds));
  const best = Math.min(...data.models.map((m) => m.mae_seconds));

  return (
    <div className="models">
      <h2>Model comparison</h2>
      <p className="sub">{data.note}</p>

      <div className="mtable">
        {data.models
          .slice()
          .sort((a, b) => a.mae_seconds - b.mae_seconds)
          .map((m) => (
            <div className={"mrow" + (m.deployed ? " deployed" : "")} key={m.key}>
              <div className="mhead">
                <span className="mname">{m.name}</span>
                <span className={"mfam " + m.family.toLowerCase().replace(/[^a-z]/g, "")}>{m.family}</span>
                {m.deployed && <span className="mtag live">● live</span>}
                {m.key === "pytorch" && (
                  <span className={"mtag " + (data.pytorch_available ? "ok" : "off")}>
                    {data.pytorch_available ? "torch ready" : "torch not loaded here"}
                  </span>
                )}
                {m.mae_seconds === best && <span className="mtag best">best MAE</span>}
              </div>
              <div className="mbarwrap">
                <div
                  className="mbar"
                  style={{ width: `${(m.mae_seconds / maxMae) * 100}%` }}
                />
                <span className="mmae">{m.mae_seconds.toFixed(1)}s MAE</span>
              </div>
              <div className="mblurb">{m.blurb}</div>
              <div className="minput">input: {m.input}</div>
            </div>
          ))}
      </div>

      <p className="mfoot">
        Held-out mean absolute error, in seconds, on the same 540-train / 134-validation
        athlete split. The PyTorch model wins on accuracy but needs rich per-run data and
        loads <code>torch</code> on demand — so LinearRegression is the default that serves
        every prediction instantly.
      </p>
    </div>
  );
}

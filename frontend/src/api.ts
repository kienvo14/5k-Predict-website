//Central place for talking to the backend + storing the login token.
const API = "https://fivek-backend.onrender.com";
//const API = "http://localhost:8000";

export function getToken(): string | null {
  return localStorage.getItem("token");
}
export function setToken(t: string | null) {
  if (t) localStorage.setItem("token", t);
  else localStorage.removeItem("token");
}

//Every request auto-attaches the token (if logged in).
async function req(path: string, opts: RequestInit = {}) {
  const token = getToken();
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API}${path}`, { ...opts, headers });
  return res.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  signup: (username: string, password: string) => req("/signup", json({ username, password })),
  login: (username: string, password: string) => req("/login", json({ username, password })),
  logout: () => req("/logout", { method: "POST" }),
  me: () => req("/me"),
  predict: (body: unknown) => req("/predict", json(body)),
  predictFile: (form: FormData) => req("/predict-from-file", { method: "POST", body: form }),
  feedback: (id: number, pr_time: string) => req("/feedback", json({ id, pr_time })),
  claim: (id: number) => req("/claim", json({ id })),
  history: () => req("/history"),
  progress: () => req("/progress"),
  addRun: (week_key: string, dist_km: number, pace: string, date?: string, hr?: number) =>
    req("/add-run", json({ week_key, dist_km, pace, date, hr })),
  editRun: (run_id: number, hr: number | null) => req(`/runs/${run_id}/edit`, json({ hr })),
  deleteRun: (run_id: number) => req(`/runs/${run_id}`, { method: "DELETE" }),
};

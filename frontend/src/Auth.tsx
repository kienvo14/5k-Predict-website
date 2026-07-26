import { useState } from "react";
import { api, setToken } from "./api";

// Login / signup modal. Calls onAuthed(username) once a token is stored.
export default function Auth({
  onClose,
  onAuthed,
}: {
  onClose: () => void;
  onAuthed: (username: string) => void;
}) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    setError("");
    setLoading(true);
    const data = mode === "login"
      ? await api.login(username, password)
      : await api.signup(username, password);
    setLoading(false);
    if (data.error) {
      setError(data.error);
      return;
    }
    setToken(data.token);
    onAuthed(data.username);
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="tabs">
          <button className={mode === "login" ? "tab active" : "tab"} onClick={() => setMode("login")}>
            Log in
          </button>
          <button className={mode === "signup" ? "tab active" : "tab"} onClick={() => setMode("signup")}>
            Sign up
          </button>
        </div>

        <div className="form">
          <label>Username</label>
          <input value={username} onChange={(e) => setUsername(e.target.value)} placeholder="runner123" />
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••"
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          {error && <div className="error">{error}</div>}
          <button className="predict" onClick={submit} disabled={loading}>
            {loading ? "…" : mode === "login" ? "Log in" : "Create account"}
          </button>
        </div>
      </div>
    </div>
  );
}

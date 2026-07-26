// Pace is stored/computed as decimal minutes/km (e.g. 5.25) but shown as MM:SS (5:15).

/** 5.25 -> "5:15". null/NaN -> "–". */
export function paceToStr(dec: number | null | undefined): string {
  if (dec == null || isNaN(dec)) return "–";
  let m = Math.floor(dec);
  let s = Math.round((dec - m) * 60);
  if (s === 60) {
    m += 1;
    s = 0;
  }
  return `${m}:${s.toString().padStart(2, "0")}`;
}

/** "5:15" -> 5.25. Also accepts plain decimals like "5.25". */
export function paceToDec(str: string): number {
  const t = (str || "").trim();
  if (t.includes(":")) {
    const [m, s] = t.split(":");
    return parseInt(m || "0", 10) + (parseFloat(s) || 0) / 60;
  }
  return parseFloat(t);
}

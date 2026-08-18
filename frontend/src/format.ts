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

// ---------------------------------------------------------------------------
// Units. The backend/model works entirely in kilometers (it was trained on
// metric data). The UI shows miles + min/mile. So we convert ONLY at the edges:
// km -> mi for display, mi -> km right before any API call.
// ---------------------------------------------------------------------------
export const KM_PER_MILE = 1.609344;

/** kilometers -> miles */
export const kmToMi = (km: number) => km / KM_PER_MILE;
/** miles -> kilometers */
export const miToKm = (mi: number) => mi * KM_PER_MILE;

/** pace in min/km -> pace in min/mile (a mile is longer, so the number grows) */
export const paceKmToMi = (perKm: number) => perKm * KM_PER_MILE;
/** pace in min/mile -> pace in min/km */
export const paceMiToKm = (perMi: number) => perMi / KM_PER_MILE;

/** A km-pace (as the backend stores it) shown as a MM:SS min/mile string. */
export function paceKmToMiStr(perKm: number | null | undefined): string {
  if (perKm == null || isNaN(perKm)) return "–";
  return paceToStr(paceKmToMi(perKm));
}

/** A distance in km shown as a miles number, 1 decimal (e.g. 24.9). */
export const kmToMiStr = (km: number | null | undefined) =>
  km == null || isNaN(km) ? "–" : kmToMi(km).toFixed(1);

/**
 * Shared HTTP-error → human message helper.
 *
 * Three response shapes matter here:
 *   - FastAPI HTTPException  -> { detail: "..." }
 *   - Pydantic 422 validation -> { detail: [{ loc: [...], msg: "..." }, ...] }
 *   - slowapi rate limiting   -> 429 + { error: "..." } (note: `error`, not
 *     `detail`) plus a Retry-After header when the proxy/CORS exposes it.
 *
 * Dumping raw JSON for the second and "HTTP 429" for the third is what users
 * used to see; both are now turned into something readable.
 */
export async function parseErrorDetail(res) {
  if (res.status === 429) {
    const retryAfter = Number(res.headers?.get?.("Retry-After"));
    if (Number.isFinite(retryAfter) && retryAfter > 0) {
      return `請求過於頻繁，請等待 ${Math.ceil(retryAfter)} 秒後再試`;
    }
    return "請求過於頻繁，請稍候再試";
  }
  try {
    const body = await res.json();
    const detail = body?.detail ?? body?.error;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map((e) => {
          const loc = Array.isArray(e?.loc)
            ? e.loc.filter((p) => p !== "body" && p !== "query").join(".")
            : "";
          const msg = e?.msg || "欄位格式錯誤";
          return loc ? `${loc}：${msg}` : msg;
        })
        .join("；");
    }
    if (detail) return JSON.stringify(detail);
    return `HTTP ${res.status}`;
  } catch {
    return `HTTP ${res.status}`;
  }
}

export default parseErrorDetail;

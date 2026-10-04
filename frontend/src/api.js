// Thin client for the FastAPI backend. All URLs are relative: Vite (dev) or nginx (Docker)
// proxy /api and /health to the backend container.
const BASE = import.meta.env.VITE_API_BASE || "";

async function handle(res) {
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

export const getHealth = () => fetch(`${BASE}/health`).then(handle);
export const getInfo = () => fetch(`${BASE}/api/info`).then(handle);
export const getSamples = () => fetch(`${BASE}/api/samples`).then(handle);
export const sampleUrl = (id) => `${BASE}/api/samples/${id}`;

/** Run one of the restoration endpoints.
 *  source: { file?: File, sampleId?: string }
 *  corruption: { type, severity, prob, kernel, sigma, coverage, numRects, seed } */
export function restore(endpoint, source, corruption) {
  const fd = new FormData();
  if (source.file) fd.append("file", source.file);
  else if (source.sampleId) fd.append("sample_id", source.sampleId);
  fd.append("corruption", corruption.type);
  fd.append("severity", corruption.severity);
  if (corruption.seed !== "" && corruption.seed != null) fd.append("seed", corruption.seed);
  if (corruption.severity === "custom") {
    if (corruption.type === "salt_pepper") fd.append("prob", corruption.prob);
    if (corruption.type === "blur") {
      fd.append("kernel", corruption.kernel);
      fd.append("sigma", corruption.sigma);
    }
    if (corruption.type === "occlusion") {
      fd.append("coverage", corruption.coverage);
      fd.append("num_rects", corruption.numRects);
    }
  }
  return fetch(`${BASE}/api/${endpoint}`, { method: "POST", body: fd }).then(handle);
}

export function faceToSketch(file, style) {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("style", style);
  return fetch(`${BASE}/api/face-to-sketch`, { method: "POST", body: fd }).then(handle);
}

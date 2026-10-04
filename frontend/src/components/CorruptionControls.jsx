import { Shuffle } from "lucide-react";
import { Segmented } from "./ui.jsx";

export const DEFAULT_CORRUPTION = {
  type: "blur", severity: "medium", prob: 0.08, kernel: 5, sigma: 1.5, coverage: 0.2, numRects: 2, seed: "42",
};

// Presets are the fixed test levels from the assignment (PDF p.3).
const PRESETS = {
  salt_pepper: { low: "p = 0.03", medium: "p = 0.08", high: "p = 0.15" },
  blur: { low: "k = 3, σ = 0.7", medium: "k = 5, σ = 1.5", high: "k = 7, σ = 2.5" },
  occlusion: { low: "≈10 %, 1 rectangle", medium: "≈20 %, 2 rectangles", high: "≈35 %, 3 rectangles" },
};

const TYPES = [
  { value: "none", label: "None (as uploaded)", dot: "bg-slate-300" },
  { value: "salt_pepper", label: "Salt & Pepper", dot: "bg-amber-400" },
  { value: "blur", label: "Gaussian Blur", dot: "bg-sky-600" },
  { value: "occlusion", label: "Occlusion Mask", dot: "bg-rose-400" },
];
export const TYPE_LABEL = Object.fromEntries(TYPES.map((t) => [t.value, t.label]));

function Slider({ label, value, min, max, step, onChange, format = (v) => v }) {
  return (
    <label className="block">
      <div className="flex justify-between text-[15px] text-slate-800">
        <span>{label}</span>
        <span className="font-mono text-sm text-brand-700">{format(value)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
             onChange={(e) => onChange(Number(e.target.value))} className="mt-2 w-full accent-brand-700" />
      <div className="flex justify-between font-mono text-[11px] text-slate-500">
        <span>{format(min)}</span>
        <span>{format(max)}</span>
      </div>
    </label>
  );
}

export default function CorruptionControls({ value, onChange }) {
  const set = (patch) => onChange({ ...value, ...patch });
  return (
    <div className="space-y-5">
      <div>
        <div className="label-mono mb-2">Corruption type</div>
        <div className="grid grid-cols-2 gap-2">
          {TYPES.map((t) => (
            <button
              key={t.value}
              type="button"
              onClick={() => set({ type: t.value })}
              className={`flex items-center justify-between rounded-xl px-3.5 py-2.5 text-left text-[15px] transition ${
                value.type === t.value ? "bg-sky-100 font-medium text-slate-900" : "bg-brand-50/60 text-slate-700 hover:bg-brand-50"
              }`}
            >
              {t.label}
              <span className={`h-2.5 w-2.5 rounded-full ${t.dot}`} />
            </button>
          ))}
        </div>
      </div>
      {value.type !== "none" && (
        <>
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="label-mono">Corruption severity</span>
              {value.severity !== "custom" && (
                <span className="font-mono text-xs text-brand-700">{PRESETS[value.type][value.severity]}</span>
              )}
            </div>
            <Segmented
              className="w-full"
              value={value.severity}
              onChange={(severity) => set({ severity })}
              options={[
                { value: "low", label: "Low" },
                { value: "medium", label: "Medium" },
                { value: "high", label: "High" },
                { value: "custom", label: "Custom" },
              ]}
            />
          </div>
          {value.severity === "custom" && (
            <div className="space-y-4 rounded-xl bg-brand-50/50 p-4">
              {value.type === "salt_pepper" && (
                <Slider label="Pixel probability p" value={value.prob} min={0.01} max={0.3} step={0.01}
                        onChange={(prob) => set({ prob })} format={(v) => v.toFixed(2)} />
              )}
              {value.type === "blur" && (
                <>
                  <Slider label="Sigma (σ)" value={value.sigma} min={0.3} max={4} step={0.1}
                          onChange={(sigma) => set({ sigma })} format={(v) => `${v.toFixed(1)} px`} />
                  <Slider label="Kernel size" value={value.kernel} min={3} max={11} step={2}
                          onChange={(kernel) => set({ kernel })} format={(v) => `${v} × ${v}`} />
                </>
              )}
              {value.type === "occlusion" && (
                <>
                  <Slider label="Coverage" value={value.coverage} min={0.05} max={0.5} step={0.01}
                          onChange={(coverage) => set({ coverage })} format={(v) => `${Math.round(v * 100)} %`} />
                  <Slider label="Rectangles" value={value.numRects} min={1} max={5} step={1} onChange={(numRects) => set({ numRects })} />
                </>
              )}
            </div>
          )}
          <div className="flex items-center justify-between gap-3 rounded-xl bg-brand-50/50 px-4 py-3">
            <span className="text-[15px] text-slate-800">Random seed</span>
            <div className="flex items-center gap-2">
              <input type="number" value={value.seed} placeholder="random" onChange={(e) => set({ seed: e.target.value })}
                     className="w-28 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-center font-mono text-sm focus:border-brand-500 focus:outline-none" />
              <button type="button" title="New random seed" onClick={() => set({ seed: String(Math.floor(Math.random() * 1e6)) })}
                      className="rounded-lg bg-white p-2 text-slate-700 shadow-sm hover:bg-brand-50">
                <Shuffle className="h-4 w-4" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

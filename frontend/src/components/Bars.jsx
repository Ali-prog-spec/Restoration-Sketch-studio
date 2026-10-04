// Probability / routing-weight visualisations (absolute 0-100 % scale), Stitch style.
export const COLORS = {
  clean: "bg-emerald-500",
  identity: "bg-emerald-500",
  salt_pepper: "bg-amber-500",
  blur: "bg-sky-600",
  occlusion: "bg-rose-500",
};

/** One row per item: dot + label (+ optional tag) | value, and a bar underneath. */
export default function Bars({ items, highlight, tags = {} }) {
  return (
    <ul className="space-y-2.5">
      {items.map((it) => (
        <li key={it.key} className={`rounded-xl px-3 py-2.5 ${highlight === it.key ? "bg-brand-50" : "bg-slate-50"}`}>
          <div className="flex items-center justify-between gap-2 text-[15px]">
            <span className="flex items-center gap-2 font-medium text-slate-900">
              <span className={`h-2.5 w-2.5 rounded-full ${COLORS[it.key] || "bg-brand-500"}`} />
              {it.label}
              {tags[it.key] && (
                <span className="rounded bg-brand-100 px-1.5 py-0.5 font-mono text-[10px] font-normal text-brand-700">{tags[it.key]}</span>
              )}
            </span>
            <span className="font-mono text-sm tabular-nums text-slate-800">
              {it.value.toFixed(3)} <span className="text-slate-500">({(it.value * 100).toFixed(1)}%)</span>
            </span>
          </div>
          <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
            <div className={`h-full rounded-full ${COLORS[it.key] || "bg-brand-500"} transition-all duration-500`}
                 style={{ width: `${Math.max(it.value * 100, 0.5)}%` }} />
          </div>
        </li>
      ))}
    </ul>
  );
}

/** Single stacked bar of all weights with a legend (Stitch "soft routing distribution"). */
export function StackedBar({ items }) {
  return (
    <div>
      <div className="flex h-3 w-full overflow-hidden rounded-full">
        {items.map((it) => (
          <div key={it.key} className={COLORS[it.key]} style={{ width: `${it.value * 100}%` }} title={`${it.label} ${(it.value * 100).toFixed(1)}%`} />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 font-mono text-xs text-slate-700">
        {items.map((it) => (
          <span key={it.key} className="flex items-center gap-1.5">
            <span className={`h-2 w-2 rounded-full ${COLORS[it.key]}`} /> {it.short || it.label} {(it.value * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  );
}

// Presentational building blocks, styled after the Google Stitch design (report/stitch/).

export function Card({ title, icon: Icon, badge, subtitle, children, className = "", actions }) {
  return (
    <section className={`rounded-2xl border border-slate-200/80 bg-white p-6 shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="mb-5 flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            {Icon && <Icon className="mt-0.5 h-5 w-5 shrink-0 text-brand-600" strokeWidth={2} />}
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-lg font-semibold tracking-tight text-slate-900">{title}</h3>
                {badge}
              </div>
              {subtitle && <p className="mt-0.5 text-sm text-slate-500">{subtitle}</p>}
            </div>
          </div>
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Chip({ children, tone = "brand", className = "" }) {
  const tones = {
    brand: "bg-brand-50 text-brand-700",
    green: "bg-emerald-50 text-emerald-700",
    sky: "bg-sky-50 text-sky-700",
    amber: "bg-amber-50 text-amber-700",
    red: "bg-red-50 text-red-700",
    slate: "bg-slate-100 text-slate-600",
    dark: "bg-slate-900/80 text-white",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 font-mono text-[11px] ${tones[tone]} ${className}`}>
      {children}
    </span>
  );
}

export function Dot({ className = "bg-emerald-500" }) {
  return <span className={`inline-block h-2 w-2 rounded-full ${className}`} />;
}

export function Button({ children, variant = "primary", size = "md", className = "", ...props }) {
  const styles = {
    primary: "bg-brand-700 text-white shadow-sm hover:bg-brand-800 disabled:bg-slate-300 disabled:shadow-none",
    secondary: "bg-brand-50 text-slate-800 hover:bg-brand-100 disabled:text-slate-400",
    ghost: "text-slate-600 hover:bg-slate-100",
  };
  const sizes = { md: "px-4 py-2 text-sm", lg: "px-6 py-4 text-lg" };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2.5 rounded-xl font-medium transition disabled:cursor-not-allowed ${styles[variant]} ${sizes[size]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}

export function Segmented({ options, value, onChange, className = "" }) {
  return (
    <div className={`inline-flex flex-wrap gap-1 rounded-xl bg-brand-50/70 p-1 ${className}`}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded-lg px-3 py-1.5 text-sm transition ${
            value === o.value ? "bg-white font-medium text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Image panel with a monospace overlay tag (e.g. y_true / x_corrupted / y_hat) and an optional
 *  metric badge, as in the Stitch "Restoration Inspection Stage". */
export function ImagePanel({ title, note, noteClass = "text-slate-500", src, tag, badge, highlight, placeholder = "No image yet", footer }) {
  return (
    <figure className="flex min-w-0 flex-col">
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <figcaption className={`text-base font-medium ${highlight ? "text-brand-700" : "text-slate-900"}`}>{title}</figcaption>
        {note && <span className={`truncate font-mono text-xs ${noteClass}`}>{note}</span>}
      </div>
      <div className={`relative aspect-square w-full overflow-hidden rounded-xl bg-slate-100 ${highlight ? "ring-2 ring-brand-200 ring-offset-2" : "border border-slate-200"}`}>
        {src ? (
          <img src={src} alt={title} className="pixel-view h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center p-4 text-center text-xs text-slate-400">{placeholder}</div>
        )}
        {src && tag && (
          <span className={`absolute left-2 top-2 rounded-md px-2 py-0.5 font-mono text-[11px] text-white ${highlight ? "bg-brand-700" : "bg-slate-800/85"}`}>
            {tag}
          </span>
        )}
        {src && badge && (
          <span className={`absolute bottom-2 right-2 rounded-md px-2 py-0.5 font-mono text-[11px] ${highlight ? "bg-emerald-600 text-white" : "bg-slate-900/80 text-white"}`}>
            {badge}
          </span>
        )}
      </div>
      {footer && <div className="mt-2 font-mono text-xs text-slate-600">{footer}</div>}
    </figure>
  );
}

export function Stat({ label, value, unit, hint, highlight }) {
  return (
    <div className={`rounded-xl px-3 py-3 ${highlight ? "bg-brand-100/70" : "bg-brand-50/60"}`}>
      <div className="label-mono text-[10px]!">{label}</div>
      <div className={`mt-1 font-mono text-xl font-medium tabular-nums ${highlight ? "text-brand-700" : "text-slate-900"}`}>
        {value}
        {unit && <span className="ml-1 text-sm text-slate-500">{unit}</span>}
      </div>
      {hint && <div className="mt-0.5 font-mono text-[11px] text-slate-500">{hint}</div>}
    </div>
  );
}

export function JsonBlock({ title, data }) {
  return (
    <div>
      {title && <div className="label-mono mb-2">{title}</div>}
      <pre className="overflow-x-auto rounded-xl bg-slate-900 p-4 font-mono text-[12px] leading-relaxed text-slate-100">
        {JSON.stringify(data, null, 2)}
      </pre>
    </div>
  );
}

export function ErrorBanner({ error }) {
  if (!error) return null;
  return (
    <div role="alert" className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
      {error}
    </div>
  );
}

export function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />;
}

/** Page header: monospace breadcrumb, title, description, and a right-side checkpoint chip. */
export function WorkspaceHeader({ crumbs, title, description, checkpoint }) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <div className="label-mono flex flex-wrap items-center gap-2">
          {crumbs.map((c, i) => (
            <span key={i} className="flex items-center gap-2">
              {i > 0 && <span className="text-slate-400">/</span>}
              <span className={i === 1 ? "text-brand-700" : ""}>{c}</span>
            </span>
          ))}
        </div>
        <h2 className="mt-2 text-3xl font-semibold tracking-tight text-slate-900">{title}</h2>
        {description && <p className="mt-2 text-[15px] leading-relaxed text-slate-600">{description}</p>}
      </div>
      {checkpoint && (
        <div className="rounded-xl bg-brand-50/80 px-4 py-2.5 font-mono text-[13px] text-slate-700">
          <span className="text-slate-500">Checkpoint: </span>
          {checkpoint}
        </div>
      )}
    </div>
  );
}

export const fmt = (v, d = 2) => (v == null || Number.isNaN(v) ? "—" : Number(v).toFixed(d));

/** Convert a data URL (e.g. a restored output) into a File so it can be sent again. */
export async function dataUrlToFile(url, name) {
  const blob = await (await fetch(url)).blob();
  return new File([blob], name, { type: blob.type || "image/png" });
}

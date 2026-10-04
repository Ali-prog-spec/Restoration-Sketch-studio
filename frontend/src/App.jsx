import { ChartColumn, Network, PenTool, Split, WandSparkles } from "lucide-react";
import { useEffect, useState } from "react";
import { getHealth } from "./api.js";
import FaceToSketch from "./workspaces/FaceToSketch.jsx";
import SystemStatus from "./workspaces/SystemStatus.jsx";
import { HardRouted, SoftMixture, UniversalRestoration } from "./workspaces/Workspaces.jsx";

const WORKSPACES = [
  { id: "universal", label: "Task 1 · Universal Restoration", icon: WandSparkles, component: UniversalRestoration },
  { id: "hard", label: "Task 2 · Hard-Routed Restoration", icon: Split, component: HardRouted },
  { id: "soft", label: "Task 3 · Soft Mixture-of-Experts Restoration", icon: Network, component: SoftMixture },
  { id: "sketch", label: "Task 4 · Face-to-Sketch Generator", icon: PenTool, component: FaceToSketch },
];
const STATUS = { id: "status", label: "System status", icon: ChartColumn, component: SystemStatus };

function useHash() {
  const read = () => window.location.hash.replace("#", "") || "universal";
  const [hash, setHash] = useState(read);
  useEffect(() => {
    const on = () => setHash(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

function Logo({ className = "h-11 w-11" }) {
  return <img src="/logo.png" alt="" className={`${className} rounded-xl`} />;
}

function NavItem({ w, active }) {
  const Icon = w.icon;
  return (
    <a href={`#${w.id}`}
       className={`flex shrink-0 items-start gap-3 rounded-xl px-3 py-2.5 text-[15px] transition ${
         active ? "bg-brand-700 text-white shadow-sm" : "text-slate-700 hover:bg-brand-50"}`}>
      <Icon className={`mt-0.5 h-5 w-5 shrink-0 ${active ? "text-white" : "text-slate-600"}`} />
      <span>{w.label}</span>
    </a>
  );
}

export default function App() {
  const active = useHash();
  const [health, setHealth] = useState({ state: "connecting" });
  useEffect(() => {
    const t0 = performance.now();
    getHealth()
      .then((h) => setHealth({ state: h.all_models_available ? "ready" : "models-missing", ms: performance.now() - t0, ort: h.onnxruntime }))
      .catch(() => setHealth({ state: "offline" }));
  }, []);
  const Current = ([...WORKSPACES, STATUS].find((w) => w.id === active) || WORKSPACES[0]).component;
  const dot = { ready: "bg-emerald-500", "models-missing": "bg-amber-500", offline: "bg-red-500", connecting: "bg-slate-300" }[health.state];
  const text = { ready: "Backend ready", "models-missing": "Models missing", offline: "Backend offline", connecting: "Connecting…" }[health.state];

  return (
    <div className="min-h-screen bg-canvas lg:flex">
      {/* ---------------- sidebar ---------------- */}
      <aside className="border-b border-slate-200 bg-white lg:sticky lg:top-0 lg:flex lg:h-screen lg:w-80 lg:shrink-0 lg:flex-col lg:border-b-0 lg:border-r">
        <div className="flex items-center gap-3 border-b border-slate-200 px-6 py-5">
          <Logo />
          <div>
            <h1 className="text-lg font-semibold leading-tight">Restoration &amp; Sketch Studio</h1>
            <p className="label-mono text-[11px]! text-slate-600!">Generative AI Lab</p>
          </div>
        </div>
        <div className="label-mono hidden px-6 pb-2 pt-6 lg:block">Workspaces &amp; tasks</div>
        <nav className="flex gap-1 overflow-x-auto px-3 py-3 lg:flex-col lg:overflow-visible lg:py-0">
          {WORKSPACES.map((w) => <NavItem key={w.id} w={w} active={active === w.id} />)}
          <div className="my-2 hidden border-t border-slate-200 lg:block" />
          <NavItem w={STATUS} active={active === STATUS.id} />
        </nav>
        <div className="mt-auto hidden p-4 lg:block">
          <div className="rounded-xl border border-slate-200 px-4 py-3 font-mono text-[13px]">
            <div className="flex items-center gap-2 text-slate-900"><span className={`h-2.5 w-2.5 rounded-full ${dot}`} /> {text}</div>
            {health.ms != null && (
              <div className="mt-1 text-slate-500">Latency: {health.ms.toFixed(0)}ms · ONNX Runtime {health.ort}</div>
            )}
          </div>
        </div>
      </aside>

      {/* ---------------- main ---------------- */}
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-10 hidden items-center justify-between border-b border-slate-200 bg-white/90 px-10 py-4 backdrop-blur lg:flex">
          <div className="flex items-center gap-3">
            <Logo className="h-10 w-10" />
            <span className="text-lg font-medium">Restoration &amp; Sketch Studio</span>
          </div>
          <span className="rounded-full bg-brand-50 px-4 py-1.5 font-mono text-[13px] text-slate-700 ring-1 ring-brand-100">
            GenAI Assignment 1
          </span>
        </header>
        <main className="px-4 py-8 sm:px-8 lg:px-10">
          <Current />
        </main>
      </div>
    </div>
  );
}

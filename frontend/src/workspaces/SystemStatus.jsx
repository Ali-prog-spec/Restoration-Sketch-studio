import { Cpu, Layers, RefreshCw, Server, ShieldCheck, Timer } from "lucide-react";
import { useEffect, useState } from "react";
import { getHealth } from "../api.js";
import { Button, Card, Chip, Dot, ErrorBanner, WorkspaceHeader } from "../components/ui.jsx";

const ROWS = {
  task1: "Task 1 · Universal Restoration",
  task2_classifier: "Task 2 · Corruption Classifier",
  task2_salt_pepper: "Task 2/3 · Salt Specialist Expert",
  task2_blur: "Task 2/3 · Blur Specialist Expert",
  task2_occlusion: "Task 2/3 · Occlusion Specialist Expert",
  task3: "Task 3 · Complete Soft MoE Pipeline",
  task4: "Task 4 · Face-to-Sketch Generator",
};
const STATE = {
  loaded: ["green", "Loaded · in memory"],
  available: ["sky", "Available · on disk"],
  missing: ["red", "Missing"],
};

function Tile({ icon: Icon, label, value, sub }) {
  return (
    <div className="rounded-2xl bg-brand-50/60 p-4">
      <div className="flex items-center justify-between">
        <span className="label-mono">{label}</span>
        <Icon className="h-4 w-4 text-brand-700" />
      </div>
      <div className="mt-3 text-lg font-semibold text-slate-900">{value}</div>
      {sub && <div className="mt-1 font-mono text-xs text-slate-600">{sub}</div>}
    </div>
  );
}

export default function SystemStatus() {
  const [health, setHealth] = useState(null);
  const [ping, setPing] = useState(null);
  const [error, setError] = useState("");
  const load = () => {
    setError("");
    const t0 = performance.now();
    getHealth()
      .then((h) => {
        setPing(performance.now() - t0);
        setHealth(h);
      })
      .catch((e) => setError(e.message));
  };
  useEffect(load, []);
  const nAvailable = health ? Object.values(health.models).filter((v) => v !== "missing").length : 0;

  return (
    <div>
      <WorkspaceHeader crumbs={["System", "Backend health", "ONNX runtime"]} title="System Status & Runtime"
                       description="Backend health, inference runtime and the ONNX model registry mounted into the backend container." />
      <ErrorBanner error={error} />
      {health && (
        <div className="space-y-6">
          <Card title="Backend Environment" icon={Server}
                badge={<Chip tone="green"><Dot /> Backend online</Chip>}
                actions={<Button variant="secondary" onClick={load}><RefreshCw className="h-4 w-4" /> Refresh</Button>}>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Tile icon={Cpu} label="Execution provider" value={`ONNX Runtime ${health.onnxruntime}`}
                    sub={health.providers.includes("CPUExecutionProvider") ? "CPUExecutionProvider" : health.providers.join(", ")} />
              <Tile icon={Layers} label="Host framework" value={`FastAPI ${health.fastapi ?? ""}`} sub={`Python ${health.python ?? "?"} · API v${health.version}`} />
              <Tile icon={Timer} label="Health round-trip" value={`${ping != null ? ping.toFixed(0) : "—"} ms`} sub="measured from this browser" />
              <Tile icon={ShieldCheck} label="Model directory" value={`${nAvailable} / 7 models`} sub={health.model_dir} />
            </div>
          </Card>

          <Card title="ONNX Model Registry" icon={Layers}
                badge={<Chip tone={nAvailable === 7 ? "green" : "amber"}>{nAvailable} of 7 available</Chip>}
                subtitle="Models are loaded lazily on their first request.">
            <div className="overflow-x-auto">
              <table className="w-full min-w-160 text-left text-sm">
                <thead>
                  <tr className="bg-brand-50/70">
                    {["Model artifact", "Workspace / task", "State", "Size"].map((h) => (
                      <th key={h} className="label-mono px-4 py-3 font-normal first:rounded-l-xl last:rounded-r-xl">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {Object.entries(health.models).map(([k, v]) => {
                    const f = health.model_files?.[k] || {};
                    const [tone, label] = STATE[v] || ["slate", v];
                    return (
                      <tr key={k}>
                        <td className="px-4 py-4">
                          <div className="font-mono text-[13px] text-slate-900">{f.file || k}</div>
                          {f.sha256 && <div className="font-mono text-[11px] text-slate-500">SHA256: {f.sha256.slice(0, 6)}…{f.sha256.slice(-4)}</div>}
                        </td>
                        <td className="px-4 py-4 text-slate-700">{ROWS[k] || k}</td>
                        <td className="px-4 py-4"><Chip tone={tone}><Dot className={tone === "red" ? "bg-red-500" : tone === "sky" ? "bg-sky-600" : "bg-emerald-500"} /> {label}</Chip></td>
                        <td className="px-4 py-4 font-mono text-slate-700">{f.size_mb != null ? `${f.size_mb} MB` : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}

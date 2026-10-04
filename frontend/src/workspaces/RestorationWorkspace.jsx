import { Activity, Download, Grid3x3, Image as ImageIcon, Repeat, ScanSearch } from "lucide-react";
import { useEffect, useState } from "react";
import { getSamples, restore, sampleUrl } from "../api.js";
import CorruptionControls, { DEFAULT_CORRUPTION, TYPE_LABEL } from "../components/CorruptionControls.jsx";
import ImageSource from "../components/ImageSource.jsx";
import {
  Button, Card, Chip, Dot, ErrorBanner, ImagePanel, JsonBlock, Spinner, Stat, WorkspaceHeader, dataUrlToFile, fmt,
} from "../components/ui.jsx";

function corruptionNote(c) {
  if (!c) return "as uploaded";
  if (c.type === "salt_pepper") return `p=${fmt(c.prob, 2)}`;
  if (c.type === "blur") return `k=${c.kernel}, σ=${fmt(c.sigma, 1)}`;
  if (c.type === "occlusion") return `${Math.round(c.coverage * 100)}% · ${c.num_rects} rect`;
  return "";
}

/** Shared layout for the three restoration workspaces (Tasks 1-3), following the Stitch design:
 *  left = input + corruption + run button; right = inspection stage, analysis, telemetry. */
export default function RestorationWorkspace({
  endpoint, crumbs, title, description, checkpoint, runLabel, runIcon: RunIcon, outputNote,
  renderAnalysis, analysisFullWidth = false,
}) {
  const [source, setSource] = useState(null);
  const [corruption, setCorruption] = useState(DEFAULT_CORRUPTION);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    setLoading(true);
    setError("");
    try {
      setResult(await restore(endpoint, source, corruption));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  // "?autorun" (optionally "?autorun=<sample_id>") loads a clean sample and runs the model once:
  // used for scripted screenshots of the app for the report.
  useEffect(() => {
    const p = new URLSearchParams(window.location.search);
    if (!p.has("autorun")) return;
    getSamples().then(async (list) => {
      const s = list.find((x) => x.id === p.get("autorun")) || list[0];
      if (!s) return;
      const src = { sampleId: s.id, preview: sampleUrl(s.id), label: s.name };
      setSource(src);
      try {
        setResult(await restore(endpoint, src, DEFAULT_CORRUPTION));
      } catch (e) {
        setError(e.message);
      }
    });
  }, [endpoint]);

  // feed the restored output back in as a new (uncorrupted) input
  const copyToInput = async () => {
    const file = await dataUrlToFile(result.output_image, "restored.png");
    setSource({ file, preview: result.output_image, label: "previous restored output" });
    setCorruption({ ...corruption, type: "none" });
    setResult(null);
  };

  const m = result?.metrics;
  const analysis = result && renderAnalysis ? renderAnalysis(result) : null;
  const telemetry = result && (
    <Card title="System Telemetry & Metrics" icon={Activity} badge={<Chip tone="green">Ready</Chip>}>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Pre-proc" value={fmt(result.timing_ms.preprocess, 1)} unit="ms" />
        <Stat label="Corrupt" value={fmt(result.timing_ms.corruption, 1)} unit="ms" />
        <Stat label="Inference" value={fmt(result.timing_ms.inference, 1)} unit="ms" highlight />
        <Stat label="Total" value={fmt(result.timing_ms.total, 1)} unit="ms" />
      </div>
      {m && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <Stat label="PSNR (output)" value={fmt(m.output_psnr, 1)} unit="dB"
                hint={`input ${fmt(m.input_psnr, 1)} dB · ${m.output_psnr >= m.input_psnr ? "+" : ""}${fmt(m.output_psnr - m.input_psnr, 1)} dB`} />
          <Stat label="SSIM (output)" value={fmt(m.output_ssim, 3)}
                hint={`input ${fmt(m.input_ssim, 3)} · ${m.output_ssim >= m.input_ssim ? "+" : ""}${fmt(m.output_ssim - m.input_ssim, 3)}`} />
        </div>
      )}
      <div className="mt-4">
        <JsonBlock title="Inference request configuration" data={{
          task: result.task,
          corruption: result.corruption ?? "none (image used as uploaded)",
          source: result.source,
          models: result.models.map((x) => x.file),
          runtime: "onnxruntime (CPU)",
        }} />
      </div>
    </Card>
  );

  return (
    <div>
      <WorkspaceHeader crumbs={crumbs} title={title} description={description} checkpoint={checkpoint} />
      <div className="grid gap-6 xl:grid-cols-[400px_1fr]">
        {/* ---------------- left column ---------------- */}
        <div className="space-y-6">
          <Card title="Input image" icon={ImageIcon} badge={<Chip tone="slate">RGB · 128²</Chip>}>
            <ImageSource value={source} onChange={setSource} />
          </Card>
          <Card title="Synthetic Corruption" icon={Grid3x3}
                badge={<Chip tone="sky">Active: {TYPE_LABEL[corruption.type]}</Chip>}
                subtitle="Applied on the server at runtime, before restoration.">
            <CorruptionControls value={corruption} onChange={setCorruption} />
          </Card>
          <Button size="lg" className="w-full" disabled={!source || loading} onClick={run}>
            {loading ? <Spinner /> : <RunIcon className="h-5 w-5" />} {loading ? "Running…" : runLabel}
          </Button>
          <ErrorBanner error={error} />
        </div>

        {/* ---------------- right column ---------------- */}
        <div className="min-w-0 space-y-6">
          <Card title="Restoration Inspection Stage" icon={ScanSearch}
                badge={result ? <Chip tone="green"><Dot /> Complete</Chip> : <Chip tone="slate">Waiting for input</Chip>}>
            <div className="grid gap-5 sm:grid-cols-3">
              <ImagePanel title="Clean reference" note="Ground truth" src={result?.reference_image} tag="y_true"
                          placeholder={result ? "No clean reference: the upload was used as-is" : "Shown after a run"}
                          footer={result?.reference_image ? "Original (resized to 128×128)" : null} />
              <ImagePanel title="Model input" note={result ? corruptionNote(result.corruption) : "preview"} noteClass="text-sky-700"
                          src={result?.input_image || source?.preview} tag={result ? "x_corrupted" : "preview"}
                          badge={m ? `PSNR: ${fmt(m.input_psnr, 1)} dB` : null}
                          footer={m ? <>SSIM: {fmt(m.input_ssim, 3)}</> : null} />
              <ImagePanel title="Restored output" note={outputNote} noteClass="text-brand-700" highlight src={result?.output_image}
                          tag="y_hat (inference)" placeholder="Run the model to see the restoration"
                          badge={m ? `PSNR: ${fmt(m.output_psnr, 1)} dB` : null}
                          footer={m ? (
                            <>SSIM: {fmt(m.output_ssim, 3)}{" "}
                              <span className={m.output_psnr >= m.input_psnr ? "text-emerald-600" : "text-red-600"}>
                                {m.output_psnr >= m.input_psnr ? "+" : ""}{fmt(m.output_psnr - m.input_psnr, 1)} dB
                              </span></>
                          ) : null} />
            </div>
            {result && (
              <div className="mt-6 flex flex-wrap justify-end gap-3">
                <Button variant="secondary" onClick={copyToInput}><Repeat className="h-4 w-4" /> Copy to input</Button>
                <a href={result.output_image} download={`${endpoint}.png`}
                   className="inline-flex items-center gap-2 rounded-xl bg-brand-700 px-4 py-2 text-sm font-medium text-white hover:bg-brand-800">
                  <Download className="h-4 w-4" /> Download restored PNG
                </a>
              </div>
            )}
          </Card>

          {result && (analysisFullWidth ? (
            <>
              {analysis}
              {telemetry}
            </>
          ) : (
            <div className="grid gap-6 2xl:grid-cols-2">
              {analysis}
              {telemetry}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

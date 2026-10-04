import { Activity, Camera, Download, Palette, PenTool, Signature } from "lucide-react";
import { useState } from "react";
import { faceToSketch } from "../api.js";
import ImageSource from "../components/ImageSource.jsx";
import { Button, Card, Chip, Dot, ErrorBanner, ImagePanel, JsonBlock, Spinner, Stat, WorkspaceHeader, fmt } from "../components/ui.jsx";

// The three FS2K sketch-style categories (dataset labels 0/1/2). No extra semantics are
// claimed beyond the dataset's own categories.
const STYLES = [
  { value: 1, title: "Style 1", label: "FS2K style category 0" },
  { value: 2, title: "Style 2", label: "FS2K style category 1" },
  { value: 3, title: "Style 3", label: "FS2K style category 2" },
];

export default function FaceToSketch() {
  const [source, setSource] = useState(null);
  const [style, setStyle] = useState(1);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    setLoading(true);
    setError("");
    try {
      setResult(await faceToSketch(source.file, style));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <WorkspaceHeader
        crumbs={["Task 04", "Conditional GAN", "ONNX"]}
        title="Face-to-Sketch Generator"
        description="A style-conditioned pix2pix GAN (U-Net generator, PatchGAN discriminator) turns a face photograph into a sketch in one of the three FS2K styles. The style is a learned embedding used by both networks."
        checkpoint="task4_generator.onnx"
      />
      <div className="grid gap-6 xl:grid-cols-[440px_1fr]">
        <div className="space-y-6">
          <Card title="Face Photograph" icon={Camera}>
            <ImageSource value={source} onChange={setSource} allowSamples={false} allowWebcam />
          </Card>
          <Card title="Sketch Style" icon={Palette} badge={<Chip>Learned style embedding</Chip>}>
            <div className="space-y-3">
              {STYLES.map((s) => (
                <button key={s.value} type="button" onClick={() => setStyle(s.value)}
                        className={`flex w-full items-center gap-4 rounded-2xl p-4 text-left transition ${
                          style === s.value ? "bg-white ring-2 ring-brand-700" : "bg-brand-50/50 hover:bg-brand-50"}`}>
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-brand-100 text-brand-700">
                    <PenTool className="h-5 w-5" />
                  </span>
                  <span className="flex-1">
                    <span className="block text-base font-semibold text-slate-900">{s.title}</span>
                    <span className="block font-mono text-xs text-slate-500">{s.label}</span>
                  </span>
                  {style === s.value && <Chip>Selected</Chip>}
                </button>
              ))}
            </div>
          </Card>
          <Button size="lg" className="w-full" disabled={!source?.file || loading} onClick={run}>
            {loading ? <Spinner /> : <Signature className="h-5 w-5" />} {loading ? "Generating…" : "Generate Sketch"}
          </Button>
          <ErrorBanner error={error} />
        </div>

        <div className="min-w-0 space-y-6">
          <Card title="Sketch Synthesis Result" icon={PenTool}
                badge={result ? <Chip tone="green"><Dot /> Generated in {fmt(result.timing_ms.inference, 0)} ms</Chip> : null}
                subtitle="The model input photograph and the generated sketch, side by side.">
            <div className="grid gap-5 sm:grid-cols-2">
              <ImagePanel title="Original photograph" note="RGB · 128×128" src={result?.input_image || source?.preview}
                          tag={result ? "x (model input)" : "preview"} />
              <ImagePanel title="Generated sketch" note={result ? result.style : `Style ${style}`} noteClass="text-brand-700" highlight
                          src={result?.output_image} tag="G(x, s)" placeholder="Generate to see the sketch" />
            </div>
            {result && (
              <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-slate-100 pt-4">
                <span className="font-mono text-sm text-slate-600">Resolution: 128 × 128 px · grayscale</span>
                <a href={result.output_image} download={`sketch_style${result.style_index}.png`}
                   className="inline-flex items-center gap-2 rounded-xl bg-brand-700 px-5 py-2.5 text-sm font-medium text-white hover:bg-brand-800">
                  <Download className="h-4 w-4" /> Download PNG
                </a>
              </div>
            )}
          </Card>
          {result && (
            <Card title="Synthesis Diagnostics" icon={Activity}>
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Pre-process" value={fmt(result.timing_ms.preprocess, 1)} unit="ms" hint="decode · RGB · resize" />
                <Stat label="Generator" value={fmt(result.timing_ms.inference, 1)} unit="ms" hint="ONNX Runtime (CPU)" />
                <Stat label="Total" value={fmt(result.timing_ms.total, 1)} unit="ms" highlight />
              </div>
              <div className="mt-4">
                <JsonBlock title="Request" data={{ task: result.task, style: result.style, style_index: result.style_index,
                  source: result.source, models: result.models.map((x) => x.file) }} />
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

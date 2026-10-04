import { Network, Split, Waypoints, WandSparkles } from "lucide-react";
import Bars, { StackedBar } from "../components/Bars.jsx";
import { Card, Chip, Stat, fmt } from "../components/ui.jsx";
import RestorationWorkspace from "./RestorationWorkspace.jsx";

const CLASS_LABELS = { clean: "Clean", salt_pepper: "Salt-and-pepper", blur: "Gaussian blur", occlusion: "Occlusion" };
const BRANCH_LABELS = {
  identity: "Identity (clean bypass)", salt_pepper: "Salt & Pepper Expert", blur: "Blur Expert", occlusion: "Occlusion Expert",
};
const BRANCH_SHORT = { identity: "Identity", salt_pepper: "Salt", blur: "Blur", occlusion: "Occlusion" };

function FactTable({ rows }) {
  return (
    <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 rounded-xl bg-brand-50/50 p-4 font-mono text-[13px]">
      {rows.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-slate-500">{k}</dt>
          <dd className="text-slate-900">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function UniversalRestoration() {
  return (
    <RestorationWorkspace
      endpoint="universal-restoration"
      crumbs={["Task 01", "Universal checkpoint", "ONNX"]}
      title="Universal Blind Restoration"
      description="One convolutional denoising autoencoder restores clean, salt-and-pepper, blurred and occluded images without being told which corruption was applied."
      checkpoint="task1_universal_dae.onnx"
      runLabel="Run Universal Restoration"
      runIcon={WandSparkles}
      outputNote="Universal DAE"
      renderAnalysis={() => (
        <Card title="Universal Restorer" icon={WandSparkles} badge={<Chip>Task 1 model</Chip>}>
          <p className="text-[15px] leading-relaxed text-slate-600">
            A single shared representation for every input condition: the network is never told which
            corruption was applied, and all information passes through a compressed latent code
            (no skip connections).
          </p>
          <FactTable rows={[
            ["Architecture", "Conv encoder → bottleneck → conv decoder"],
            ["Skip connections", "none (genuine bottleneck)"],
            ["Trained on", "clean · salt & pepper · blur · occlusion"],
            ["Loss", "α·L1 + (1−α)·(1 − SSIM)"],
            ["Input", "RGB 128 × 128, values in [0, 1]"],
          ]} />
        </Card>
      )}
    />
  );
}

export function HardRouted() {
  return (
    <RestorationWorkspace
      endpoint="hard-routing"
      crumbs={["Task 02", "Classifier + specialists", "ONNX"]}
      title="Hard-Routed Restoration"
      description="A CNN classifier predicts the corruption; the image is then sent to exactly one specialist autoencoder. Clean predictions use an identity bypass and no expert is run."
      checkpoint="task2_classifier.onnx + 3 experts"
      runLabel="Run Hard-Routed Restoration"
      runIcon={Split}
      outputNote="Selected expert"
      renderAnalysis={(r) => (
        <Card title="Corruption Classifier & Routing" icon={Split}
              badge={<Chip tone="sky">Top-1 confidence: {fmt(r.probabilities[r.predicted_class] * 100, 1)}%</Chip>}
              subtitle="Four class probabilities; the most probable class selects the expert.">
          <Bars highlight={r.predicted_class}
                items={Object.entries(r.probabilities).map(([k, v]) => ({ key: k, label: CLASS_LABELS[k], value: v }))}
                tags={{ [r.predicted_class]: "Predicted" }} />
          <div className="mt-4 grid grid-cols-2 gap-2">
            <Stat label="Predicted corruption" value={<span className="font-sans text-lg">{r.predicted_display}</span>} />
            <Stat label="Selected expert" value={<span className="font-sans text-lg">{r.selected_expert_display}</span>} />
            <Stat label="Classifier time" value={fmt(r.timing_detail_ms.classifier, 1)} unit="ms" />
            <Stat label="Expert time" value={fmt(r.timing_detail_ms.expert, 1)} unit="ms"
                  hint={r.selected_expert === "identity" ? "identity bypass: no expert run" : null} />
          </div>
        </Card>
      )}
    />
  );
}

export function SoftMixture() {
  return (
    <RestorationWorkspace
      endpoint="soft-mixture"
      crumbs={["Task 03", "Soft mixture-of-experts", "ONNX"]}
      title="Soft Mixture-of-Experts Restoration"
      description="A gating network assigns a continuous weight to the identity branch and each specialist; the output is their weighted sum, fine-tuned end to end."
      checkpoint="task3_soft_moe.onnx"
      runLabel="Run Soft MoE Restoration"
      runIcon={Network}
      outputNote="MoE blended"
      analysisFullWidth
      renderAnalysis={(r) => {
        const items = r.ranking.map((b) => ({ key: b.branch, label: BRANCH_LABELS[b.branch], short: BRANCH_SHORT[b.branch], value: b.weight }));
        const ordered = ["identity", "salt_pepper", "blur", "occlusion"].map((k) => items.find((i) => i.key === k));
        const [first, second] = r.ranking;
        return (
          <Card title="Gating Network Soft Routing Distribution" icon={Waypoints}
                badge={<Chip>Σ w_k = {fmt(Object.values(r.weights).reduce((a, b) => a + b, 0), 3)}</Chip>}
                subtitle={`w = softmax(G(x̃) / τ)${r.tau ? `, τ = ${fmt(r.tau, 2)}` : ""} across the 4 branches`}>
            <StackedBar items={ordered} />
            <div className="mt-5">
              <Bars items={items} highlight={first.branch} tags={{ [first.branch]: "Primary", [second.branch]: "Secondary" }} />
            </div>
            <div className="mt-4 grid gap-2 sm:grid-cols-3">
              <Stat label="Strongest contributor" value={<span className="font-sans text-lg text-brand-700">{BRANCH_SHORT[first.branch]}</span>}
                    hint={`weight ${fmt(first.weight, 3)} (${fmt(first.weight * 100, 1)}%)`} />
              <Stat label="Second contributor" value={<span className="font-sans text-lg text-amber-600">{BRANCH_SHORT[second.branch]}</span>}
                    hint={`weight ${fmt(second.weight, 3)} (${fmt(second.weight * 100, 1)}%)`} />
              <Stat label="Routing entropy" value={fmt(r.entropy_nats, 3)} unit="nats"
                    hint={r.entropy_nats < 0.35 ? "one expert dominates" : r.entropy_nats > 0.9 ? "high mixture" : "partly mixed"} />
            </div>
          </Card>
        );
      }}
    />
  );
}

import { Camera, CloudUpload } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { getSamples, sampleUrl } from "../api.js";
import { Button, Segmented } from "./ui.jsx";

/** Upload / clean-sample picker / (optionally) webcam capture.
 *  onChange({ file?: File, sampleId?: string, preview: string, label: string }) */
export default function ImageSource({ value, onChange, allowSamples = true, allowWebcam = false }) {
  const modes = [{ value: "upload", label: "Upload custom" }];
  if (allowSamples) modes.push({ value: "sample", label: "Clean samples" });
  if (allowWebcam) modes.push({ value: "webcam", label: "Webcam" });
  const [mode, setMode] = useState(allowSamples ? "sample" : "upload");
  const [samples, setSamples] = useState([]);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (allowSamples) getSamples().then(setSamples).catch(() => setSamples([]));
  }, [allowSamples]);

  const pickFile = (file) => {
    if (!file) return;
    onChange({ file, preview: URL.createObjectURL(file), label: file.name });
  };

  return (
    <div className="space-y-4">
      <Segmented options={modes} value={mode} onChange={setMode} className="w-full" />
      {mode === "upload" && (
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            pickFile(e.dataTransfer.files[0]);
          }}
          className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 py-7 text-center transition ${
            dragging ? "border-brand-500 bg-brand-50" : "border-brand-200 bg-brand-50/40 hover:border-brand-500"
          }`}
        >
          <span className="mb-3 flex h-11 w-11 items-center justify-center rounded-full bg-white shadow-sm">
            <CloudUpload className="h-5 w-5 text-brand-700" />
          </span>
          <span className="text-[15px] text-slate-800">Drop an image or browse</span>
          <span className="mt-1 font-mono text-xs text-slate-500">PNG, JPG, WEBP, BMP (max 10 MB) · resized to 128×128</span>
          <input type="file" accept="image/jpeg,image/png,image/webp,image/bmp" className="hidden"
                 onChange={(e) => pickFile(e.target.files[0])} />
        </label>
      )}
      {mode === "sample" && (
        <div>
          <div className="label-mono mb-2">Clean samples (Oxford-IIIT Pet test set)</div>
          <div className="grid grid-cols-4 gap-2">
            {samples.length === 0 && <p className="col-span-4 text-xs text-slate-500">No samples available.</p>}
            {samples.map((s) => (
              <button
                key={s.id}
                type="button"
                title={s.name}
                onClick={() => onChange({ sampleId: s.id, preview: sampleUrl(s.id), label: s.name })}
                className={`aspect-square overflow-hidden rounded-xl transition ${
                  value?.sampleId === s.id ? "ring-2 ring-brand-700 ring-offset-2" : "hover:opacity-80"
                }`}
              >
                <img src={sampleUrl(s.id)} alt={s.name} className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
        </div>
      )}
      {mode === "webcam" && <Webcam onCapture={pickFile} />}
      {value?.label && <p className="truncate font-mono text-xs text-slate-500">Selected: {value.label}</p>}
    </div>
  );
}

function Webcam({ onCapture }) {
  const video = useRef(null);
  const [stream, setStream] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  const start = async () => {
    setError("");
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      setStream(s);
      if (video.current) video.current.srcObject = s;
    } catch (e) {
      setError(`Camera unavailable: ${e.message}. Webcam access needs https or localhost.`);
    }
  };
  const capture = () => {
    const v = video.current;
    const side = Math.min(v.videoWidth, v.videoHeight);
    const c = document.createElement("canvas");
    c.width = c.height = side; // square centre crop (the model input is square)
    c.getContext("2d").drawImage(v, (v.videoWidth - side) / 2, (v.videoHeight - side) / 2, side, side, 0, 0, side, side);
    c.toBlob((b) => onCapture(new File([b], "webcam.png", { type: "image/png" })), "image/png");
  };

  return (
    <div className="space-y-3">
      <div className="relative aspect-video overflow-hidden rounded-xl bg-slate-200">
        <video ref={video} autoPlay playsInline muted className="h-full w-full object-cover" />
        {stream && (
          <span className="absolute left-3 top-3 flex items-center gap-1.5 rounded-md bg-slate-900/80 px-2 py-0.5 font-mono text-[11px] text-white">
            <span className="h-2 w-2 rounded-full bg-red-500" /> LIVE
          </span>
        )}
        {/* square guide = the region that is captured */}
        {stream && <div className="pointer-events-none absolute inset-y-3 left-1/2 aspect-square -translate-x-1/2 rounded-xl border-2 border-white/70" />}
      </div>
      {!stream ? (
        <Button variant="secondary" className="w-full" onClick={start}><Camera className="h-4 w-4" /> Start camera</Button>
      ) : (
        <Button className="w-full" onClick={capture}><Camera className="h-4 w-4" /> Snap frame</Button>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}

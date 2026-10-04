"""ONNX export + numerical validation against PyTorch.

For every exported model we:
  1. export with torch.onnx.export (torch.export-based "dynamo" exporter, the
     default since PyTorch 2.9; the legacy TorchScript exporter is a fallback),
     with a dynamic batch dimension and weights embedded in one .onnx file
  2. check the graph with onnx.checker
  3. run the SAME inputs through PyTorch (eval, fp32, CPU) and ONNX Runtime (CPU)
  4. compare output shapes and values (max / mean absolute difference)
  5. mark the export FAILED if np.allclose(onnx, torch, atol, rtol) is false
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

try:  # the dynamo exporter prints emoji progress messages; Windows consoles need UTF-8
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def export_onnx(model: torch.nn.Module, example_inputs: tuple, path, input_names: list[str],
                output_names: list[str], opset: int = 18) -> dict:
    model = model.eval().cpu()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arg_names = list(model.forward.__code__.co_varnames[1 : 1 + len(example_inputs)])
    batch = torch.export.Dim("batch", min=1, max=256)
    info = {"path": str(path), "opset": opset}
    try:
        torch.onnx.export(model, example_inputs, str(path), dynamo=True, input_names=input_names,
                          output_names=output_names, opset_version=opset, external_data=False,
                          dynamic_shapes={n: {0: batch} for n in arg_names})
        info["exporter"] = "dynamo"
    except Exception as exc:  # noqa: BLE001
        print(f"[onnx] dynamo export failed ({type(exc).__name__}: {exc}); falling back to legacy exporter")
        torch.onnx.export(model, example_inputs, str(path), dynamo=False, input_names=input_names,
                          output_names=output_names, opset_version=17,
                          dynamic_axes={n: {0: "batch"} for n in input_names + output_names})
        info["exporter"] = "torchscript-legacy"
        info["opset"] = 17
    onnx.checker.check_model(onnx.load(str(path)))
    info["size_mb"] = round(path.stat().st_size / 1024**2, 3)
    return info


@torch.no_grad()
def validate_onnx(model: torch.nn.Module, path, inputs: tuple, atol: float = 1e-4, rtol: float = 1e-3) -> dict:
    """``inputs``: tuple of torch tensors (same order as the ONNX inputs)."""
    model = model.eval().cpu()
    ref = model(*inputs)
    ref = ref if isinstance(ref, (tuple, list)) else (ref,)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {i.name: t.numpy() for i, t in zip(sess.get_inputs(), inputs)}
    t0 = time.perf_counter()
    outs = sess.run(None, feeds)
    ort_ms = (time.perf_counter() - t0) * 1000
    results = []
    passed = len(outs) == len(ref)
    for o, r, meta in zip(outs, ref, sess.get_outputs()):
        r = r.float().numpy()
        same_shape = tuple(o.shape) == tuple(r.shape)
        diff = np.abs(o - r) if same_shape else np.array([np.inf])
        ok = same_shape and bool(np.allclose(o, r, atol=atol, rtol=rtol))
        passed &= ok
        results.append({"name": meta.name, "onnx_shape": list(o.shape), "torch_shape": list(r.shape),
                        "shape_match": same_shape, "max_abs_diff": float(diff.max()),
                        "mean_abs_diff": float(diff.mean()), "allclose": ok})
    return {"passed": bool(passed), "atol": atol, "rtol": rtol, "batch_size": int(inputs[0].shape[0]),
            "outputs": results, "onnxruntime_ms_batch": round(ort_ms, 2),
            "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in sess.get_inputs()]}

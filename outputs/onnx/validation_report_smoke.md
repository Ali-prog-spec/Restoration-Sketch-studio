# ONNX validation report

Mode: `smoke` — tolerance atol=0.0001, rtol=1e-3 (np.allclose)

| Model | File | Exporter | Batch | Output | Shape match | max abs diff | mean abs diff | Passed |
|---|---|---|---|---|---|---|---|---|
| task1 | task1_universal_dae.onnx | dynamo | 20 | restored | True | 8.941e-08 | 1.546e-08 | PASS |
| task2_classifier | task2_classifier.onnx | dynamo | 20 | logits | True | 1.490e-07 | 2.910e-08 | PASS |
| task2_salt_pepper | task2_salt_expert.onnx | dynamo | 20 | restored | True | 8.941e-08 | 1.559e-08 | PASS |
| task2_blur | task2_blur_expert.onnx | dynamo | 20 | restored | True | 5.960e-08 | 1.542e-08 | PASS |
| task2_occlusion | task2_occlusion_expert.onnx | dynamo | 20 | restored | True | 5.960e-08 | 1.540e-08 | PASS |
| task3 | task3_soft_moe.onnx | dynamo | 20 | restored | True | 2.384e-07 | 2.381e-08 | PASS |
| task3 | task3_soft_moe.onnx | dynamo | 20 | weights | True | 8.941e-08 | 1.434e-08 | PASS |
| task4 | task4_generator.onnx | dynamo | 9 | sketch | True | 7.153e-06 | 6.862e-07 | PASS |

**All passed: True**
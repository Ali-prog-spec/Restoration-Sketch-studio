# ONNX validation report

Mode: `final` — tolerance atol=0.0001, rtol=1e-3 (np.allclose)

| Model | File | Exporter | Batch | Output | Shape match | max abs diff | mean abs diff | Passed |
|---|---|---|---|---|---|---|---|---|
| task1 | task1_universal_dae.onnx | dynamo | 20 | restored | True | 2.831e-06 | 9.698e-08 | PASS |
| task2_classifier | task2_classifier.onnx | dynamo | 20 | logits | True | 3.815e-06 | 7.640e-07 | PASS |
| task2_salt_pepper | task2_salt_expert.onnx | dynamo | 20 | restored | True | 2.921e-06 | 1.062e-07 | PASS |
| task2_blur | task2_blur_expert.onnx | dynamo | 20 | restored | True | 3.457e-06 | 1.268e-07 | PASS |
| task2_occlusion | task2_occlusion_expert.onnx | dynamo | 20 | restored | True | 7.570e-06 | 1.886e-07 | PASS |
| task3 | task3_soft_moe.onnx | dynamo | 20 | restored | True | 3.159e-06 | 1.039e-07 | PASS |
| task3 | task3_soft_moe.onnx | dynamo | 20 | weights | True | 2.086e-07 | 3.353e-08 | PASS |
| task4 | task4_generator.onnx | dynamo | 9 | sketch | True | 8.076e-06 | 3.067e-07 | PASS |

**All passed: True**
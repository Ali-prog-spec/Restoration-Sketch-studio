# Research notes — design decisions and evidence

Every citation key refers to `docs/references.bib` (entries verified against DOI / arXiv /
proceedings pages on 2026-09-23). "Experiment" names the artifact that tests the decision.
Results are **not** written here; they live in `outputs/` and are pulled into the report.

Format of each entry: **Question → Alternatives → Evidence → Decision → Why → Verification experiment.**

---

## 1. Data and corruption protocol

### 1.1 Occlusion coverage: sum of rectangle areas or union?
- **Alternatives:** (a) sample rectangle areas whose *sum* is 10–35 %; (b) require the *union* (what is actually hidden) to be 10–35 %.
- **Evidence:** the PDF says the masks must "jointly cover between 10 % and 35 % of the image area" (p.3). Overlapping rectangles make (a) cover less than claimed.
- **Decision:** (b). The target coverage ~ U(0.10, 0.35) is split between 1–3 rectangles (Dirichlet α = 3, aspect ratio in [0.5, 2], random positions), with rejection sampling until the union is within ±1.5 points of the target *and* inside [10 %, 35 %].
- **Verification:** `tests/test_corruptions.py::test_training_parameter_ranges` checks 3,000 samples; `outputs/tables/manifest_summary_test.csv` gives realised test coverage (min/mean/max per level).

### 1.2 Salt-and-pepper: per pixel or per channel?
- **Alternatives:** corrupt channels independently (coloured noise) or whole pixels.
- **Evidence:** the PDF says to "replace the selected pixels with black or white values" (p.3). Impulse-noise models in Gonzalez & Woods `gonzalez2018dip` set pixels to the extreme values.
- **Decision:** whole-pixel corruption (all three channels → 0 or 1), each with probability 0.5.
- **Verification:** `test_salt_and_pepper_pixels` checks the ≈ p fraction, pure black/white values and a 50/50 split.

### 1.3 Blur border handling
- **Decision:** separable Gaussian with **reflect** padding. Zero padding would darken borders, adding a second, unintended corruption.
- **Verification:** the result matches `scipy.ndimage.convolve1d(mode="mirror")` to 1e-5 (`test_gaussian_blur_matches_scipy`).

### 1.4 Deterministic validation / test
- **Decision:**
  - The **validation** manifest has 4 records per validation image (clean + 3 corruptions), with severities drawn from the *training* ranges using per-record `SeedSequence` seeds. The set is therefore exactly balanced, and each Task 2 specialist gets 736 validation images.
  - The **test** manifest has 10 records per test image: clean + 3 corruptions × the 3 fixed PDF levels.
  - Records store type, level, seed, probability, kernel, σ, rectangle coordinates and achieved coverage.
- **Why not save corrupted images:** the PDF forbids this for training (p.3). For validation/test, the parameters + seed fully determine the image, so storing them is enough and far smaller (8.7 MB JSON instead of ~1.8 GB of images).
- **Verification:** `tests/test_manifests_and_data.py` checks that reloading a manifest reproduces identical tensors, that generation is deterministic, and that all fields are present.

### 1.5 Balanced batches
- **Question:** the PDF requires equal probability per image (p.3) *and* balanced classifier batches (p.4), and defines the MoE balance loss "within a balanced training batch" (p.6).
- **Decision:** a `BalancedBatchSampler` shuffles images and assigns exactly B/4 of each condition to every batch, in random order. The marginal probability per image stays 1/4 while every batch is exactly balanced. Used for Tasks 1, 2 (classifier) and 3.
- **Verification:** `test_balanced_batch_sampler`. The classifier history also logs `train_class_min/max` per epoch.

### 1.6 Dataset source for Oxford-IIIT Pet
- **Problem:** the official server throttles each connection to ~150 KB/s and ignores HTTP range requests, so the 792 MB archive could not be fetched in reasonable time.
- **Decision:** use the timm Hugging Face mirror, which carries the original bytes plus `image_id` (CC BY-SA 4.0, same licence), and **verify it against the official source**:
  - The mirror's train/test ids equal the official `annotations/trainval.txt` / `test.txt`.
  - Every image recoverable from a partial official `images.tar.gz` download is byte-identical (MD5).
- **Evidence:** `data/manifests/oxford_source_verification.json`. `scripts/prepare_oxford.py --download` still supports the official URLs.

### 1.7 Resizing
- The PDF says "resized to 128 × 128" (p.3). We resize directly (bicubic with antialiasing, no crop), so no image content is lost. The cost is aspect-ratio distortion, which is identical for the clean target and the input and therefore does not bias restoration.

---

## 2. Task 1 — universal denoising autoencoder

### 2.1 Architecture family
- **Alternatives:**
  - Fully-connected DAE (Vincent et al. `vincent2008dae`, `vincent2010sdae`)
  - Convolutional encoder–decoder with symmetric skips (RED-Net, `mao2016rednet`)
  - U-Net (`ronneberger2015unet`)
  - Residual denoisers predicting the noise (DnCNN, `zhang2017dncnn`)
- **Evidence:**
  - Symmetric skips / U-Nets / global residuals give much better restoration *because* they pass high-resolution input detail around the bottleneck (`mao2016rednet`).
  - The PDF explicitly forbids "simply copying the input through unrestricted skip connections" (p.3–4).
  - A DnCNN-style global residual (x̂ = x̃ − r) is also an unrestricted identity path.
- **Decision:** a convolutional autoencoder with **no skips by default**:
  - 4 stride-2 stages (channels C, 2C, 4C, 8C), then a 1×1 bottleneck conv to L channels at 8×8, then 4 bilinear-upsample + conv stages, then a sigmoid output.
  - Optuna may also select a 3-stage variant (16×16 grid).
  - Upsampling is bilinear + conv instead of transposed conv, to avoid checkerboard artefacts (`odena2016checkerboard`).
- **Verification:** `test_no_skip_by_default_output_depends_only_on_latent` shows that with random encoder features and the same z, the decoder output is identical, so information can only flow through z.

### 2.2 How big may the bottleneck be? (and what "bottleneck dimension" means)
- The input has 3·128·128 = 49,152 values.
- Optuna chooses the bottleneck *shape* from {8×8×16, 8×8×32, 8×8×64, 16×16×8, 16×16×16}. That is 1,024–4,096 latent values, i.e. **12×–48× compression**.
- Every choice is a genuine compression, so a trial cannot "win" by removing the bottleneck.
- Spatial latents (rather than a flat vector) were chosen because restoration needs spatial layout. A 1-D (fully-connected) latent of the same size would force the decoder to re-learn positions from a dense layer. Hypothesis, NOT tested in this project: this would give blurrier outputs.
- **Experiment:** Optuna parameter `bottleneck` (importance plot); `outputs/task1/optuna/final/trials.csv` has J per bottleneck.

### 2.3 Limited skip connections (investigated, off by default)
- **Question (PDF p.4):** if limited skips are used, investigate their purpose and effect.
- **Design of the ablation:** one skip at the **coarsest** decoder level only (16×16), squeezed to 4 channels through a 1×1 conv (`--skip-levels 0`). No skip at 64×64 or 128×128, and no input→output residual.
- **Hypothesis:** the skip carries some spatial detail (sharper edges, better PSNR on clean/blur) but also passes corruption through (salt pixels, occluder edges), so it can hurt salt/occlusion.
- **Experiment:** `scripts/train_task1.py --skip-levels 0 --tag skip0` + `evaluate_task1.py --tag skip0`, compared per condition with the no-skip model (`outputs/task1/metrics_by_type_skip0.csv`). The deployed model is the no-skip one unless the ablation argues otherwise.

### 2.4 Loss: L1 + SSIM
- **Evidence:** Zhao et al. `zhao2017loss` show that L2 correlates poorly with perceived quality and that L1 plus (MS-)SSIM mixes outperform L2, using α = 0.84 on MS-SSIM. SSIM is defined in `wang2004ssim`.
- **Decision:** the PDF's L = α·L1 + (1−α)(1−SSIM), with single-scale SSIM (11×11 Gaussian, σ = 1.5, K1 = 0.01, K2 = 0.03). MS-SSIM needs ≥ 5 dyadic scales, which is impractical at 128 px with an 11-pixel window. α starts at 0.8 (PDF) and Optuna searches [0.5, 0.95], which contains Zhao's 0.84.
- **Implementation check:** our SSIM matches `skimage.metrics.structural_similarity` to 2e-3 (`test_ssim_matches_skimage`).

### 2.5 Optuna objective (must not depend on α)
- **Problem:** if trials were compared on their *training* loss, α itself would change the objective's scale (α → 1 leaves only the small L1 term), and Optuna would "optimise" the metric instead of the model.
- **Decision:** a fixed selection objective J = 0.5·L1 + 0.5·(1 − SSIM) on the deterministic validation manifest. It combines reconstruction quality and structural similarity, as the PDF requires (p.4).
- **Sampler/pruner:**
  - Sampler: TPE `bergstra2011tpe`, seeded with 42, as implemented in Optuna `akiba2019optuna`.
  - Pruner: median pruning (warm-up 3 epochs, 4 start-up trials), a successive-halving-style early-stopping rule (`jamieson2016sh`, `li2018hyperband`).
  - The PDF baseline (α = 0.8, lr 1e-3, batch 32, 8×8×16, C = 32, dropout 0.1) is enqueued as trial 0, so "baseline vs. tuned" is part of the study.

### 2.6 Search space
| Parameter | Range | Justification |
|---|---|---|
| lr | 1e-4 – 3e-3 (log) | Adam defaults to 1e-3 `kingma2015adam`; one decade around it |
| batch size | 16, 32, 64 | multiples of 4 (balanced batches); fits 6 GB VRAM |
| bottleneck | 5 shapes, 12×–48× | §2.2 |
| encoder channels C | 16, 32, 48 | capacity vs. 6 GB / time budget |
| dropout | 0 – 0.3 | `srivastava2014dropout`; Dropout2d on the latent and first decoder stage |
| α | 0.5 – 0.95 | contains the PDF 0.8 and Zhao's 0.84 |

### 2.7 Metrics
- **Required by the PDF:** results per condition and severity; L1 and SSIM are the named quantities.
- **Additional:** PSNR, the standard restoration number, which is complementary to SSIM (`hore2010psnr`).
- The metrics of the corrupted input are also reported (identity baseline), so every number shows the *gain* from restoration.
- PSNR of an exactly lossless output (identity on a clean image, Task 2) is infinite; it is capped at 100 dB and flagged in tables.

---

## 3. Task 2 — classifier and hard routing

### 3.1 Classifier design
- **Cues:**
  - Salt-and-pepper produces isolated extreme pixels (high-frequency, sparse).
  - Blur removes high frequencies everywhere (global).
  - Occlusion produces large perfectly black, sharp-edged regions (sparse, extreme).
- **Decision:** a VGG-style CNN (conv–BN–ReLU ×2 + max-pool per stage; BN `ioffe2015bn`). The first stage runs at full resolution before any pooling so single-pixel cues survive. The pooled feature is **[global average ‖ global max]**: average pooling captures global blur statistics, max pooling captures sparse extreme evidence.
- **Expected difficulty:** mild blur (k = 3, σ ≈ 0.5–0.7) is close to clean, so most confusion should be between clean and low-severity blur. The per-severity accuracy table (`classifier_accuracy_by_level.csv`) tests this.
- **Optimiser:** AdamW `loshchilov2019adamw`, because the PDF asks to tune weight decay, and decoupled decay makes that parameter mean the same thing at every lr.
- **Selection objective:** 1 − macro-F1 on the balanced validation manifest.

### 3.2 Specialists
- Same DAE architecture. The PDF allows a shared Optuna search (p.5): every trial trains all three specialists with identical hyperparameters, and the objective is the mean validation J of the three. The final three are then trained independently.
- Each specialist is trained and validated **only** on its own corruption (`allowed_types=[type]`).
- The same random initialisation (seed 42) is used for all three, so any difference between the experts comes from their training data alone (a controlled comparison). Their parameters are nonetheless independent after training.

### 3.3 Hard routing, oracle vs. predicted
- Clean → identity bypass: no expert is executed, and the output is the input bit-exactly (`hard_route`, `test_hard_route_identity_and_experts`).
- **Oracle mode** uses the manifest label; **predicted mode** uses the classifier argmax on the same records.
- A "classifier-caused failure" is a record where predicted ≠ true **and** J(predicted) − J(oracle) > 0.01. These records are listed in `classifier_caused_failures.csv` and visualised in `routing_failures.png`.

---

## 4. Task 3 — soft mixture of experts

### 4.1 Mixture formulation
- **Evidence:** mixtures of experts with a softmax gate (Jacobs et al. `jacobs1991moe`), and sparse gating at scale (`shazeer2017moe`, `fedus2022switch`, `lepikhin2021gshard`).
- **Decision:** dense 4-branch soft routing, exactly as in the PDF: w = softmax(G(x̃)/τ), x̂ = Σ w_k·branch_k. With only 4 branches, dense routing is cheap and fully differentiable, so sparse top-k routing (and its load/capacity machinery) is unnecessary.
- **Temperature:** τ as in knowledge distillation `hinton2015distill`. Small τ approaches hard routing; large τ approaches averaging. τ is a fixed hyperparameter tuned by Optuna, stored as a buffer and baked into the ONNX graph.

### 4.2 Initialisation and training schedule
- The gate is the Task 2 classifier (identical class order = branch order); the experts are the Task 2 specialists. The initial validation J of the *untrained* combination is logged (`initial` in `train_summary.json`) as the Task 2 → Task 3 starting point.
- **Stage 1 (warm-up):** experts frozen (parameters *and* BatchNorm statistics), gate trained with lr_gate = 1e-4.
- **Stage 2 (joint):** everything unfrozen, lr_joint ∈ [1e-6, 5e-5] (smaller than lr_gate and the specialists' lr), as the PDF requires.
- **Expert BatchNorm:** statistics stay frozen during the joint stage (`freeze_expert_bn`). In the MoE every expert sees *all four* conditions. Updating its running statistics with inputs it never specialised on would shift its normalisation, while fine-tuning only the weights at a small lr keeps the specialist behaviour. This is standard fine-tuning practice.

### 4.3 Where does CE apply?
- CE is computed on the **untempered** logits G(x̃), so the gate stays a calibrated classifier. τ only controls how that belief becomes mixture weights; it cannot be used to game the CE term.

### 4.4 Balance regulariser — baseline vs. alternative
- **Baseline (PDF):** L = Σ_k (w̄_k − 1/4)².
  - For a dense softmax, the batch-mean weights always sum to 1, so their mean is 1/4. Shazeer's importance loss (`shazeer2017moe`) CV(w̄)² = Var(w̄)/mean² then equals **4·Σ(w̄_k − 1/4)²**.
  - So the PDF baseline *is* the importance loss up to a constant (derivation checked in `test_squared_balance_equals_scaled_cv2`).
  - The Switch Transformer loss N·Σ f_i P_i (`fedus2022switch`) needs a hard assignment f_i and a capacity limit, neither of which exists in a dense 4-branch mixture.
- **Alternative (implemented):** marginal-entropy balance L = log 4 − H(w̄).
  - Like the baseline, it is minimised by uniform *batch* usage and does not penalise sharp *per-sample* routing (a one-hot but balanced batch has zero loss for both).
  - Its gradient grows like −log w̄_k as a branch dies out, so it resists collapse more strongly than the quadratic, whose gradient vanishes near w̄_k = 0.
  - Entropy-based confidence regularisation: `pereyra2017confidence`.
- **How the comparison is made:** the balance type is itself a categorical Optuna parameter, so TPE compares the two under otherwise tuned settings. The final model is then retrained with the *other* balance type (`--tag balance_ablation`) and both are evaluated on the test manifest.
- **Why balance on balanced batches is correct:** each batch holds 1/4 of each true class, so the ideal *average* usage really is 1/4 per branch. Correct routing and the balance term do not conflict.

### 4.5 Collapse detection / pruning
- After each validation epoch we compute the mean weight per (true class, branch). A branch is **inactive** if its mean weight is < 0.05 for every class; a branch **dominates unrelated inputs** if it takes > 0.6 mean weight on another class.
- Trials showing either during the joint stage are pruned (the PDF allows this, p.6); the reason is stored as a trial user attribute.
- The final test analysis reports the same diagnostics with a stricter 0.3 dominance threshold (`routing_diagnostics.json`).

### 4.6 ONNX export of the complete pipeline
- The forward pass is static (gate, 3 experts, softmax, weighted sum) with no data-dependent control flow, so the complete MoE exports as one graph with outputs (restored, weights). No architectural change was needed.
- Hard routing (Task 2) *does* have data-dependent control flow. It is deployed as 4 separate models plus an argmax in the backend, which also truly skips the experts for clean inputs.

---

## 5. Task 4 — style-conditioned pix2pix

### 5.1 Base model
- pix2pix `isola2017pix2pix`: a U-Net generator (`ronneberger2015unet`) plus a 70×70 PatchGAN discriminator; objective L_cGAN + λ·L1 with λ = 100; Adam (lr 2e-4, β1 = 0.5); dropout in the first three decoder blocks. Conditional GANs: `mirza2014cgan`, `goodfellow2014gan`; DCGAN-style initialisation N(0, 0.02) `radford2016dcgan`.
- **Adapted to 128 px:** 7 down-samplings (128 → 1). The 70×70 PatchGAN yields 14×14 logits.
- **Normalisation:** InstanceNorm `ulyanov2016instancenorm` instead of BatchNorm. pix2pix used batch size 1, where BN ≈ IN; with batch 4–16, BN's train/test statistics mismatch would appear in the deployed eval-mode generator.

### 5.2 Style conditioning — alternatives
| Option | Where | Pros | Cons |
|---|---|---|---|
| **Spatial replication + concatenation** (chosen) | G input + G bottleneck; D input | simple, used by cGANs `mirza2014cgan` and StarGAN label maps `choi2018stargan`; easy to verify; ONNX-friendly | conditioning signal can be diluted in deep layers (mitigated by the bottleneck injection) |
| Conditional normalisation / FiLM | every norm layer | strong, layer-wise control (`devries2017cbn`, `perez2018film`) | more parameters and code; harder to explain/verify |
| Projection discriminator | D only | principled for class-conditional D (`miyato2018projection`) | conditions D only; G still needs another mechanism |

- **Decision:** a learned `nn.Embedding(3, d)` in **each** network (G and D have their own), tiled and concatenated. In G the embedding enters at the input *and* at the 1×1 bottleneck, so the style reaches both local and global features. d is tuned by Optuna ∈ {4, 8, 16, 32}.
- **Verification:**
  - Unit test: changing the style changes the outputs of both G and D, and both embeddings are trainable (`test_style_embedding_influences_generator_and_discriminator`).
  - Test-time evidence: mean |G(x,a) − G(x,b)| across styles (`test_summary.json → style_influence`) and the style-swap grid.

### 5.3 Output channels
- **Evidence** (`scripts/prepare_fs2k.py` inspection): FS2K sketches are stored as RGB, but all three channels are identical (mean channel difference 0.00 in every subset). The generator therefore outputs 1 channel. A 3-channel output would triple the L1 target with redundant copies.

### 5.4 Paired augmentation
- The PDF requires identical spatial transforms (p.8).
- `PairedTransform` samples crop offset, flip and optional rotation **once** and applies them to both images: resize to 143, random crop 128, flip p = 0.5 (pix2pix "jitter", scaled from 286→256).
- **Verification:** `tests/test_paired_augmentation.py` shows pixel-identical geometry between photo and sketch, and a negative control proves that independent transforms break it.

### 5.5 Dataset observations that matter for the analysis
- **Official split:** 1,058 train / 1,046 test. Validation is 15 % stratified by style, giving 899 train / 159 val (styles train 303/298/298).
- The official **test** set is style-imbalanced (Style 1: 619, Style 2: 381, Style 3: 46), so the overall test mean is dominated by Styles 1–2. We therefore report per style.
- Style is **confounded with the photo source**:
  - photo2 (actors, 223×318 px) is entirely Style 3.
  - photo1 (CASIA-WebFace, 250×250) is Styles 1/2 (one Style 3 image).
  - photo3 (stock, 475×340) is Styles 1/3.
- The style embedding may therefore partly encode photo domain. This is a limitation to discuss, visible in the style-swap grid.

### 5.6 Metrics
- **Assignment-required:** the "validation measurements" of the generator. We use the paired L1 (the reconstruction term the PDF defines), on validation during training and on test at the end.
- **Additional, research-based (clearly separated in tables):**
  - SSIM `wang2004ssim` and PSNR: paired structural/pixel fidelity.
  - LPIPS `zhang2018lpips` (optional flag `--lpips`): perceptual distance that correlates better with human judgement for generated images.
- **Not used:** FID `heusel2017fid`. It is biased for small sample sizes (the authors use ≥ 10k samples; FS2K test has 1,046), and Inception features are not designed for grayscale line drawings.
- **GAN stability measures:**
  - The losses are logged separately (D-real, D-fake, G-adv, G-L1), plus D's mean probability on real/fake, to detect D over-powering (`salimans2016improved`).
  - Separate G/D learning rates are tuned (two-time-scale idea, `heusel2017fid`).
  - pix2pix-style linear lr decay over the second half of training.
- **Optuna objective:** J = 0.5·L1 + 0.5·(1 − SSIM) on validation, trained with a 25-epoch schedule (the PDF allows shorter trials). The best configuration is retrained for 200 epochs.
- **Known bias:** L1-based selection favours large λ_L1 (blurrier but "closer" outputs), so the selected λ is interpreted together with the qualitative grids.

---

## 6. Engineering / deployment decisions

| Decision | Alternatives | Reason |
|---|---|---|
| **MLflow** (SQLite backend, local artifacts) | Weights & Biases | Runs fully offline with no account or API key; an evaluator can browse every run with `mlflow ui --backend-store-uri sqlite:///mlflow.db` (`mlflow`) |
| **Optuna studies in SQLite** (`optuna_studies/*.db`) | in-memory, CSV only | Resumable and committed to Git (required deliverable); trials also exported to CSV and plots |
| **ONNX via `torch.onnx.export(dynamo=True)`**, opset 18, dynamic batch, single file | legacy TorchScript exporter; TorchScript runtime | The torch.export-based exporter is the default since PyTorch 2.9 (`pytorchonnx`); legacy exporter kept as fallback. The backend needs only ONNX Runtime (`onnxruntime`) — no PyTorch in the container |
| **ONNX tolerance** atol 1e-4, rtol 1e-3 on real validation inputs plus random inputs, at a batch size different from the export batch | exact equality | fp32 kernels differ across runtimes (~1e-7 – 1e-5 observed in smoke tests); 1e-4 is far below one 8-bit grey level (3.9e-3) |
| Backend reuses `src/corruptions` (NumPy only) | re-implement in JS | identical corruption code in training, tests and the app; verified torch-free by `test_backend_isolation.py` |
| Images returned as base64 PNG data URLs inside JSON | multipart/binary responses | one response carries images + probabilities + timing; documented in `backend/app/main.py` |
| **GitHub Releases** for model files | Git LFS, Hugging Face | no LFS bandwidth quota; `scripts/download_models.*` fetch and unzip |
| Mixed precision (fp16 autocast + GradScaler) for Tasks 1–3 | fp32 | ~2× faster on an RTX 2060 (`micikevicius2018amp`); losses/metrics computed in fp32. GAN kept in fp32 for stability |

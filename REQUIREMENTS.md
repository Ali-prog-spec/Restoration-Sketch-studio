# REQUIREMENTS — Generative AI Assignment #1

Source of truth: `GenAI_Assignment#1.pdf` (9 pages). Every item cites the page (p.N) where it appears.
Items marked *(derived)* are not literal PDF text but are direct consequences of it; items marked
*(student prompt)* come from the implementation brief, not the PDF.

Classification tags: **[CODE] [TRAINING] [EXPERIMENT] [RESEARCH] [REPORT] [MANUAL] [DEPLOYMENT]**

Status vocabulary (tracked in `PROGRESS.md` and `docs/final_compliance_checklist.md`):
NOT STARTED · IMPLEMENTED · TESTED · EXPERIMENTALLY VERIFIED · MANUAL ACTION REQUIRED · BLOCKED

> Note: the PDF header states "Deadline: March 16, 2024". Submission is via Google Classroom (p.1).

---

## A. Global requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| A1 | Design, train, evaluate and deploy four related generative AI systems (3 restoration tasks + 1 cGAN task). | p.1 | [CODE][TRAINING] |
| A2 | Research component: independently investigate concepts/techniques/tools not fully covered in lectures; consult credible papers, official docs, open-source implementations. | p.1 | [RESEARCH] |
| A3 | Investigate alternative architectures, loss functions, training strategies, hyperparameter ranges, evaluation methods and deployment approaches **before** final design decisions. | p.1 | [RESEARCH] |
| A4 | Every important technical decision must be supported by evidence; report explains alternatives investigated, why selected, difficulties encountered, how research/experiments resolved them. | p.1 | [RESEARCH][REPORT] |
| A5 | Simply reproducing an existing implementation or accepting AI code without investigation does not satisfy the research component. | p.1 | [RESEARCH] |
| A6 | Individual work. AI assistants/libraries/open-source allowed, but student must verify all generated info and code, test every component, acknowledge reused material, and understand the full system. | p.1 | [MANUAL] |
| A7 | Student may be asked during evaluation to justify architecture, explain a research decision, interpret a result, **modify part of the implementation**, or run the system on unseen images. → code must be readable and modifiable. | p.1 | [MANUAL][CODE] |
| A8 | Not complete if models can only run from a notebook/CLI/dev environment. All four tasks must be integrated into a browser-based (or mobile) application where the evaluator provides input, runs the selected model, inspects output and observes relevant system information. | p.1 | [CODE][DEPLOYMENT] |
| A9 | Implement and train with PyTorch or TensorFlow (or other suitable language). | p.2 | [CODE] |
| A10 | Submit through Google Classroom before the deadline; late submissions not accepted. | p.1 | [MANUAL] |

## B. Dataset requirements

### B.1 Oxford-IIIT Pet (Tasks 1–3)

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| B1 | Use Oxford-IIIT Pet Dataset (37 cat/dog categories). Category labels not required; original images are clean targets. | p.2 | [CODE] |
| B2 | Official training+validation collection (`trainval.txt`) = development data. | p.2 | [CODE] |
| B3 | Split development data 80% train / 20% validation with **random seed 42**. | p.2 | [CODE] |
| B4 | Official test set untouched until final evaluation (no tuning/model selection on it). | p.2–3 | [CODE][EXPERIMENT] |
| B5 | Convert all images to RGB and resize to 128×128. | p.3 | [CODE] |
| B6 | Same split used throughout Tasks 1, 2 and 3. | p.3 | [CODE] |
| B7 | Corrupted inputs generated programmatically. During training, corruption applied **at runtime inside the data-loading pipeline**; new type + severity sampled every time an image is loaded. | p.3 | [CODE] |
| B8 | Do not permanently save thousands of corrupted copies. | p.3 | [CODE] |
| B9 | For every clean training image the loader selects one of four conditions **with equal probability**: clean, salt-and-pepper, Gaussian blur, rectangular occlusion. | p.3 | [CODE] |
| B10 | The selected corruption label is known at generation time and used later to train the classifier. | p.3 | [CODE] |
| B11 | Clean: original image, no artificial corruption. | p.3 | [CODE] |
| B12 | Salt-and-pepper: probability p ~ U(0.02, 0.15); selected pixels replaced with black or white with equal probability. | p.3 | [CODE] |
| B13 | Gaussian blur: kernel size ∈ {3, 5, 7}; σ ~ U(0.5, 2.5). | p.3 | [CODE] |
| B14 | Occlusion: 1–3 black rectangular masks jointly covering 10%–35% of image area; random locations. | p.3 | [CODE] |
| B15 | Validation and test corruptions are **deterministic**. Generate a validation manifest and a test manifest **once**, storing corruption type, severity, mask coordinates, blur settings and random seed for every image. | p.3 | [CODE] |
| B16 | Final test: every clean test image gets **three fixed severity levels for every corruption**. | p.3 | [CODE][EXPERIMENT] |
| B17 | Test salt-and-pepper p ∈ {0.03, 0.08, 0.15}. | p.3 | [CODE] |
| B18 | Test blur (k, σ) ∈ {(3, 0.7), (5, 1.5), (7, 2.5)}. | p.3 | [CODE] |
| B19 | Test occlusion ≈10% with 1 rectangle, ≈20% with 2, ≈35% with 3. | p.3 | [CODE] |
| B20 | *(derived)* Test clean images are also evaluated (Task 1 requires "clean" results; Task 2 routing needs clean inputs). | p.4 | [CODE] |
| B21 | *(derived)* Validation manifest mirrors the training distribution (equal class probability) so validation is comparable across Tasks 1–3. | p.3 | [CODE] |

### B.2 FS2K (Task 4)

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| B30 | Use FS2K Facial Sketch Synthesis Dataset (2,104 photo–sketch pairs, 3 sketch styles). | p.7 | [CODE] |
| B31 | Use official FS2K train and test definitions. | p.7 | [CODE] |
| B32 | Reserve 15% of the official training portion as validation, seed 42, **stratified by sketch style**. | p.7 | [CODE] |
| B33 | Official test set not used during training or hyperparameter selection. | p.7 | [CODE][EXPERIMENT] |
| B34 | Resize photos and sketches to 128×128; preserve exact photo↔sketch pairing. | p.7 | [CODE] |
| B35 | Any spatial augmentation (crop, flip, rotation, resize) applied **identically** to both members of a pair. | p.8 | [CODE] |

## C. Task 1 — Universal Multi-Corruption Denoising Autoencoder

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| C1 | Single universal DAE that handles clean + all 3 corruptions; model is **not told** which corruption was applied. | p.3 | [CODE] |
| C2 | Convolutional encoder, **genuine compressed latent**, convolutional decoder. | p.3 | [CODE] |
| C3 | Encoder progressively reduces spatial size while increasing channels. | p.3 | [CODE] |
| C4 | Decoder reconstructs an RGB image at original resolution (128×128×3). | p.3 | [CODE] |
| C5 | Meaningful bottleneck; copying input through unrestricted skip connections does **not** satisfy the requirement. | p.3–4 | [CODE] |
| C6 | If limited skip connections are used, their purpose and effect must be investigated and justified in the report. | p.4 | [RESEARCH][EXPERIMENT][REPORT] |
| C7 | x̂ = D(E(x̃)). | p.4 | [CODE] |
| C8 | Loss L = α·L1(x, x̂) + (1−α)·(1 − SSIM(x, x̂)). | p.4 | [CODE] |
| C9 | α = 0.8 is only an initial value; final α selected via Optuna. | p.4 | [EXPERIMENT] |
| C10 | Optuna investigates at least: learning rate, batch size, bottleneck dimension, number of encoder channels, dropout rate, α. | p.4 | [EXPERIMENT] |
| C11 | Validation objective combines reconstruction quality and structural similarity. | p.4 | [CODE] |
| C12 | Report: complete search space, number of completed trials, best trial, final selected configuration. | p.4 | [REPORT] |
| C13 | Final results separately for clean / salt-and-pepper / blur / occlusion. | p.4 | [EXPERIMENT][REPORT] |
| C14 | Results separated by low / medium / high severity. | p.4 | [EXPERIMENT][REPORT] |
| C15 | Visuals: clean target, corrupted input, reconstruction, absolute error map. | p.4 | [EXPERIMENT][REPORT] |
| C16 | Discuss ≥ 12 representative examples and ≥ 4 meaningful failure cases. | p.4 | [EXPERIMENT][REPORT] |
| C17 | Export final model to ONNX. | p.4 | [CODE][DEPLOYMENT] |
| C18 | App workspace named **"Universal Restoration"**. | p.4 | [CODE] |
| C19 | User can upload a corrupted image **or** select a clean sample and apply one of the corruptions in the UI. | p.4 | [CODE] |
| C20 | Display input, restored output, selected corruption settings, inference time. | p.4 | [CODE] |

## D. Task 2 — Corruption classifier + hard-routed specialists

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| D1 | Convolutional classifier with 4 classes: clean, salt-and-pepper, blur, occlusion. | p.4 | [CODE] |
| D2 | Training labels from the runtime corruption pipeline. | p.4 | [CODE] |
| D3 | Training batches balanced across classes. | p.4 | [CODE] |
| D4 | Output p = C(x̃) = [p_clean, p_salt, p_blur, p_occlusion]; prediction r = argmax_k p_k. | p.4–5 | [CODE] |
| D5 | Multiclass cross-entropy loss. | p.5 | [CODE] |
| D6 | Optuna tunes: learning rate, batch size, convolutional channel configuration, dropout, weight decay. | p.5 | [EXPERIMENT] |
| D7 | Report overall accuracy, macro precision/recall/F1, per-class metrics, normalized 4×4 confusion matrix. | p.5 | [EXPERIMENT][REPORT] |
| D8 | Three specialist DAEs: salt-and-pepper, blur, occlusion. | p.5 | [CODE][TRAINING] |
| D9 | Each specialist trained **only** on its own corruption type; clean image is the target. | p.5 | [CODE] |
| D10 | Same basic architecture allowed, but **independently trained parameters**. | p.5 | [TRAINING] |
| D11 | Optuna applied to specialists; shared Optuna search allowed to find a common architecture, then train three independently. | p.5 | [EXPERIMENT] |
| D12 | Tune learning rate, bottleneck size, channel configuration, batch size, L1-to-SSIM weighting. | p.5 | [EXPERIMENT] |
| D13 | Hard-routed inference: x̂ = x̃ if r=clean; A_salt(x̃) if salt; A_blur(x̃) if blur; A_occ(x̃) if occlusion. | p.5 | [CODE] |
| D14 | Clean input uses **identity bypass**, not processed by an expert. | p.5 | [CODE] |
| D15 | Test in **oracle-routing** mode (label from deterministic test manifest). | p.5 | [EXPERIMENT] |
| D16 | Test in **predicted-routing** mode (classifier prediction). | p.5 | [EXPERIMENT] |
| D17 | Identify and discuss cases where classifier errors cause restoration failures. | p.5 | [EXPERIMENT][REPORT] |
| D18 | App workspace named **"Hard-Routed Restoration"**: shows 4 classifier probabilities, predicted corruption, selected expert, reconstruction, inference time. | p.5 | [CODE] |
| D19 | Export classifier and all 3 specialists to ONNX. | p.5 | [CODE][DEPLOYMENT] |

## E. Task 3 — Jointly trained soft Mixture-of-Experts

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| E1 | Differentiable soft MoE converted from the hard-routing system. | p.5 | [CODE] |
| E2 | Contains a gating network, the 3 Task-2 specialists, and an **identity branch** for clean images. | p.6 | [CODE] |
| E3 | Gate assigns a continuous weight to every branch: w = softmax(G(x̃)/τ), w = [w0 clean, w1 salt, w2 blur, w3 occ]. | p.6 | [CODE] |
| E4 | τ controls sharp vs distributed routing. | p.6 | [CODE][EXPERIMENT] |
| E5 | x̂ = w0·x̃ + w1·A_salt(x̃) + w2·A_blur(x̃) + w3·A_occ(x̃). | p.6 | [CODE] |
| E6 | Gate and experts trainable jointly via the final reconstruction error. | p.6 | [CODE] |
| E7 | Must **not** start from random components: gate ← Task 2 classifier, experts ← Task 2 specialists. | p.6 | [CODE][TRAINING] |
| E8 | Short warm-up: experts frozen, only gate trained. | p.6 | [TRAINING] |
| E9 | Then unfreeze experts and jointly fine-tune complete system with a **smaller** learning rate. | p.6 | [TRAINING] |
| E10 | L_MoE = λ1·L1 + λs·(1−SSIM) + λc·L_CE + λb·L_balance. | p.6 | [CODE] |
| E11 | CE keeps gate related to the known runtime label. | p.6 | [CODE] |
| E12 | Balance loss prevents sending nearly all inputs to one expert. Baseline: L_balance = Σ_k (w̄_k − 1/4)², w̄_k = mean routing weight of branch k within a **balanced training batch**. | p.6 | [CODE] |
| E13 | A different differentiable balance/entropy regularizer may be proposed but must be research-supported and justified. | p.6 | [RESEARCH][REPORT] |
| E14 | Initial values λ1=0.8, λs=0.2, λc=0.1, λb=0.01. | p.6 | [CODE] |
| E15 | Optuna investigates: joint fine-tuning LR, τ, classification weight, balance weight, reconstruction-loss weighting. | p.6 | [EXPERIMENT] |
| E16 | Trial pruning may be used for poor configurations or routing collapse. | p.6 | [CODE][EXPERIMENT] |
| E17 | Report average expert weights per true corruption type **and** severity level. | p.7 | [EXPERIMENT][REPORT] |
| E18 | Show examples where one expert dominates and where weights are distributed. | p.7 | [EXPERIMENT][REPORT] |
| E19 | Routing heatmap or weight-distribution diagram in the report. | p.7 | [REPORT] |
| E20 | Determine whether any expert became inactive or one expert dominates unrelated inputs. | p.7 | [EXPERIMENT][REPORT] |
| E21 | App workspace named **"Soft Mixture-of-Experts Restoration"**: show all 4 routing weights, reconstruction, inference time, visual indication of strongest contributors. | p.7 | [CODE] |
| E22 | **Complete** soft-MoE inference pipeline exported to ONNX. | p.7 | [CODE][DEPLOYMENT] |

## F. Task 4 — Style-conditioned face-to-sketch cGAN

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| F1 | Conditional GAN for face→sketch on FS2K. | p.7 | [CODE] |
| F2 | Generator: U-Net-style encoder-decoder; inputs photo + style condition. | p.7 | [CODE] |
| F3 | Discriminator: PatchGAN judging local regions of photo–sketch pairs. | p.7 | [CODE] |
| F4 | ŷ = G(x, s). | p.7 | [CODE] |
| F5 | Style represented by a **learned categorical embedding** for the 3 styles. | p.7 | [CODE] |
| F6 | Embedding incorporated into **both** generator and discriminator (not only an interface label). | p.7 | [CODE] |
| F7 | D receives photo, style, and real or generated sketch: D(x, y, s), D(x, G(x,s), s). | p.7 | [CODE] |
| F8 | D classifies real pairs real and generated fake; G fools D while staying close to ground truth. | p.7 | [CODE] |
| F9 | L_G = L_adv + λ_L1·L1(y, G(x,s)). | p.7 | [CODE] |
| F10 | BCE-with-logits may be used for D and adversarial terms. | p.8 | [CODE] |
| F11 | λ_L1 = 100 initial value; final value investigated with Optuna. | p.8 | [EXPERIMENT] |
| F12 | Optuna tunes at least: G LR, D LR, batch size, base channels, dropout, style-embedding dim, reconstruction weight. | p.8 | [EXPERIMENT] |
| F13 | Optuna trials may use fewer epochs; selected configuration **retrained for full schedule**. | p.8 | [TRAINING] |
| F14 | Paired augmentation identical for photo and sketch. | p.8 | [CODE] |
| F15 | Record separately: D real loss, D fake loss, G adversarial loss, G reconstruction loss, validation measurements. | p.8 | [CODE][EXPERIMENT] |
| F16 | Log generated samples at fixed intervals using the **same** validation photos. | p.8 | [CODE][EXPERIMENT] |
| F17 | App workspace named **"Face-to-Sketch Generator"**: upload photo **or webcam capture**, choose Style 1/2/3, generate, show original and sketch side-by-side, **download** result. | p.8 | [CODE] |
| F18 | Three fixed style names suffice; natural-language prompting not required. | p.8 | — |
| F19 | Export final generator to ONNX; discriminator is training-only. | p.8 | [CODE][DEPLOYMENT] |

## G. Optuna requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| G1 | Optuna must be used for hyperparameter optimization in **each of the four tasks**. | p.2 | [EXPERIMENT] |
| G2 | Optuna studies included in the repository. | p.2 | [CODE][EXPERIMENT] |
| G3 | Task-specific search variables: see C10, D6, D12, E15, F12. | p.4–8 | [EXPERIMENT] |
| G4 | Report search space, #completed trials, best trial, final configuration (explicit for Task 1; applied to all tasks for consistency). | p.4, p.9 | [REPORT] |
| G5 | Report must include Optuna results. | p.9 | [REPORT] |

## H. Experiment tracking

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| H1 | Use MLflow or Weights & Biases. | p.2 | [CODE] |
| H2 | Record important training experiments, hyperparameters, losses, evaluation results, checkpoints, visual outputs. | p.2 | [EXPERIMENT] |
| H3 | Demo video shows experiment-tracking records. | p.2 | [MANUAL] |

## I. Evaluation requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| I1 | Task 1 per-corruption and per-severity results (C13–C14). | p.4 | [EXPERIMENT] |
| I2 | Task 2 classifier metrics (D7); oracle vs predicted routing (D15–D17). | p.5 | [EXPERIMENT] |
| I3 | Task 3 reconstruction results + gating behaviour analysis (E17–E20). | p.7 | [EXPERIMENT] |
| I4 | Task 4 validation measurements (F15); test only after model selection (B33). | p.7–8 | [EXPERIMENT] |
| I5 | Visual outputs, error maps, failure cases for all tasks. | p.1–2, p.9 | [EXPERIMENT][REPORT] |
| I6 | *(derived)* Restoration metrics: SSIM is named in the PDF; L1 is named; PSNR added as a standard restoration metric (justified in research notes). | p.4 | [RESEARCH] |

## J. ONNX requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| J1 | Export all models required for inference to ONNX: Task 1 DAE; Task 2 classifier + 3 specialists; Task 3 complete MoE pipeline; Task 4 generator. | p.2, 4, 5, 7, 8 | [CODE] |
| J2 | Verify ONNX outputs are consistent with PyTorch outputs. | p.2 | [CODE][EXPERIMENT] |
| J3 | Include the ONNX models or provide valid download links. | p.2 | [DEPLOYMENT][MANUAL] |

## K. Application requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| K1 | One coherent browser-based application with four clearly accessible workspaces. | p.1, 2, 8 | [CODE] |
| K2 | Visual design, page structure, colours, controls, cards, image panels, responsive layout **first developed in Google Stitch**. | p.2, 8 | [MANUAL] |
| K3 | Evidence of original Stitch design in the report. | p.2, 8 | [MANUAL][REPORT] |
| K4 | Frontend: React + Tailwind CSS. | p.2 | [CODE] |
| K5 | Backend: FastAPI (or similar); frontend communicates with it. | p.2, 8 | [CODE] |
| K6 | Backend validates uploads, preprocesses images, loads ONNX models, runs inference, returns results with routing and timing info. | p.8 | [CODE] |
| K7 | Backend operations at minimum: health-check, universal-restoration, hard-routing, soft-mixture, face-to-sketch. | p.8 | [CODE] |
| K8 | Evaluator can select corruption types and severity levels, upload an already corrupted image, inspect classifier and mixture weights, generate sketches, restart the app. | p.8 | [CODE] |
| K9 | Must work on previously unseen images (arbitrary sizes/formats → preprocessing). | p.1, 2, 8 | [CODE] |
| K10 | Workspace-specific displays: C18–C20, D18, E21, F17. | p.4–8 | [CODE] |

## L. Docker / deployment requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| L1 | Frontend and backend run inside Docker containers. | p.2 | [DEPLOYMENT] |
| L2 | Docker Compose starts the complete application with **one documented command**. | p.2 | [DEPLOYMENT] |
| L3 | Evaluator: clone repo → obtain model files → start containers → open browser; no VS Code, no manually running separate Python scripts. | p.2 | [DEPLOYMENT] |
| L4 | Local containerized deployment compulsory; public hosting optional (if used, must stay accessible during assessment). | p.8 | [DEPLOYMENT] |

## M. Technical report requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| M1 | IEEE research-paper format, written in LaTeX. | p.1 | [REPORT] |
| M2 | Discuss each task separately: architecture, loss, training procedure, hyperparameter optimization, results, visual outputs, failure cases, application design, conclusions. | p.1 | [REPORT] |
| M3 | Include (where relevant) architecture diagrams, training graphs, confusion matrices, routing visualizations, generated images, error maps, result tables, application screenshots. | p.1–2 | [REPORT] |
| M4 | Include GitHub repository URL. | p.2 | [REPORT][MANUAL] |
| M5 | Include only the YouTube link to the demo video. | p.2 | [REPORT][MANUAL] |
| M6 | Sections: concise introduction, related research, dataset preparation, architecture design, loss functions, training procedure, Optuna search design, experimental setup, results, analysis, application architecture, limitations, conclusion. | p.9 | [REPORT] |
| M7 | Each task has a clearly identifiable methodology and results discussion. | p.9 | [REPORT] |
| M8 | Include complete corruption configuration, training and validation curves, Optuna results, confusion matrices, quantitative tables, routing-weight visualizations, generated-image grids, error maps, app screenshots, meaningful failure cases. | p.9 | [REPORT] |
| M9 | Not only screenshots/raw output: every important table/diagram/image interpreted in text (what it shows, why failures occurred, how evidence influenced conclusions). | p.9 | [REPORT] |
| M10 | Explain alternatives investigated, difficulties encountered, how research/experiments addressed them (A4). | p.1 | [REPORT][RESEARCH] |
| M11 | Evidence of Google Stitch design (K3). | p.2, 8 | [REPORT][MANUAL] |

## N. Demo / video requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| N1 | 5–7 minute demonstration video uploaded to YouTube; only the link in the report; do not upload video to Google Classroom. | p.2 | [MANUAL] |
| N2 | Video shows: application startup, all four tasks working, image uploading, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch generation, result downloading, experiment-tracking records. | p.2 | [MANUAL] |

## O. GitHub / repository requirements

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| O1 | Complete implementation in a GitHub repository. | p.2 | [MANUAL][CODE] |
| O2 | Repository contains: source code, configuration files, dependency file, data-preparation scripts, training scripts, evaluation scripts, Optuna studies, ONNX export code, application code, Dockerfiles, Docker Compose config, README with complete execution instructions. | p.2 | [CODE] |
| O3 | Do not upload complete datasets or very large model files; provide documented download link or Git LFS for trained models. | p.2 | [DEPLOYMENT][MANUAL] |

## P. AI-use appendix

| ID | Requirement | Page | Tag |
|----|-------------|------|-----|
| P1 | AI-use appendix identifying tools used, tasks used for, and how outputs were tested or corrected. | p.2 | [REPORT][MANUAL] |
| P2 | Acknowledge reused material. | p.1 | [REPORT] |

---

## Ambiguities found in the PDF and chosen interpretation

| # | Ambiguity | Interpretation (details in `docs/research_notes.md`) |
|---|-----------|------------------------------------------------------|
| Q1 | Occlusion training: is "10–35%" the union area or the sum of rectangle areas? | Use **union** area of the masks (what is actually hidden); sample a target coverage ~U(0.10, 0.35) and rejection-sample rectangles until the union is within ±1.5% of target and inside [10%, 35%]. |
| Q2 | Test occlusion "approximately" 10/20/35%. | Same generator with fixed target and number of rectangles; tolerance ±1.5 percentage points; actual coverage stored in manifest. |
| Q3 | Salt-and-pepper: per-pixel or per-channel? | Per **pixel** (all 3 channels set to 0 or 1 together) — matches "replace the selected pixels with black or white". |
| Q4 | Gaussian blur boundary handling. | Reflect padding (avoids dark borders that zero-padding would introduce). |
| Q5 | Validation manifest severity distribution. | Mirrors training distribution: each val image gets one condition (equal probability) with parameters sampled from the training ranges using a per-image seed. |
| Q6 | "Low/medium/high" for Task 1 metrics. | Defined by the three fixed test levels (B17–B19). Clean has no severity. |
| Q7 | "Balanced training batch" for classifier and MoE. | Each batch contains an equal number of samples of each class: every image in a batch of size B is assigned class `i mod 4` via a balanced batch sampler, so each batch contains exactly B/4 of each class. |
| Q8 | Task 3 gate architecture. | Gate = Task-2 classifier architecture, weights copied from the trained classifier (logits → /τ → softmax). |
| Q9 | Task 3 test inputs with multiple corruptions (motivation text p.5). | Primary evaluation on the official manifest; an **additional, clearly labelled** mixed-corruption stress test is provided as analysis only. |
| Q10 | FS2K "Style 1/2/3". | Map to FS2K style labels 0/1/2 (reported as Style 1/2/3). |
| Q11 | Task 4 metrics are not specified by the PDF. | Required: "validation measurements" → validation L1. Additional research-based: SSIM, PSNR, (optionally) LPIPS/FID — marked as additional. |

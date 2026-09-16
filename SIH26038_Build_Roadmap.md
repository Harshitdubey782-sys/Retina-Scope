# SIH26038 — Explainable AI for Diabetic Retinopathy Screening
## Complete Build Roadmap

**PS:** SIH26038 | **Org:** MathWorks | **Theme:** MedTech / BioTech / HealthTech
**Idea submission deadline:** 30 September 2026 | **Current submissions:** 1/500

---

## 1. The Goal, In One Line

Upload a retina photo → get back a DR severity grade (0–4) + a visual heatmap showing *why* the AI thinks so + a simple capacity-planning view for scaling this to a district. That's the whole prototype.

---

## 2. Two Phases You're Actually Building For

SIH has two checkpoints, and your roadmap needs to serve both:

| Phase | Deadline | What's needed |
|---|---|---|
| **A. Idea Submission (PPT)** | 30 Sept 2026 (~2 weeks away) | A convincing PPT + ideally a rough working demo/screenshot to prove feasibility |
| **B. Grand Finale Build** | 36-hour hackathon (date TBD after shortlisting) | The full working prototype, live demo, judging |

Don't wait until the 36-hour event to start — Phase A prep work (data download, baseline model) should happen *now*, so the 36 hours is spent polishing, not starting from zero.

---

## 3. Team Roles (6 members)

| Role | Responsibility | Best fit |
|---|---|---|
| **ML Lead** | Model training, severity classification | Strongest in Python/deep learning |
| **CV/Explainability Engineer** | Preprocessing, Grad-CAM, image quality checks | Comfortable with OpenCV/image pipelines |
| **Backend/Integration Dev** | Ties model + explainability + dashboard into one app | Python/Flask/FastAPI experience |
| **Frontend Dev** | Demo UI (upload → results screen) | Streamlit/React, whoever's fastest to ship UI |
| **Systems/Dashboard Dev** | The "Simulink-alternative" resource-planning calculator | Comfortable with basic simulation/math logic |
| **Research & Pitch Lead** | Clinical framing, benchmarks, PPT, judge Q&A prep | Strongest communicator, does literature check |

Everyone should be able to explain the whole pipeline end-to-end by the demo — judges ask random team members questions.

---

## 4. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python | Fastest path to a working model; MATLAB is a bonus, not mandatory |
| Model | PyTorch + `timm` (EfficientNet-B0 or ResNet50, pretrained) | Pretrained weights = no training from scratch |
| Explainability | `pytorch-grad-cam` (or `captum`) | Plug-and-play Grad-CAM, well documented |
| Preprocessing | OpenCV, `albumentations` | CLAHE, denoising, resizing in a few lines |
| Demo UI | Streamlit | Fastest way to get an interactive upload → result screen |
| Dashboard/calculator | Plotly / Streamlit + basic Python math | No real Simulink needed — see Phase 6 |
| Deployment (optional, for polish) | Streamlit Community Cloud / Hugging Face Spaces | Free, gives you a shareable live link for the PPT |

---

## 5. Phase A: Idea Submission Prep (Do This Now → Sept 30)

**Goal:** a PPT that shows you've already de-risked the hard parts.

- [ ] Download **APTOS 2019** (Kaggle) — main dataset for training/testing
- [ ] Skim **IDRiD** and **Messidor-2** — mention them as validation/generalization sets in your PPT (shows rigor)
- [ ] Train a **basic baseline classifier** (even 70–75% accuracy is fine at this stage) — screenshot it working
- [ ] Get Grad-CAM working on at least 2–3 sample images — this single visual is often the most convincing slide in the whole deck
- [ ] Write the PPT with this structure:
  1. Problem (use the "1 ophthalmologist per 100,000" stat — it's in the PS itself, judges will recognize you read it carefully)
  2. Your approach (pipeline diagram: image → quality check → segmentation → grading → explainability → dashboard)
  3. Early proof-of-concept screenshot (severity output + Grad-CAM heatmap)
  4. Tech stack & datasets you're using
  5. What you'll build further at the finale
  6. Future scope / scaling story (ABDM integration, edge deployment, etc.)

A team that shows even a rough working screenshot at idea stage looks dramatically more credible than one with only diagrams.

---

## 6. Phase B: The 36-Hour Build — Hour-by-Hour

### Block 1: Setup & Data (Hours 0–4)
- [ ] Re-confirm environment, GPU access (Colab/Kaggle notebooks as backup if no local GPU)
- [ ] Load APTOS dataset, do train/val split
- [ ] Build preprocessing pipeline: resize → CLAHE contrast enhancement → denoise
- [ ] Add a simple **image quality check** (Laplacian variance for blur detection is a 5-line fallback if time-pressed) → reject/flag bad images

### Block 2: Model Training (Hours 4–12)
- [ ] Fine-tune pretrained EfficientNet-B0 on 5-class severity labels (0–4)
- [ ] Track sensitivity/specificity specifically for **referable DR (Level 2+)** — the PS explicitly asks for >90% sensitivity, >85% specificity here, so report against that benchmark, not just raw accuracy
- [ ] Save best checkpoint; keep a simple confusion matrix for your pitch slide

### Block 3: Explainability Layer (Hours 12–18)
- [ ] Wire up Grad-CAM on top of the trained model
- [ ] Overlay heatmap on original image (side-by-side view)
- [ ] Add a **confidence score** output alongside the severity grade
- [ ] (Stretch goal) Auto-generate a 2-line text summary: *"Moderate NPDR detected — evidence concentrated in superior temporal quadrant, confidence 87%"* — this single feature reads as "clinically useful" to judges

### Block 4: The Resource-Planning Dashboard (Hours 18–24)
This replaces the Simulink requirement — you don't need Simulink itself, you need to **demonstrate the same thinking**:
- [ ] Build a simple calculator: inputs = number of devices, images/hour per device, review time per flagged case → outputs = patients screened/day, ophthalmologist-hours needed, expected bottlenecks
- [ ] Visualize with a basic bar/line chart (Plotly) — "at 50,000 patients/year, you need X reviewers"
- [ ] Frame this explicitly in your pitch as *"a simplified simulation of the Simulink-style resource model the PS calls for"* — judges care that you addressed the concept

### Block 5: Integration + Frontend (Hours 20–30, parallel with Block 4)
- [ ] Build Streamlit app: upload image → preprocessing → severity + confidence + Grad-CAM overlay → link to dashboard tab
- [ ] Add a "batch mode" toggle if time allows (upload 10 images, get a summary table) — shows district-scale thinking
- [ ] Test on images the model hasn't seen (from IDRiD/Messidor-2) — mention this cross-dataset test as your "generalization" claim

### Block 6: Validation & Benchmarking (Hours 30–33)
- [ ] Run final sensitivity/specificity numbers, compare against published DR-screening benchmarks (mention 1–2 papers by name in your slide, don't need full citations)
- [ ] Stress-test the demo: broken images, non-retina images, low-quality uploads — have the quality-check module catch these gracefully live in the demo (this specific moment often impresses judges more than raw accuracy numbers)

### Block 7: Pitch Prep (Hours 33–36)
- [ ] Build the final deck: Problem → Approach → Live demo → Benchmarks vs. clinical thresholds → Resource-planning dashboard → Future scope (ABDM, edge deployment, multi-disease)
- [ ] Assign Q&A prep: expect questions like *"How does this generalize to different camera types?"* and *"What happens with a borderline image?"* — have answers ready
- [ ] Rehearse the live demo at least twice, including a rehearsed fallback (recorded video) in case of live-demo failure

---

## 7. Risk Map — What Could Go Wrong, and the Fallback

| Risk | Fallback |
|---|---|
| Model accuracy too low | Focus pitch on the explainability + workflow story, not just raw numbers; 75–80% accuracy with strong Grad-CAM output still demos well |
| No MATLAB/Simulink experience | Explicitly frame the Python dashboard as an "MVP simulation" — reviewers care about concept coverage, not tool compliance |
| GPU/training time runs out | Pre-train before the event (Phase A) and fine-tune further during the 36 hours, don't start cold |
| Live demo breaks on stage | Have a 60-second screen-recorded backup video ready |
| Judges ask about clinical validation rigor | Be upfront: "this is a research prototype benchmarked against public datasets, not yet clinically certified" — honesty here reads better than overclaiming |

---

## 8. Mapping Back to What the PS Actually Asked For

Use this as a final checklist before submission — judges will literally re-read the PS and check for these:

- [x] Image quality assessment + enhancement → Block 1
- [x] Retinal structure segmentation (optic disc, vessels, lesions) → covered implicitly via CNN feature learning; explicitly call out in pitch that full pixel-level segmentation is a stretch/future-scope item if time-constrained
- [x] DR severity grading (0–4) with sensitivity/specificity targets → Block 2
- [x] Explainability module (Grad-CAM, confidence, reports) → Block 3
- [x] Simulink-style workflow simulation → Block 4 (Python equivalent)
- [x] Validation against benchmarks → Block 6

---

## 9. Key Resources

- Dataset: [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- Dataset: [IDRiD](https://ieeedataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid)
- Dataset: [DRIVE (vessel extraction)](https://drive.grand-challenge.org/)
- Dataset: [Messidor-2](https://www.adcis.net/en/third-party/messidor2/)
- Model library: `timm` (PyTorch Image Models)
- Explainability: `pytorch-grad-cam`
- Demo framework: Streamlit

---

### One-line summary of the whole roadmap
**Now → Sept 30:** get a baseline model + Grad-CAM screenshot working, submit a strong idea PPT.
**36-hour finale:** polish the pipeline, add the resource dashboard, integrate into one demo, benchmark, pitch.

RetinaScope 

AI-Powered Diabetic Retinopathy Screening & Healthcare Resource Planning
RetinaScope is an explainable AI-based screening platform designed to assist in the detection and assessment of Diabetic Retinopathy from retinal fundus images.
It combines **AI-based DR severity classification, Grad-CAM explainability, patient screening reports, dual-eye analysis, and district-level healthcare resource planning** into a unified workflow.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Key Features:-

AI-Based DR Screening:-
- EfficientNet-B0 based 5-class classification
- Predicts Diabetic Retinopathy severity from **Grade 0–4**
- Supports referable DR identification

Dual-Eye Screening:-
- Analyze left and right eye fundus images
- Compare results across both eyes
- Provides eye-specific predictions and explanations

Explainable AI:-
- Grad-CAM based visualization
- Highlights regions contributing to the model's prediction
- Helps make AI predictions more interpretable

Patient Screening Report:-
Captures relevant patient information including:
- Case ID
- Patient details
- Age & Sex
- Screening centre
- Diabetes information
- Previous DR history
- Eye examined
- AI screening result

District-Level Resource Planning:-
RetinaScope extends beyond individual screening by providing a planning layer using district-level data.

It estimates:
- Screening demand
- Available capacity
- Capacity gaps
- Priority levels

This can help visualize where additional screening resources may be required.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

AI Pipeline:=

Fundus Image
     ↓
Preprocessing
Resize + CLAHE + Denoising
     ↓
EfficientNet-B0
     ↓
DR Grade 0–4
     ↓
Referable DR Assessment
     ↓
Grad-CAM
     ↓
Explainable Screening Report
     ↓
District-Level Resource Planning

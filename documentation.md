# ✅ Summary of Tool Usage

| Preprocessing Step       | FSL       | HCP Pipelines | ANTs     | dcm2niix         |
|--------------------------|-----------|--------------|----------|-------------------|
| **DICOM to NIfTI**       | Partial   | ✅           | ❌       | ✅ (Best)         |
| **GradWarp**             | ❌        | ✅           | ❌       | ❌                |
| **B1 Correction**         | ❌        | ✅           | ✅       | ❌                |
| **Slice Timing**          | ✅        | ❌           | ❌       | ✅ (Metadata)     |
| **N3 Bias Correction**    | ✅        | ✅           | ✅       | ❌                |
| **Brain Extraction**      | ✅        | ✅           | ❌       | ❌                |
| **Motion Correction**     | ✅        | ✅           | ❌       | ❌                |
| **Spatial Smoothing**     | ✅        | ❌           | ❌       | ❌                |
| **Intensity Scaling**     | ✅        | ❌           | ❌       | ❌                |
| **Coregistration**        | ✅        | ✅           | ❌       | ❌                |
| **Normalization (MNI)**   | ✅        | ✅           | ✅       | ❌                |
| **Distortion Correction** | ✅        | ✅           | ❌       | ❌                |
| **High-pass Filtering**   | ✅        | ❌           | ❌       | ❌                |
| **ICA Denoising**         | ✅        | ❌           | ❌       | ❌                |

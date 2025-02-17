# ✅ Summary of Tool Usage (Updated with New Process Order, B1 Correction and GradWarp Marked as Pending)

| Preprocessing Step                               | FSL       | HCP Pipelines | ANTs     | dcm2niix | Implemented | Tool Used                |
|--------------------------------------------------|-----------|--------------|----------|----------|-------------|--------------------------|
| **DICOM to NIfTI**                               | Partial   | ✅           | ❌       | ✅ (Best) | ✅         | `dcm2niix`             |
| **Slice Timing**                                 | ✅        | ❌           | ❌       | ✅ (Metadata) | ✅         | `slicetimer`            |
| **Create Mean Image (Tmean)**                    | ✅    | ❌           | ❌       | ❌        | ✅         | `fslmaths -Tmean`       |
| **Brain Extraction (BET)**                       | ✅        | ✅           | ❌       | ❌        | ✅         | `bet`                   |
| **Intensity Scaling (After BET)**                | ✅  | ❌           | ❌       | ❌        | ✅         | `fslmaths -ing`         |
| **B1 Correction (Before Bias Field Correction)** | ❌ | ✅           | ✅       | ❌        | ❌ (Pending) | `HCP Pipelines`         |
| **Bias Field Correction (FAST)**                 | ✅   | ✅           | ✅       | ❌        | ✅         | `fast`                  |
| **GradWarp (Before Motion Correction)**          | ❌ | ✅           | ❌       | ❌        | ❌ (Pending) | `HCP Pipelines`         |
| **Motion Correction (MCFLIRT)**                  | ✅   | ✅           | ❌       | ❌        | ✅         | `mcflirt`               |
| **Spatial Smoothing**                            | ✅        | ❌           | ❌       | ❌        | ✅         | `fslmaths -s`           |
| **Coregistration (FLIRT)**                       | ✅       | ✅           | ❌       | ❌        | ✅         | `flirt`                 |
| **Normalization (MNI, FNIRT)**                   | ✅    | ✅           | ✅       | ❌        | ✅         | `fnirt`                 |
| **High-pass Filtering** (optional)               | ✅        | ❌           | ❌       | ❌        | ✅         | `fslmaths -bptf`        |
| **ICA Denoising (MELODIC)** (optional)           | ✅     | ❌           | ❌       | ❌        | ✅         | `melodic`               |

**Note:**
- `B1 Correction` and `GradWarp` are **planned but not yet implemented**, and their placements are shown appropriately in the pipeline.
- `Intensity Scaling` remains **immediately after Brain Extraction** to preserve downstream data quality.
- `Distortion Correction` is not performed as very few fMRI images have `FieldMap` data in the DICOM files.
- `High-pass Filtering` and `ICA Denoising` are kept optional as in some cases the output is losing significant data.
# ✅ Summary of Tool Usage

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
| **Spatial Smoothing** [optional]                 | ✅        | ❌           | ❌       | ❌        | ✅         | `fslmaths -s`           |
| **Coregistration (FLIRT)**                       | ✅       | ✅           | ❌       | ❌        | ✅         | `flirt`                 |
| **Normalization (MNI, FNIRT)**                   | ✅    | ✅           | ✅       | ❌        | ✅         | `fnirt`                 |
| **High-pass Filtering** [optional]               | ✅        | ❌           | ❌       | ❌        | ✅         | `fslmaths -bptf`        |
| **ICA Denoising (MELODIC)** [optional]           | ✅     | ❌           | ❌       | ❌        | ✅         | `melodic`               |

**Note:**
- `B1 Correction` and `GradWarp` are **planned but not yet implemented**, and their placements are shown appropriately in the pipeline.
- `Intensity Scaling` remains **immediately after Brain Extraction** to preserve downstream data quality.
- `Distortion Correction` is not performed as very few fMRI images have `FieldMap` data in the DICOM files.
- `Spatial Smoothing`, `High-pass Filtering` and `ICA Denoising` are kept optional as in some cases the output is losing significant data.

## Updated Preprocessing Pipeline

| **Preprocessing Step**                                  | **FSL** | **HCP Pipelines** | **ANTs** | **dcm2niix**    | **Implemented**  | **Tool Used**          |
|---------------------------------------------------------|---------|-------------------|----------|-----------------|------------------|------------------------|
| **DICOM to NIfTI**                                      | Partial | ✅                 | ❌       | ✅ (Best)       | ✅               | `dcm2niix`             |
| **Gradient Distortion Correction (GradWarp)**           | ❌       | ✅                 | ❌       | ❌               | ❌ (Pending)     | **HCP Pipelines**      |
| **B1 Correction**                                       | ❌       | ✅                 | ✅       | ❌               | ❌ (Pending)     | **HCP Pipelines/ANTs** |
| **Slice Timing Correction**                             | ✅       | ❌                 | ❌       | ✅ (Metadata)   | ✅               | `slicetimer`           |
| **Create Mean Image (Tmean)**                           | ✅       | ❌                 | ❌       | ❌               | ✅               | `fslmaths -Tmean`      |
| **Brain Extraction (BET)**                              | ✅       | ✅                 | ❌       | ❌               | ✅               | `bet`                  |
| **Intensity Scaling** *(after BET)*                     | ✅       | ❌                 | ❌       | ❌               | ✅               | `fslmaths -ing`        |
| **Bias Field Correction (FAST)**                        | ✅       | ✅                 | ✅       | ❌               | ✅               | `fast`                 |
| **Motion Correction (MCFLIRT)**                         | ✅       | ✅                 | ❌       | ❌               | ✅               | `mcflirt`              |
| **Spatial Smoothing [optional]**                        | ✅       | ❌                 | ❌       | ❌               | ✅               | `fslmaths -s`          |
| **Coregistration (FLIRT)**                              | ✅       | ✅                 | ❌       | ❌               | ✅               | `flirt`                |
| **Normalization (MNI, FNIRT)**                          | ✅       | ✅                 | ✅       | ❌               | ✅               | `fnirt`                |
| **High-pass Filtering [optional]**                      | ✅       | ❌                 | ❌       | ❌               | ✅               | `fslmaths -bptf`       |
| **ICA Denoising (MELODIC) [optional]**                  | ✅       | ❌                 | ❌       | ❌               | ✅               | `melodic`              |

---

### Notes

1. **GradWarp (Gradient Distortion Correction)**  
   - Placed **before** slice timing or motion correction, but **not yet implemented**.  
   - Requires a manufacturer-specific gradient nonlinearity coefficient file (e.g., `Prisma_fit.grad`), **which is not part of DICOM or JSON metadata**.  

2. **B1 Correction**  
   - Also **not implemented** (placed before standard bias correction).  
   - Would require a **separately acquired B1 map** (not typically included in standard fMRI sessions).  

3. **Impact of Omitting GradWarp & B1 Correction**  
   - Since most of the distortion/inhomogeneity is more pronounced at the **edges** of the brain, **lack of GradWarp and B1 correction minimally affects** central regions (e.g., hippocampus, medial temporal lobe).  
   - For many routine fMRI analyses—especially those focusing on subcortical structures near the isocenter—**omitting these steps does not critically degrade data quality**.  

4. **Other Steps**  
   - We do **not** perform EPI distortion correction (field-map-based or topup) here because most datasets lack the necessary field map acquisitions.  
   - **Spatial smoothing, high-pass filtering, and ICA denoising** are optional and can depend on study design.  

This pipeline thus reflects the typical *recommended order*, with GradWarp and B1 Correction planned but **skipped** due to missing specialized data, and recognizes that for certain ROIs (like the hippocampus), **the omission’s impact is small**.

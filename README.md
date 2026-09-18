<<<<<<< HEAD
# Content-Aware Seam Carving: Sobel vs Scharr

Implementation and comparative evaluation of **energy-based Seam Carving**
for content-aware image retargeting.

## Project

**Content-Aware Image Retargeting via Energy-Based Seam Carving: A Comparative Study of Sobel and Scharr Operators**

This project implements Seam Carving using gradient-based energy maps and
compares the **Sobel** and **Scharr** operators across four image domains:
architecture, landscape, portrait, and texture.

## Repository Structure

```text
Seam-Carving-Sobel-Scharr/
├── README.md
├── notebooks/
│   └── Seam_Carving.ipynb
├── src/
│   └── notebook_code_reference.py
├── results/
│   ├── figures/
│   └── metrics/
├── report/
│   └── Seam_Carving_Report.pdf
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Methodology

```text
Input RGB Image
       │
       ▼
Energy Map
(Sobel / Scharr)
       │
       ▼
Cumulative Energy Matrix
(Dynamic Programming)
       │
       ▼
Optimal Seam Identification
       │
       ▼
Seam Removal / Insertion
       │
       ▼
Retargeted RGB Image
```

The energy is based on image gradients. The optimal seam is obtained by
dynamic programming and backtracking from the minimum cumulative energy
in the final row.

## Experimental Domains

| Domain | Structural profile |
|---|---|
| Architecture | High rigidity, strong geometric structures |
| Landscape | Low-gradient natural scenes |
| Portrait | Semantic saliency and human subjects |
| Texture | High-frequency repetitive patterns |

The evaluation dataset contains **12 images** across these four categories.

## Evaluation Metrics

- **SSIM (Structural Similarity Index Measure):** perceptual/structural similarity.
- **EPR (Edge Preservation Ratio):** retention of significant edge information.
- MSE and PSNR trajectories are also used in the experimental visualizations.

## Reported Results

| Domain | Sobel SSIM | Scharr SSIM | Sobel EPR | Scharr EPR |
|---|---:|---:|---:|---:|
| Architecture | 0.473 | 0.473 | 0.998 | 0.998 |
| Landscape | 0.474 | 0.476 | 0.998 | 0.998 |
| Portrait | 0.584 | 0.584 | 0.999 | 0.999 |
| Texture | 0.331 | 0.326 | 0.994 | 0.994 |

The report describes the overall results as showing practical equivalence
between Sobel and Scharr for the evaluated seam-carving setting.

## Key Observations

- Architecture and portrait categories have identical reported SSIM/EPR
  values to three decimal places.
- Texture shows the largest SSIM degradation among the reported categories.
- High edge-retention values do not necessarily imply preservation of
  semantic structures.
- The study identifies the global dynamic-programming seam path as an
  important factor in the observed behavior.

## Future Work

The report proposes:

- Semantic-aware seam carving using segmentation/object-aware masking.
- Forward-energy optimization.
- GPU-accelerated dynamic programming.
- Spatio-temporal video retargeting.

## Running the Notebook

Create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Then open:

```text
notebooks/Seam_Carving.ipynb
```

Run the notebook cells sequentially.

> **Note:** The notebook may depend on local image paths or locally available
> evaluation images. Those data files are intentionally not included in this
> repository unless they are explicitly redistributable.

## Authors

- Sayantan Chakraborty
- Srijani Mondal
- Netali Singh
- Vishakh Anant
- Saurav Kumar

**M.Tech in Computer Science and Engineering**  
Indian Institute of Technology Goa, India

## References

1. Avidan, S. & Shamir, A. (2007). *Seam Carving for Content-Aware Image Resizing.*
2. Rubinstein, M., Shamir, A. & Avidan, S. (2008). *Improved Seam Carving for Video Retargeting.*
3. Scharr, H. (2000). *Optimale Operatoren in der Digitalen Bildverarbeitung.*
4. Sobel, I. & Feldman, G. (1968). *A 3x3 Isotropic Gradient Operator for Image Processing.*
5. Wang, Y.-S., Tai, C.-L., Sorkine, O. & Lee, T.-Y. (2008). *Optimized Scale-and-Stretch for Image Resizing.*
6. Wolf, L., Guttmann, M. & Cohen-Or, D. (2007). *Non-Homogeneous Content-Driven Video-Retargeting.*

## Citation

If you use this repository in academic work, please cite the associated
project/report and acknowledge the authors listed above.
=======
# Content-Aware-Seam-Carving
>>>>>>> 13b2b074c47031489b29f262875bff96783ee57f

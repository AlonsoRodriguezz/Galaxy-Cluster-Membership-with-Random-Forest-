# Galaxy Cluster Membership with Random Forest

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)

We use a Random Forest classifier to identify galaxy cluster members out to large cluster-centric distances ($5\ \mathrm{R}_{200}$), developed in the **CHANCES (CHileAN Cluster galaxy Evolution Survey)** [Haines et al. (2023)](https://doi.org/10.18727/0722-6691/5308), this tool leverages mock catalogs from the CHANCES Low - $z$ sub-survey. By using distinct physically motivated features, the model achieves high completeness and standard purity, making it ideal to apply as a cleaning step to study galaxy pre-processing. **You can read the full thesis [here](thesis/Rodriguez_Thesis_2026.pdf)**

---

## Key Updates

* **Feature Normalization:** Features $V_{pec}$ and $r_{proj}$ are normalized by $\sigma_{200}$ and $R_{200}$ respectively to ensure similarity across different cluster mass regimes.

* **Robust Background Rejection:** Implements a more strict redshift cut ($z < 0.08$) to filter trivial interloper cases.

## Methodology & Workflow

The pipeline consists of the following steps:

1.  **Data Ingestion:** Loading mock catalogs (FITS) and cluster metadata.
2.  **Preprocessing & Feature Engineering:**
    * Redshift cut ($z < 0.08$) and data inspection.
    * Calculation of Local Density ($\Sigma_{10}$).
    * Normalization of Projected Phase Space coordinates ($R_{norm}$, $V_{norm}$).
4.  **Training:** Random Forest classifier with hyperparameter optimization (GridSearchCV) and class balancing (SMOTE/Random Undersampling strategies tested), leading to 6 model variants.
5.  **Evaluation:** Validation using Leave-One-Group-Out cross-validation scheme to ensure generalization across different clusters.*(A flowchart of the methodology will be added here soon)*

## Performance

The model prioritizes Completeness while maintaining a Purity comparable to dynamical methods like Caustics ([Diaferio & Geller (1997)](https://iopscience.iop.org/article/10.1086/304075).

| Metric | Score | Significance |
| :--- | :--- | :--- |
| **Completeness** | **~0.98** | Nearly all true members are recovered. |
| **Purity** | **~0.66** | Consistent with Caustic mass estimation profiles. |
| **F1-Score** | **~0.79** | High overall classification success. |*(See `results/figures` for the rest of the standard ML metrics.)*

## Installation & Usage

1.  **Clone the repository:**    
```bash
git clone [https://github.com/AlonsoRodriguezz/Galaxy-Cluster-Membership-with-Random-Forest.git]([https://github.com/AlonsoRodriguezz/galaxy-cluster-membership](https://github.com/AlonsoRodriguezz/Galaxy-Cluster-Membership-with-Random-Forest).git)
cd galaxy-cluster-membership
```
2.  **Install dependencies:**
```bash
pip install -r requirements.txt
```
3.  **Run the analysis:**    You can run the full script (requires CHANCES mocks) or explore the notebooks (recommended):

```bash
python RF_implementation.py
```

## Repository Structure* 

* `notebooks/`: Jupyter notebooks containing EDA and step-by-step model training.
* `results/`: Generated plots (PPS, 3D distributions, Learning Curves).
* `thesis/`: Full PDF document of the undergraduate thesis.

---

## Author & Acknowledgments

**Author:** Gerardo Alonso Rodríguez Jorquera  
**Supervisors:** PhD. Yara Jaffé, PhD. Pía Amigo  
**Affiliation:** Universidad Técnica Federico Santa María (USM), Department of Physics.

*This work uses data from the CHANCES Survey.*

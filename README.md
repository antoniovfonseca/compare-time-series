# **Comparison of Two Time Series of Maps**
[![AI Assisted](https://img.shields.io/badge/AI%20Assisted-Code%20Generation-success)](#AIdisclosure)
---
This repository contains the implementation associated with the research framework for comparing temporal map series. 

**Authors:** [Antonio Fonseca](https://orcid.org/my-orcid?orcid=0000-0001-6309-6204), [Robert Gilmore Pontius Jr](https://wordpress.clarku.edu/rpontius/)

**Institution:** [Clark University](https://www.clarku.edu/)

**Purpose:**

This notebook implements the framework from the article “Foundational concepts and equations to compare two time series of maps” to quantify and visualize agreement and change between two temporal map series. It defines modular Python functions to compute presence‐agreement components, gains and losses, and full‐extent change metrics. The script produces visualizations and exportable results for reproducible analysis.

**Notebook Outline:**
1. **Environmental Setup:** Load required Python libraries and define parameters for input paths, time points, and NoData values.
2. **Definition of Calculation Functions:** Establish modular functions to calculate presence agreement metrics for single time points, gross change metrics for time intervals, and net change components.
3. **Execute the functions:** Run a central execution pipeline to process data arrays and store the results for presence hits, gross change, and net change in memory.
4. **Plot the graphics:** Generate and export detailed bar charts to visualize presence components and temporal differences across the time series.

**Acknowledgements:**

The United States National Aeronautical and Space Administration supported this work through the Land-Cover and Land-Use Change Mission Directorate via the grant 80NSSC23K0508 entitled ["Irrigation as climate-change adaptation in the Cerrado biome of Brazil evaluated with new quantitative methods, socio-economic analysis, and scenario models."](https://lcluc.umd.edu/projects/irrigation-climate-change-adaptation-cerrado-biome-brazil-evaluated-new-quantitative)

<img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/nasa_lulc_dark.png" width="150" height="100"> <img src="https://raw.githubusercontent.com/antoniovfonseca/compare-time-series/figures/LTER-Network-logo-200x200px.jpg" width="120" height="70">
          <img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/clark_logo_horizontal.png" width="150" height="70">

<a id="AIdisclosure"></a>
## AI Use Disclosure
During the preparation of this work, the author used AI tools in order to generate helper functions and debug visualization scripts. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

---
## Citation

If you use this code or methodology in your research, please cite:

> Pontius Jr, R. G., & Fonseca, A. V. (2026). Foundational concepts and equations to compare two time series of maps. 

**BibTeX:**
```bibtex
@article{pontius2024gis,
  title={Foundational concepts and equations to compare two time series of maps},
  author={Pontius Jr, Robert Gilmore and Fonseca, Ant{\^o}nio V.},
  year={2026}
}
```
## Quickstart: Run Locally on Your Computer

1. **Clone the repository**
    ```bash
    git clone [https://github.com/antoniovfonseca/compare-time-series.git](https://github.com/antoniovfonseca/compare-time-series.git)
    cd compare-time-series
    ```

2. **Create and activate a virtual environment**
    ```bash
    conda env create -f environment.yml
    # or, with mamba:
    # mamba env create -f environment.yml
    ```

3. **Activate environment**
    ```bash
    conda activate cts_env
    ```

4. **Install JupyterLab (if you do not have it yet)**
    ```bash
    pip install jupyterlab
    ```

5. **Launch JupyterLab**
    ```bash
    jupyter lab
    ```

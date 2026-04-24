# **Comparison of Two Time Series of Maps**
[![AI Assisted](https://img.shields.io/badge/AI%20Assisted-Code%20Generation-success)](#AIdisclosure)
---
This repository contains the implementation associated with the research framework for comparing temporal map series. 

**Authors:** [Antonio Fonseca](https://orcid.org/my-orcid?orcid=0000-0001-6309-6204), [Robert Gilmore Pontius Jr](https://wordpress.clarku.edu/rpontius/), [Aiyin ZHANG]([https://orcid.org/my-orcid?orcid=0000-0001-6309-6204](https://orcid.org/0000-0001-6236-7105))

**Institution:** [Clark University](https://www.clarku.edu/)

**Purpose:**

This notebook implements the framework from the article “Foundational concepts and equations to compare two time series of maps” to quantify and visualize agreement, disagreement, and change between two time sereis maps. It defines modular Python functions to compute presence‐agreement components, gains and losses, and full‐extent change metrics. The script produces visualizations and exportable results for reproducible analysis.

**Notebook Outline:**
1. **Environmental Setup:** Load required Python libraries and define parameters for input paths, time points, and NoData values.
2. **Definition of Calculation Functions:** Establish modular functions to calculate presence agreement and disagreement metrics for single time points, gross change metrics for time intervals, and net change components.
3. **Execute the functions:** Run a central execution pipeline to process data arrays and store the results for presence and change assessments in csv file.
4. **Plot the graphics:** Generate and export bar charts to visualize presence components and temporal differences across the time series.
5. **Plot the maps:** Generate maps to visualize presence and change components across the time series.

**Acknowledgments:**

The United States National Aeronautical and Space Administration supported this work through the Land-Cover and Land-Use Change Mission Directorate via the grant 80NSSC23K0508 entitled ["Irrigation as climate-change adaptation in the Cerrado biome of Brazil evaluated with new quantitative methods, socio-economic analysis, and scenario models."](https://lcluc.umd.edu/projects/irrigation-climate-change-adaptation-cerrado-biome-brazil-evaluated-new-quantitative)

The United States National Science Foundation supported this work via the Long Term Ecological Research network via grant OCE-2224608 for [Plum Island Ecosystems.](https://pie-lter.mbl.edu/)

<p align="left">
  <img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/nasa_lulc_dark.png" height="80" alt="NASA LULC" />&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/antoniovfonseca/compare-time-series/blob/main/figures/LTER-Network-logo-200x200px.jpg?raw=1" height="80" alt="LTER Network logo" />&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/antoniovfonseca/summarize-change-components/refs/heads/main/logos/clark_logo_horizontal.png" height="80" alt="Clark logo" />
</p>

<a id="AIdisclosure"></a>
## AI Use Disclosure
During the preparation of this work, the author used AI tools in order to generate helper functions and debug visualization scripts. After using these tools, the author reviewed and edited the content as needed and takes full responsibility for the content of the publication.

---
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

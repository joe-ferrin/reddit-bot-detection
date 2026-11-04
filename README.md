# Reddit Bot Detection Using Graph-Based Anomaly Detection

This project explores unsupervised machine learning methods for identifying potentially automated Reddit accounts based on user behavior and interaction patterns.

Using Reddit comment data from May 2015, we constructed a user interaction network, engineered behavioral features, and applied Isolation Forest and Deep Graph Infomax (DGI) to identify anomalous accounts.

The project also includes an interactive D3.js visualization for exploring relationships between potentially anomalous users.

## Methodology

### 1. Data Processing and Graph Construction

Reddit comments were processed to construct a weighted user interaction graph. Users were connected when they commented on the same posts, with edge weights representing the frequency of shared interactions.

### 2. Feature Engineering

Features were extracted across four categories:

- **Graph:** Degree, weighted degree, clustering coefficient, PageRank, and betweenness centrality.
- **Temporal:** Posting patterns, activity across hours and days, and time between comments.
- **Content:** Comment length, text similarity, URL frequency, and capitalization patterns.
- **Metadata:** Comment scores, username patterns, and account activity statistics.

### 3. Anomaly Detection

Two approaches were evaluated:

- **Isolation Forest:** Applied directly to engineered user features to identify unusual behavioral patterns.
- **Deep Graph Infomax (DGI):** Used a graph convolutional network to learn user embeddings from the interaction graph. Isolation Forest was subsequently applied to the learned embeddings.

We compared anomaly rankings based on engineered features, DGI embeddings, and combined scores. We also investigated how transformations of skewed features affected the results.

### 4. Visualization

An interactive network visualization was developed using D3.js to explore anomalous users and their relationships across subreddits.

## Results

Isolation Forest applied to the original engineered features produced the most interpretable anomaly rankings during qualitative inspection. Highly ranked accounts frequently exhibited unusual activity or interaction patterns.

DGI embeddings identified different types of structural anomalies, providing an alternative representation of user behavior.

Because reliable ground-truth bot labels were unavailable, the model outputs represent anomaly scores rather than confirmed bot classifications. Results were evaluated primarily through exploratory analysis and manual inspection.

## Repository Structure

- `src/` — Python modules for data loading, feature engineering, graph construction, DGI, Isolation Forest, and visualization export.
- `notebooks/pipeline_example.ipynb` — Main analysis notebook, including feature exploration, preprocessing experiments, model training, and anomaly comparisons.
- `results/` — Saved feature, edge, and graph data generated during the project.
- `visualization/` — D3.js visualization and its corresponding JSON dataset.

## Reproducibility

This repository contains a curated version of the original academic project, intended to demonstrate the methodology, analysis, and implementation.

The original Reddit comment dataset is not included due to its size. However, processed edge, feature, and graph data are included, along with the notebook's saved outputs.

The complete pipeline has not been validated for end-to-end execution in its current form. Running the original data processing steps requires the missing Reddit dataset and a compatible Python environment.

The notebook is intended primarily as a reference for the analysis and modeling workflow.

## Technologies

Python, pandas, NumPy, NetworkX, PyTorch, PyTorch Geometric, scikit-learn, DuckDB, Matplotlib, Seaborn, and D3.js.

## Project Resources

### Analysis Notebook
- [pipeline_example.ipynb](notebooks/pipeline_example.ipynb)

### Project Materials
- [Portfolio Project Page](https://joe-ferrin.github.io/projects/reddit-bot-detection.html)
- [Interactive Network Visualization](https://joe-ferrin.github.io/assets/visualization/network.html)
- [Project Poster](https://joe-ferrin.github.io/assets/images/reddit-bot-detection_poster.svg)
- [Final Report](https://joe-ferrin.github.io/assets/reports/reddit-bot-detection_report.pdf)
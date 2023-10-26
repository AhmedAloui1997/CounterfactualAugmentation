# CounterfactualAugmentation


## Causal Inference with Data Augmentation

This project focuses on estimating Conditional Average Treatment Effects (CATE) leveraging data augmentation techniques to improve model performance. We utilize a combination of similarity measures for imputation, various causal inference models for prediction, and then evaluate their performance.

### Dependencies

1. `os`
2. `pandas`
3. `numpy`
4. `torch`
5. `sklearn`
- To Finish with exact version of the libraries -

### Running the Code

To execute the main file:

```
python main.py
```

Ensure that you have the required libraries and the `config.yaml` file properly configured before executing.

### Configuration (`config.yaml`)

The `config.yaml` file allows users to define:

- Dataset Name: Specifies the dataset to be loaded for the experiment.
- Similarity Measure: The method used to compute embeddings for imputation (e.g., Contrastive Learning, Euclidean Distance, Propensity Scores).
- Local Regressor: The regressor used for imputation (e.g., Gaussian Processes, Linear Regression).
- Gaussian Process Kernel: The kernel to be used if Gaussian Processes are selected as the local regressor.
- Parameters: Including the number of neighbors and distance threshold for the imputation process.
- Model Name: The causal inference model to be trained (e.g., causal_forest, bart, slearner, cfrnet, tlearner).

Example:

```yaml
dataset_name: "sample_data"
similarity_measure: "contrastive_learning"
local_regressor: "gp"
gp_kernel: "RBF"
params:
  num_neighbors: 5
  distance_threshold: 1.0
model_name: "causal_forest"
```

### Code Overview

1. **Seed Initialization**: For reproducibility across different runs.
2. **Configuration Loading**: The `config.yaml` file is parsed to extract parameters and configurations.
3. **Data Loading**: Based on the dataset name provided in the config.
4. **Data Augmentation**: Using the specified similarity measure.
5. **Data Imputation**: Missing values are imputed using local regressors while locality is defnined with some embedding.
6. **Causal Inference Model Training**: The model specified in the config is trained on the augmented data.
7. **Performance Evaluation**: The trained model's performance is evaluated and results are displayed.

### Custom Modules:

- `utils.config`: Module for loading configurations from `config.yaml`.
- `models`: Contains implementations of various causal inference models.
- `data_augmentation.local_regressor`: Contains the function for imputation of missing values.
- `utils.upload_data`: Module for data loading.
- `utils.perf`: Contains performance evaluation functions.
- `utils.generate_synthetic_data`: Provides functions to generate synthetic datasets.
- `data_augmentation`: Contains methods for various similarity measures and local regressors.

---

### Datasets

We use variaous synthetic and semi-synthetic datasets to test our proposed approach:

- IHDP
- News
- Twins 
- Linear synthetic dataset
- Non Linear synthetic dataset



Feel free to raise an issue, contribute to enhance the functionalities of this project, copy parts of this project for your personal projects.

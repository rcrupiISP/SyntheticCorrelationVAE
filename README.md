# SyntheticCorrelationVAE

## Overview

SyntheticCorrelationVAE is a research project exploring the application of Variational Auto-Encoders (VAEs) to generate synthetic financial correlation matrices, with a specific focus on correlations among equity indices. The project incorporates a unique simulation approach for generating correlation matrices, allowing for a deeper understanding of the latent space and providing insights into how correlation structures evolve over time.
More details can be found in the paper:

**Quantifying Credit Portfolio sensitivity to asset correlations with interpretable generative neural networks**  
*Sergio Caprioli, Emanuele Cagliero, Riccardo Crupi*  
[ArXiv Paper](https://arxiv.org/abs/2309.08652)


![VAE_diagram](https://github.com/rcrupiISP/SyntheticCorrelationVAE/assets/92302358/eade0514-3993-4665-9996-3107dc073f5d)

## VAE Training

The project begins with the training of a Variational Auto-Encoder (VAE) designed to capture the underlying structure of financial correlation matrices, see the notebook "Github_VAE_2d_training.ipynb". The VAE architecture comprises an encoder and a decoder. The encoder transforms input correlation matrices into a latent space, and the decoder reconstructs correlation matrices from points sampled in this latent space.

- **VAE Architecture:** Three linear layers in the encoder and decoder, with appropriate activation functions. The latent space is two-dimensional, providing a compact representation of the correlation structures.

- **Loss Function:** The Mean Squared Error (MSE) loss is employed during training, guiding the VAE to reconstruct correlation matrices faithfully. A regularization term is included for promoting the learning of a structured and interpretable latent space.

- **Training Process:** The VAE is trained on historical financial data, specifically equity indices returns. The training process involves optimizing the model parameters to minimize the reconstruction error while regularizing the latent space. The training progress can be monitored through visualizations, ensuring convergence and effective learning. At the end of the process a matrix can be represented as a point in a 2D embedding space, as in the figure below.

![scatter_explanation_](https://github.com/rcrupiISP/SyntheticCorrelationVAE/assets/92302358/c4fb7c18-75d3-4050-8472-eaa78b2be6ca)


## Simulation of Correlation Matrices

The simulation notebook, "Github_Simulation_VAE.ipynb", provides a detailed exploration of the correlation matrix generation process using the trained VAE model. The simulation involves sampling points from the latent space to generate synthetic correlation matrices.

- **Latent Space Sampling:** The notebook demonstrates the sampling of points in circles of increasing radius within the latent space. This approach allows for the exploration of different correlation structures, providing insights into how correlations may vary.

- **Risk Estimation:** Synthetic correlation matrices generated through the VAE are subjected to risk estimation analysis. This involves computing eigenvalues, eigenvectors, and assessing the overlap of dominant directions across different samples. These analyses contribute to understanding the risk implications of the synthetic correlation matrices.

![var_surface_2](https://github.com/rcrupiISP/SyntheticCorrelationVAE/assets/92302358/d853fc7f-8eff-494c-bc4b-491a67aa8695)


## Data Source

The example in the notebook utilizes freely available data on Italian default rates sourced from [Banco d'Italia](https://infostat.bancaditalia.it/). However, the methodology is adaptable for various financial datasets. The list of the 44 indexes employed are listed in the file **DATASTREAM_ IT_EU_US_EMERGING.txt**.

## Key Features

- **VAE Model:** A VAE model is implemented, offering an interpretable latent space representation.
- **Data Loading and Processing:** The project provides functionalities for loading and processing financial data, such as equity indices returns, into suitable formats for correlation matrix generation.
- **Training and Validation:** The code includes functions for training and validating the VAE model, with visualizations for assessing convergence and performance.
- **Simulation Notebook:** A comprehensive notebook guides users through the simulation process, enabling experimentation with different datasets and latent space configurations.
- **Explanatory Potential:** The project emphasizes the potential for explainability regarding changes in correlation matrices over time—a crucial feature for financial applications.

## Usage

Explore the project's methodology and simulation notebook for generating synthetic correlation matrices. Use the provided Jupyter notebook as a guide to understand the implementation details and adapt the codebase for specific research and application scenarios.

**Note:** Ensure that you have the necessary packages installed as listed in the notebook's import section.

Feel free to contribute, provide feedback, or adapt the codebase for your specific requirements.


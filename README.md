# SyntheticCorrelationVAE

## Overview

SyntheticCorrelationVAE is a research project exploring the application of Variational Auto-Encoders (VAEs) to generate synthetic financial correlation matrices, with a specific focus on correlations among equity indices. The project incorporates a unique simulation approach for generating correlation matrices, allowing for a deeper understanding of the latent space and providing insights into how correlation structures evolve over time.

## VAE Training

The project begins with the training of a Variational Auto-Encoder (VAE) designed to capture the underlying structure of financial correlation matrices. The VAE architecture comprises an encoder and a decoder. The encoder transforms input correlation matrices into a latent space, and the decoder reconstructs correlation matrices from points sampled in this latent space.

- **VAE Architecture:** Three linear layers in the encoder and decoder, with appropriate activation functions. The latent space is two-dimensional, providing a compact representation of the correlation structures.

- **Loss Function:** The Mean Squared Error (MSE) loss is employed during training, guiding the VAE to reconstruct correlation matrices faithfully. A regularization term is included for promoting the learning of a structured and interpretable latent space.

- **Training Process:** The VAE is trained on historical financial data, specifically equity indices returns. The training process involves optimizing the model parameters to minimize the reconstruction error while regularizing the latent space. The training progress can be monitored through visualizations, ensuring convergence and effective learning.

## Simulation of Correlation Matrices

The simulation notebook, "Synthetic correlations with VAEs - Simulation," provides a detailed exploration of the correlation matrix generation process using the trained VAE model. The simulation involves sampling points from the latent space to generate synthetic correlation matrices.

- **Latent Space Sampling:** The notebook demonstrates the sampling of points in circles of increasing radius within the latent space. This approach allows for the exploration of different correlation structures, providing insights into how correlations may vary.

- **Risk Estimation:** Synthetic correlation matrices generated through the VAE are subjected to risk estimation analysis. This involves computing eigenvalues, eigenvectors, and assessing the overlap of dominant directions across different samples. These analyses contribute to understanding the risk implications of the synthetic correlation matrices.

## Data Source

The example in the notebook utilizes freely available data on Italian default rates sourced from [Banco d'Italia](https://infostat.bancaditalia.it/). However, the methodology is adaptable for various financial datasets.

## Key Features

- **Linear VAE Model:** A linear VAE model is implemented, offering an interpretable latent space representation.
- **Data Loading and Processing:** The project provides functionalities for loading and processing financial data, such as equity indices returns, into suitable formats for correlation matrix generation.
- **Training and Validation:** The code includes functions for training and validating the VAE model, with visualizations for assessing convergence and performance.
- **Simulation Notebook:** A comprehensive notebook guides users through the simulation process, enabling experimentation with different datasets and latent space configurations.
- **Explanatory Potential:** The project emphasizes the potential for explainability regarding changes in correlation matrices over time—a crucial feature for financial applications.

## Usage

Explore the project's methodology and simulation notebook for generating synthetic correlation matrices. Use the provided Jupyter notebook as a guide to understand the implementation details and adapt the codebase for specific research and application scenarios.

**Note:** Ensure that you have the necessary packages installed as listed in the notebook's import section.

Feel free to contribute, provide feedback, or adapt the codebase for your specific requirements.


# dynamics-informed-gps
Code used in "Dynamics-Informed Gaussian Process models in Stone Soup" paper

This repository contains implementations of Gaussian Process (GP) models integrated into the [Stone Soup](https://github.com/dstl/Stone-Soup) tracking framework. This is associated with the paper "Dynamics-Informed Gaussian Processes models in Stone Soup", Chung, Lydeard, Godsill, 2025.

The modified Stone Soup library, including these models, is included as a submodule

## Repository Structure

- **`Stone-Soup/`** - Contains the modified Stone Soup library implementing the GP models.
- **`iGP_tutorial.ipynb`** - Demonstrates how to use the Integrated GP (iGP) models within the Stone Soup framework.
- **`eval_code/`** - Contains code used for the evaluation section of the associated paper.
- **`synthetic.py`** - Generate synthetic data from a GP and track with one or more GP models.
- **`mosquito.py`** - Tracks mosquito coordinates obtained from [fruit fly and mosquito flight trajectories](https://datadryad.org/dataset/doi:10.5061/dryad.n0b8m) stored in **`mosquito_coordinates.csv`**.
- **`synthetic_optim.py`** This script generates synthetic data using a selected model, computes the marginal log likelihood over a grid of length scales and kernel variances, and visualises the results with a contour plot highlighting both global and local maxima.
- **`tracking/`** - Contains helper functions for tracking implementations.

## Installation

To set up the project, follow these steps:

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/nfcc2/dynamics-informed-gps.git
cd dynamics-informed-gps
```

(*Note:* The `--recurse-submodules` flag ensures the Stone Soup submodule is also cloned.)

### 2. Create and Activate a Virtual Environment (Optional but Recommended)

```bash
python -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate'
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running the IGP Tutorial

To run the tutorial in VS Code:

1. Open `iGP_tutorial.ipynb` in VS Code.
2. Ensure the Jupyter and Python extensions are installed.
3. Select the correct interpreter (Cmd/Ctrl+Shift+P → "Python: Select Interpreter").
4. Run cells using the ▶️ buttons.

If the environment doesn’t show up, register it with:

```bash
python -m ipykernel install --user --name=env

### Running Synthetic Data Evaluation
Replicates Figure 2, visualising an example trajectory from a generating model, along with estimated tracks from tracking models.
```bash
python eval_code/synthetic.py
```

### Running Synthetic Optimisation Plots
Replicates Figure 3, visualising marginal log likelihood surface plots from a generating model on its trajectory.
```bash
python eval_code/synthetic_optim.py
```

### Tracking Mosquito Coordinates
Replicates Figure 4, showing tracking results of the GP models on the mosquito trajectory data.

```bash
python eval_code/mosquito.py
```

## License

This project is released under the [MIT License](LICENSE).


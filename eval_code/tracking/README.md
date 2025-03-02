# Tracking Module

This folder contains helper functions for implementing tracking workflows using Kalman filtering and Gaussian Process models within the Stone Soup framework.

## Files Overview

- **`kf_tracking.py`** - Implements the main tracking workflow using Kalman filtering.
- **`models_utils.py`** - Provides helper functions for initialising and accessing properties of Stone Soup transition models.
- **`plotting.py`** - Contains functions for visualising ground truth, measurements, estimated tracks, and uncertainties.
- **`tracking_int.py`** - Includes helper functions for initialising the tracking workflow.

## Usage

These scripts are not meant to be run directly but are used as utilities within the tracking workflow. They are called by **`mosquito.py`** and **`synthetic.py`**.

Ensure that the required dependencies are installed by following the installation steps in the main `README.md` before using these scripts.


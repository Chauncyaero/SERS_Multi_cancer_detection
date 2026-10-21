**Overview:**

This repository contains the source code for the paper **Interpretable hierarchical spectral learning for multi-cancer serum SERS recognition via a wavelet-CNN framework.**

We develop a Wavelet-based hierarchical CNN framework for accurate identification of multiple cancer types (e.g., normal, lung cancer, esophageal cancer) from SERS spectra.



**Data Availability Notice:**

Due to patient privacy and the extensive experimental resources invested in spectral data acquisition, the **real clinical SERS spectra are not publicly shared** in this repository. However, we provide a set of **mock spectra** (3 classes, 20 spectra each) that allow reviewers and readers to fully verify the correctness and executability of the data processing and model training pipeline.

The real dataset can be requested from the corresponding author under a data use agreement for academic purposes.



**Repository Structure:**

├── MockNormalSpectra	/ # 10 Excel files (mock normal spectra)

├── MockECSpectra		/ # 10 Excel files (mock esophageal cancer spectra)

├── MockLCSpectra		/ # 10 Excel files (mock lung cancer spectra)

├── DataLoader.m							# Step 1: Read Excel spectra

├── DataPrepare.m		 					# Step 2: Interpolation, normalization, wavelet transform

├── CNN\_training\_validation\_test\_2D\_multi\_level.m 	# Step 3: 10 seeds ×10-folds CV training, performance evaluation, ROC curves

├── CNN\_2D\_hierarchical\_CAM.m 	# Grad-CAM calculation for hierarchical CNN

├── Controlled\_feature\_layer\_comparision.m 	# Controlled experiment to visualize relationship between CAM sharpness and feature layer resolution

├── loadSerumSpectra.m 		# Utility: load spectra from **xxx.mat** files

├── qualityControl.m 			# Utility: quality control of spectra

├── splitSampleWiseTestSet.m 	# Utility: sample-wise split for creating independent test set

├── prepare\_roc\_for\_interp.m 		# Utility: prepare data for ROC interpolation

└── README.md



**Important:** The mock spectra provided here are **not real patient data**. They are synthetically generated and are intended **only to verify the integrity and executability of the code.** Please refer to the paper for all performance claims based on the genuine dataset.


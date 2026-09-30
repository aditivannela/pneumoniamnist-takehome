# PneumoniaMNIST Medical Imaging Data Pipeline & Analysis

## Overview

For this take-home, I used the PneumoniaMNIST dataset to build a small medical imaging data pipeline and explore data validation, quality control, SQL analysis, duplicate detection, and binary classification.

I chose PneumoniaMNIST following the clarification that either PathMNIST or PneumoniaMNIST could be used. Since PneumoniaMNIST contains normal and pneumonia-labeled chest X-rays, it aligned naturally with the binary classification portion of the assignment.

The dataset contains 28 x 28 grayscale images with two labels:

- 0: Normal
- 1: Pneumonia

There are 4,708 training images, 524 validation images, and 624 test images.

## Environment and Dependencies

I completed the analysis in Python using Jupyter Notebook on a CPU-based laptop.

Main packages used:

- medmnist
- numpy
- pandas
- duckdb
- matplotlib
- scikit-learn

Packages can be installed with:

`pip install medmnist numpy pandas duckdb matplotlib scikit-learn`

The analysis is contained in `take_home.ipynb`. After installing the dependencies, the notebook can be opened in Jupyter and run in order.

## Initial Data Inspection

Before processing the images, I inspected the raw PneumoniaMNIST `.npz` file rather than assuming its structure.

The file contained six arrays:

- `train_images`: (4708, 28, 28)
- `val_images`: (524, 28, 28)
- `test_images`: (624, 28, 28)
- `train_labels`: (4708, 1)
- `val_labels`: (524, 1)
- `test_labels`: (624, 1)

All of the arrays were stored as `uint8`.

I then used the MedMNIST dataset information to confirm the task and label definitions before continuing with the analysis.

## Metadata Pipeline

For each image, I generated a metadata record containing:

- image_id
- split
- label
- width
- height
- mean_intensity
- std_intensity
- min_intensity
- max_intensity
- image_hash

Each image was converted to a NumPy array so I could calculate image-level statistics.

I created unique image IDs using the split and image index, such as `train_0` and `val_0`.

I also generated a SHA-256 hash from each image's raw pixel values to identify exact duplicates.

The completed metadata table contains 5,856 records. I stored the table in a persistent DuckDB database called `pneumonia_metadata.duckdb`.

## Data Validation and Quality Control

I checked the metadata before modeling to make sure the dataset matched the expected structure.

I confirmed that:

- labels were limited to 0 and 1
- all images were 28 x 28 pixels
- there were no missing values in the generated metadata

### Low-Variance Images

I used pixel-intensity standard deviation as a simple quantitative quality-control measure.

I flagged images in the bottom 1% of `std_intensity`. This resulted in 48 training images being flagged because some images were tied at the threshold.

After displaying several of these images, some appeared blurrier or lower contrast than typical images. I treated this as an image-quality observation rather than a medical interpretation.

### Duplicate Detection

I used SHA-256 hashes of the raw pixel values to identify exact duplicate images.

I found 18 duplicate occurrences beyond the first occurrence within the training split.

I also checked for hashes appearing in multiple dataset splits. Eight unique hashes occurred across two different splits, with exact duplicates appearing between the training and validation data.

This is a potential source of data leakage because validation data is intended to represent unseen data. I therefore interpreted the validation model performance cautiously.

## SQL Analysis

I used DuckDB and SQL for part of the analysis.

I calculated class balance for each split, including:

- total images
- normal images
- pneumonia images
- percentage of images labeled pneumonia

I also grouped the metadata by label to compare image-intensity characteristics.

Normal-labeled images had an average mean intensity of approximately 140.20 and average pixel-intensity standard deviation of 44.12.

Pneumonia-labeled images had an average mean intensity of approximately 147.55 and average pixel-intensity standard deviation of 34.41.

These are descriptive differences in the images and were not interpreted as medical findings.

## Baseline Classification Model

I used logistic regression as a simple CPU-friendly baseline for binary classification.

Each 28 x 28 image was flattened into 784 pixel features, and pixel values were scaled from 0-255 to 0-1.

I used `class_weight="balanced"` to account for class imbalance.

The model was fit using only the training data. I evaluated it on the validation set before performing the final evaluation on the test set.

### Validation Results

- Accuracy: 95.2%
- Sensitivity: 95.1%
- Specificity: 95.6%
- ROC AUC: 0.991

Because exact duplicate images were identified between the training and validation splits, these results may be optimistic and should be interpreted cautiously.

### Test Results

- Accuracy: 86.9%
- Sensitivity: 97.2%
- Specificity: 69.7%
- ROC AUC: 0.926

The test confusion matrix contained:

- True negatives: 163
- False positives: 71
- False negatives: 11
- True positives: 379

Accuracy alone does not fully describe the model's behavior. Although the model achieved 86.9% test accuracy, its sensitivity was much higher than its specificity. It correctly identified most pneumonia-labeled images but also classified a number of normal images as pneumonia.

This is why I would consider sensitivity, specificity, and the confusion matrix alongside overall accuracy when evaluating this model.

## Data Leakage

A major leakage concern in an image-classification pipeline is allowing identical images, or information derived from them, to appear in both training and evaluation data.

My duplicate analysis identified exact images shared between the training and validation splits.

Before using this pipeline for a more rigorous model comparison, I would resolve these cross-split duplicates and repeat the evaluation.

I would also ensure that any preprocessing steps that learn parameters from the data are fit only on training data.

## Scaling to 1 Million Images

If this pipeline needed to process approximately one million continuously arriving hospital images, the first three changes I would make are:

1. Process images in batches instead of keeping the entire dataset in memory.
2. Maintain persistent metadata and processing status so previously processed images do not need to be recomputed.
3. Parallelize independent image-processing operations while controlling CPU and memory usage.

I would also keep image storage separate from metadata so that applications querying metadata or predictions would not need to load the underlying image files.

## Incremental Processing

If 10,000 new images arrived, I would avoid rerunning the full pipeline.

Each incoming image could be assigned a stable identifier and content hash. Before processing it, the pipeline could check persistent metadata to determine whether the image:

- is new
- is an exact duplicate
- has already been processed successfully
- previously failed and needs to be retried

I would store processing status and error information with the metadata. This would allow only new or failed images to be processed.

The image hash could also help identify cases where the contents of an image changed even if its filename remained the same.

## Example of a Scientifically Misleading Pipeline

A pipeline can execute successfully without producing scientifically reliable results.

The cross-split duplicates found in this analysis are an example. The code can run normally and produce strong validation metrics, but if some validation images are identical to training images, the validation set is not completely independent.

This could make validation performance appear stronger than the model's ability to generalize to truly unseen images.

For this reason, I would treat data validation and leakage detection as part of the analysis rather than only checking whether the code executes successfully.

## Limitations and Next Steps

This model is intended as a simple baseline and not as a clinically deployable system.

With additional time, I would:

- resolve cross-split duplicates and repeat model evaluation
- investigate false-positive and false-negative images
- evaluate whether a different classification threshold improves the sensitivity/specificity tradeoff
- compare additional simple models or image-derived features
- add structured error handling for corrupted or unreadable images
- separate ingestion, validation, analysis, and modeling into reusable scripts
- add automated validation tests for future incoming data

## AI Assistance

I used ChatGPT as a learning and debugging assistant during this take-home.

I used it to help explain unfamiliar Python and SQL syntax, troubleshoot errors while building the notebook, and discuss approaches for image quality checks, duplicate detection, and baseline model evaluation.

I did not treat generated code as a black box. I ran the implementation step by step and inspected intermediate results to make sure they matched the expected dataset structure.

One AI-assisted suggestion I specifically checked was using SHA-256 hashes of the raw pixel values for exact duplicate detection. After implementing it, I inspected records with matching hashes and compared the corresponding metadata before using the hashes for cross-split duplicate analysis.


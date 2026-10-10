# ML Lab-7 : Naive Bayes Classification on the Raisin Dataset

**Name:** Harshit Widhwani &nbsp;|&nbsp; **Section:** B &nbsp;|&nbsp; **Roll No:** 49 &nbsp;|&nbsp; **Batch:** B4

Ramdeobaba University, Nagpur | Department of Artificial Intelligence and Machine Learning

---

## Aim

To implement the Naïve Bayes Classification algorithm using suitable dataset.

## Dataset

`Raisin_Dataset.xlsx`, the UCI Raisin dataset (900 grains, 7 features, 2 classes: Kecimen and Besni).

## Files in this repository

| File | Description |
|---|---|
| [`b-b4-49_prac7_ml.ipynb`](b-b4-49_prac7_ml.ipynb) | **Main notebook with every step and all outputs already executed.** Opens directly in the browser. |
| `naive_bayes_raisin.py` | The same pipeline as a plain Python script. |
| `Raisin_Dataset.xlsx` | The dataset used. |

## Steps applied

1. Loaded `Raisin_Dataset.xlsx` and inspected it with `head()`, `tail()`, `describe()`, `info()`, null and duplicate checks
2. Plotted the class distribution (bar plot and pie chart), histograms, box plots and class-wise histograms with KDE
3. Plotted the correlation matrix and scatter plots of the size features
4. Encoded the class with `LabelEncoder` (Besni = 0, Kecimen = 1) and split the data 80 / 20 with `stratify=y`
5. Trained a **Gaussian Naive Bayes** model and evaluated it with accuracy, precision, recall, F1-score, a classification report and a confusion matrix
6. Compared training and testing accuracy
7. Compared ten values of **var_smoothing** from 1e-12 to 1e-2
8. Predicted the variety of a new raisin entered with `input()`

## How to run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn openpyxl notebook
```

Open the notebook (or run the `.py` script) from inside this folder so the dataset path resolves.

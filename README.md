# 📊 DataAnalysis

A comprehensive anomaly detection pipeline built in Python using statistical methods and machine learning models. This project walks through the full data science workflow: from exploratory data analysis to statistical profiling, Z-score filtering, PDF/PMF modeling, and a custom implementation of a Naïve Bayes classifier — all benchmarked against scikit-learn's models.

---

## 📁 Project Structure

```
DataAnalysis/
├── Data/
│   ├── Actual_Data.csv         # Full dataset with class labels
│   └── test_data.csv           # Sample test subset
├── Notebooks/
│   ├── Milestone1.ipynb        # Data profiling and visualization
│   ├── Milestone2.ipynb        # Statistical modeling + outlier detection
│   └── Milestone3.ipynb        # Naïve Bayes (manual & scikit-learn)
├── src/
│   ├── Milestone1.py           # EDA script with PMF/PDF/CDF analysis
│   ├── Milestone2.py           # Z-score detection, PDF fitting, PMF calc
│   └── Milestone3.py           # Manual + sklearn Naïve Bayes classifiers
```

---

## 🧠 Key Features

### 🔍 Exploratory Data Analysis (Milestone 1)
- Automatic detection of data types
- Missing and infinite value detection
- Statistical summaries (mean, variance, range)
- PMF/PDF/CDF visualizations per feature
- Conditional distributions by class (normal vs anomaly)
- Joint and conditional PDFs
- Correlation heatmaps

### 📏 Statistical Outlier Detection (Milestone 2)
- Z-score based anomaly detection
- Grid search over thresholds (2.0 to 3.0)
- Performance plots for Accuracy, Precision, Recall
- PDF fitting (normal, gamma, pareto, etc.)
- PMF modeling for categorical features

### 🧠 Naïve Bayes Classification (Milestone 3)
- Fully vectorized custom Naïve Bayes implementation
- Log-space optimization for underflow stability
- Support for both categorical and numerical features
- Performance comparison against:
  - GaussianNB
  - MultinomialNB
  - BernoulliNB (from scikit-learn)

---

## 📊 Sample Visualizations

- Histogram & KDE plots for feature distributions  
- Conditional PDFs split by class  
- Heatmaps for joint PMFs  
- CDF overlays  
- Correlation matrix of numerical fields  
- Metric vs threshold performance curves  

*(See notebooks for full outputs.)*

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

---

## 🚀 How to Run

1. Clone the repo:

   ```bash
   git clone https://github.com/mahmoudelfeelig/DataAnalysis.git
   cd DataAnalysis
   ```

2. Launch Jupyter Notebooks:

   ```bash
   jupyter notebook Notebooks/Milestone1.ipynb
   ```

3. Or run individual scripts:

   ```bash
   python src/Milestone1.py
   python src/Milestone2.py
   python src/Milestone3.py
   ```

---

## 🧪 Evaluation Metrics

- Accuracy, Precision, Recall at multiple thresholds
- Cross-correlation of original vs reconstructed signals
- Distribution fitting Mean Squared Error
- Side-by-side classifier performance summaries

---

## 📚 Educational Goals

This project is ideal for:

- Practicing advanced EDA and visual analytics
- Learning unsupervised anomaly detection via statistics
- Implementing a Naïve Bayes classifier from scratch
- Exploring how distribution assumptions impact modeling
- Comparing classical models vs ML libraries

---

## 👨‍💻 Author

**Mahmoud Elfeel**  
🔗 [GitHub Profile](https://github.com/mahmoudelfeelig)

---

## 📖 License

This project is licensed under the [MIT License](LICENSE).

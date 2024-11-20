import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt

# Task 1

# load the dataset
data = pd.read_csv('Train_data.csv')

# identify the target column and separate features
class_col = 'class'
X = data.drop(columns=[class_col])
y = data[class_col]

# split data into training (70%) and testing (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# separate numeric and categorical columns
numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

# preprocess numeric data (scale to compute Z-scores)
scaler = StandardScaler()
X_train_numeric = scaler.fit_transform(X_train[numeric_cols])
X_test_numeric = scaler.transform(X_test[numeric_cols])

# preprocess categorical data (one-hot encoding)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_categorical = encoder.fit_transform(X_train[categorical_cols])
X_test_categorical = encoder.transform(X_test[categorical_cols])

# combine processed numeric and categorical data
X_train_processed = np.hstack((X_train_numeric, X_train_categorical))
X_test_processed = np.hstack((X_test_numeric, X_test_categorical))

# convert the processed data into DataFrame for Z-score computation
X_train_processed_df = pd.DataFrame(X_train_processed)
X_test_processed_df = pd.DataFrame(X_test_processed)

# compute Z-scores
z_scores = np.abs((X_train_processed_df - X_train_processed_df.mean()) / X_train_processed_df.std())

# experiment with different thresholds
thresholds = np.arange(start= 2, stop= 3, step= .05)
results = []

for threshold in thresholds:
    print(f"\nThreshold: {threshold}")

    # predict anomalies for test data
    z_test = np.abs((X_test_processed_df - X_train_processed_df.mean()) / X_train_processed_df.std())
    anomaly_preds = (z_test > threshold).any(axis=1).astype(int)

    # convert y_test to binary (assuming 'normal' = 0, 'anomaly' = 1)
    y_test_binary = (y_test == 'anomaly').astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test_binary, anomaly_preds)
    precision = precision_score(y_test_binary, anomaly_preds)
    recall = recall_score(y_test_binary, anomaly_preds)

    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}")
    results.append((threshold, accuracy, precision, recall))

# plot performance metrics for different thresholds
thresholds, accuracies, precisions, recalls = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, label='Accuracy', marker='o')
plt.plot(thresholds, precisions, label='Precision', marker='o')
plt.plot(thresholds, recalls, label='Recall', marker='o')
plt.xlabel('Z-Score Threshold')
plt.ylabel('Metric Value')
plt.title('Performance Metrics vs Z-Score Thresholds')
plt.legend()
plt.grid()
plt.show()



# Task 2

original_data = pd.read_csv("Train_data.csv")
train_data = pd.read_csv("Train_data.csv")

# bring the 'class' column back from original data
train_data['class'] = original_data.loc[train_data.index, 'class']

# convert 'class' column to binary values
train_data['class'] = train_data['class'].map({'anomaly': 1, 'normal': 0})

# fit a variety of distributions and calculate MSE
def fit_pdf_and_calculate_mse(data):
    distributions = [stats.norm, stats.expon, stats.uniform, stats.pareto, stats.lognorm, stats.gamma,
                     stats.weibull_min]
    best_fit = None
    min_mse = float('inf')
    best_params = None
    best_dist = None

    for dist in distributions:
        try:
            # fit the distribution to the data
            params = dist.fit(data)
            # generate a range of values based on the data range
            x_range = np.linspace(min(data), max(data), len(data))
            # calculate the PDF for this distribution
            pdf = dist.pdf(x_range, *params)
            # compare the empirical distribution with the fitted PDF using MSE
            mse = mean_squared_error(data, pdf)

            # track the best-fit distribution
            if mse < min_mse:
                min_mse = mse
                best_fit = pdf
                best_params = params
                best_dist = dist

        except Exception as e:
            # handle cases where fitting fails for a distribution
            print(f"Fitting failed for {dist.name} with error: {e}")

    return best_fit, best_dist, best_params, min_mse


# example usage on a numerical column in the data
test_column = data['src_bytes']
best_fit, best_dist, best_params, min_mse = fit_pdf_and_calculate_mse(test_column)

print(f"Best Distribution: {best_dist.name}")
print(f"Parameters: {best_params}")
print(f"Mean Squared Error: {min_mse}")

# A function that calculates the pmf
def calculate_pmf(data):
    value_counts = data.value_counts(normalize=True)
    total_count = len(data)
    pmf = value_counts / total_count
    return pmf

# Dictionary to store the PMFs for all categorical columns
pmf_results = {}

# calculate PMF for each categorical column
categorical_columns = train_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"calculating PMF for column: {column}")

    # calculate the overall PMF for the column
    pmf_results[column] = {}
    pmf_results[column]['overall_pmf'] = calculate_pmf(train_data[column])

    # calculate the conditional PMF for anomaly (class == 1)
    anomaly_data = train_data[train_data['class'] == 1]
    pmf_results[column]['anomaly_pmf'] = calculate_pmf(anomaly_data[column])

    # calculate the conditional PMF for normal (class == 0)
    normal_data = train_data[train_data['class'] == 0]
    pmf_results[column]['normal_pmf'] = calculate_pmf(normal_data[column])


# print PMF results for reference
for column, pmf_data in pmf_results.items():
    print(f"\nPMF for column: {column}")
    print("Overall PMF:")
    print(pmf_data['overall_pmf'])
    print("Conditional PMF for Anomalies (class=1):")
    print(pmf_data['anomaly_pmf'])
    print("Conditional PMF for Normal (class=0):")
    print(pmf_data['normal_pmf'])
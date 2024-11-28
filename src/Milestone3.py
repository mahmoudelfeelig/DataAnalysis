import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# load dataset
data = pd.read_csv('../Data/Actual_Data.csv')

# identify numerical and categorical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# calculate priors
num_anomalies = sum(data['class'] == 'anomaly')
num_no_anomalies = sum(data['class'] == 'normal')
total = len(data)

priors = {
    'anomaly': num_anomalies / total,
    'normal': num_no_anomalies / total
}

# fit PDFs for numerical columns
pdfs = {}
for feature in numerical_columns:
    anomaly_data = data[data['class'] == 'anomaly'][feature]
    no_anomaly_data = data[data['class'] == 'normal'][feature]

    # compute mean and standard deviation
    anomaly_mean = anomaly_data.mean()
    anomaly_std = anomaly_data.std()
    no_anomaly_mean = no_anomaly_data.mean()
    no_anomaly_std = no_anomaly_data.std()

    # handle zero standard deviation
    anomaly_std = anomaly_std if anomaly_std > 0 else 1e-6  # Replace 0 with a small value
    no_anomaly_std = no_anomaly_std if no_anomaly_std > 0 else 1e-6  # Replace 0 with a small value

    # store the PDFs
    pdfs[feature] = {
        'anomaly': norm(loc=anomaly_mean, scale=anomaly_std),
        'normal': norm(loc=no_anomaly_mean, scale=no_anomaly_std)
    }

# fit PMFs for categorical columns
pmfs = {}
for feature in categorical_columns:
    anomaly_counts = data[data['class'] == 'anomaly'][feature].value_counts(normalize=True)
    no_anomaly_counts = data[data['class'] == 'normal'][feature].value_counts(normalize=True)

    pmfs[feature] = {
        'anomaly': anomaly_counts.to_dict(),
        'normal': no_anomaly_counts.to_dict()
    }

# extract features and target
X_categorical = data[categorical_columns]
X_numerical = data[numerical_columns]
y = data['class']

# convert data to NumPy arrays for faster processing
numerical_data = data[numerical_columns].to_numpy()
categorical_data = data[categorical_columns].to_numpy()
target = data['class'].to_numpy()

# precompute log priors
log_priors = {
    'anomaly': np.log(priors['anomaly']),
    'normal': np.log(priors['normal'])
}

# precompute log PDFs for numerical features
log_pdfs = {}
for feature in numerical_columns:
    anomaly_data = data[data['class'] == 'anomaly'][feature]
    no_anomaly_data = data[data['class'] == 'normal'][feature]

    anomaly_mean = anomaly_data.mean()
    anomaly_std = anomaly_data.std() if anomaly_data.std() > 0 else 1e-6
    no_anomaly_mean = no_anomaly_data.mean()
    no_anomaly_std = no_anomaly_data.std() if no_anomaly_data.std() > 0 else 1e-6

    log_pdfs[feature] = {
        'anomaly': norm(loc=anomaly_mean, scale=anomaly_std).logpdf(numerical_data[:, numerical_columns.get_loc(feature)]),
        'normal': norm(loc=no_anomaly_mean, scale=no_anomaly_std).logpdf(numerical_data[:, numerical_columns.get_loc(feature)])
    }

# precompute log PMFs for categorical features
log_pmfs = {}
for idx, feature in enumerate(categorical_columns):
    anomaly_counts = data[data['class'] == 'anomaly'][feature].value_counts(normalize=True)
    no_anomaly_counts = data[data['class'] == 'normal'][feature].value_counts(normalize=True)

    log_pmfs[feature] = {
        'anomaly': {k: np.log(v) for k, v in anomaly_counts.items()},
        'normal': {k: np.log(v) for k, v in no_anomaly_counts.items()}
    }

# calculate probabilities using vectorized operations
def calculate_probabilities_vectorized(numerical_data, categorical_data):
    log_anomaly_prob = np.full(len(data), log_priors['anomaly'])
    log_normal_prob = np.full(len(data), log_priors['normal'])

    # add numerical contributions
    for idx, feature in enumerate(numerical_columns):
        log_anomaly_prob += log_pdfs[feature]['anomaly']
        log_normal_prob += log_pdfs[feature]['normal']

    # add categorical contributions
    for idx, feature in enumerate(categorical_columns):
        for i, value in enumerate(categorical_data[:, idx]):
            log_anomaly_prob[i] += log_pmfs[feature]['anomaly'].get(value, -np.inf)
            log_normal_prob[i] += log_pmfs[feature]['normal'].get(value, -np.inf)

    # convert log probabilities to normal probabilities
    anomaly_prob = np.exp(log_anomaly_prob)
    normal_prob = np.exp(log_normal_prob)
    epsilon = 1e-10

    total_prob = anomaly_prob + normal_prob + epsilon # to prevent zero denominators
    return anomaly_prob / total_prob, normal_prob / total_prob

# vectorized prediction
anomaly_probs, normal_probs = calculate_probabilities_vectorized(numerical_data, categorical_data)
predictions = np.where(anomaly_probs > normal_probs, 'anomaly', 'normal')

# calculate metrics for the manual implementation
accuracy = accuracy_score(target, predictions)
precision = precision_score(target, predictions, pos_label='anomaly')
recall = recall_score(target, predictions, pos_label='anomaly')

print("Manual Naïve Bayes Model (Optimized)")
print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# # probability calculation function
# def calculate_probabilities(row, priors, pdfs, pmfs):
#     anomaly_log_sum = np.log(priors['anomaly'])
#     normal_log_sum = np.log(priors['normal'])
#
#     for feature, value in row.items():
#         if feature in pdfs:  # Numerical features
#             anomaly_pdf = pdfs[feature]['anomaly'].pdf(value)
#             normal_pdf = pdfs[feature]['normal'].pdf(value)
#
#             if anomaly_pdf > 0:
#                 anomaly_log_sum += np.log(anomaly_pdf)
#             if normal_pdf > 0:
#                 normal_log_sum += np.log(normal_pdf)
#         elif feature in pmfs:  # Categorical features
#             anomaly_pmf = pmfs[feature]['anomaly'].get(value, 0)
#             normal_pmf = pmfs[feature]['normal'].get(value, 0)
#
#             if anomaly_pmf > 0:
#                 anomaly_log_sum += np.log(anomaly_pmf)
#             if normal_pmf > 0:
#                 normal_log_sum += np.log(normal_pmf)
#
#     # Convert log-probabilities back to normal probabilities
#     anomaly_prob = np.exp(anomaly_log_sum)
#     normal_prob = np.exp(normal_log_sum)
#
#     denominator = anomaly_prob + normal_prob
#     if denominator == 0 or np.isnan(denominator):
#         return 0.5, 0.5  # Fallback to equal probabilities
#
#     return anomaly_prob / denominator, normal_prob / denominator

# # evaluate model
# predictions = []
# for _, row in data.iterrows():
#     row_dict = row.to_dict()
#     pr_anomaly, pr_no_anomaly = calculate_probabilities(row_dict, priors, pdfs, pmfs)
#     predicted_label = 'anomaly' if pr_anomaly > pr_no_anomaly else 'normal'
#     predictions.append(predicted_label)
#
# # calculate metrics for the manual implementation
# accuracy = accuracy_score(y, predictions)
# precision = precision_score(y, predictions, pos_label='anomaly')
# recall = recall_score(y, predictions, pos_label='anomaly')
#
# print("Manual Naïve Bayes Model")
# print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

# one-Hot Encoding for categorical features
encoder = OneHotEncoder(sparse_output=False)  # Ensure dense output
X_encoded = encoder.fit_transform(X_categorical)

# combine encoded categorical features with numerical features
X_combined = np.hstack([X_encoded, X_numerical])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# train and evaluate Naïve Bayes models
models = {
    "GaussianNB": GaussianNB(),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB()
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label='anomaly'),
        "recall": recall_score(y_test, y_pred, pos_label='anomaly')
    }

# display results
print("\nScikit-Learn Naïve Bayes Models")
for model, metrics in results.items():
    print(f"{model}: Accuracy={metrics['accuracy']}, Precision={metrics['precision']}, Recall={metrics['recall']}")

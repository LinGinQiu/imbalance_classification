from collections import Counter
from imblearn.datasets import fetch_datasets
from sklearn.model_selection import StratifiedKFold
from config import Config
from utils import *
import time
from numpy import interp
from sklearn.linear_model import LogisticRegression
from aeon.classification.interval_based import TimeSeriesForestClassifier

config = Config()
"""over-sampling methods include 'ADASYN', 'RandomOverSampler', 'KMeansSMOTE', 'SMOTE', 
'BorderlineSMOTE', 'SVMSMOTE', 'SMOTENC', 'SMOTEN'"""


# Load a dataset
datasetslist = list(fetch_datasets().items())
for dataname, data_dict in datasetslist:
    data, target = data_dict['data'], data_dict['target']

    # Plot the mean ROC curve
    plt.figure(figsize=(10, 6))

    for oversampler_name in ['ros', 'rose', 'adasyn', 'smote']:
        start_time = time.time()
        print(f"Using {oversampler_name} over-sampling method")
        print("-" * 50)
        # Split the data using StratifiedKFold
        skf = StratifiedKFold(n_splits=10, random_state=config.seed, shuffle=True)

        # Initialize lists to store metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        roc_auc_scores = []
        all_fpr = []
        all_tpr = []
        mean_fpr = np.linspace(0, 1, 100)

        for fold, (train_index, test_index) in enumerate(skf.split(data, target)):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = target[train_index], target[test_index]
            # print(f"Fold {fold + 1}")
            # print(f"Training set distribution: {Counter(y_train)}")
            # print(f"Test set distribution: {Counter(y_test)}\n")

            oversampler = getattr(OverSamplingMethods(), oversampler_name)()
            X_sampled, y_sampled = oversampler.fit_resample(X_train, y_train)

            clf = TimeSeriesForestClassifier(n_estimators=50, random_state=47)
            clf.fit(X_sampled, y_sampled)
            y_pred = clf.predict(X_test)
            y_pred_proba = clf.predict_proba(X_test)
            accuracy, precision, recall, f1, roc_auc_value, fpr, tpr \
                = metric_factors(y_test, y_pred, y_pred_proba, verbose=False)

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            roc_auc_scores.append(roc_auc_value)

            all_fpr.append(fpr)
            all_tpr.append(tpr)

        # Calculate average metrics
        avg_accuracy = np.mean(accuracy_scores)
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        avg_f1 = np.mean(f1_scores)
        avg_roc_auc = np.mean(roc_auc_scores)

        # calculate mean ROC curve
        mean_tpr_all = []
        for i in range(len(all_fpr)):
            mean_tpr_all.append(interp(mean_fpr, all_fpr[i], all_tpr[i]))
        mean_tpr_all = np.array(mean_tpr_all)
        mean_tpr = np.mean(mean_tpr_all, axis=0)

        # ensure the ROC curve  ends at 1 ,1
        mean_tpr[-1] = 1.0
        plt.plot(mean_fpr, mean_tpr, label=f'{oversampler_name} ROC curve (AUC = {avg_roc_auc:.2f})')
        # verbose
        end_time = time.time()
        print(f"Time taken of {oversampler_name}: {end_time - start_time:.2f} seconds")
        print(f'Average Accuracy of {oversampler_name}: {avg_accuracy:.4f}')
        print(f'Average Precision of {oversampler_name}: {avg_precision:.4f}')
        print(f'Average Recall of {oversampler_name}: {avg_recall:.4f}')
        print(f'Average F1 Score of {oversampler_name}: {avg_f1:.4f}')
        print(f'Average ROC AUC of {oversampler_name}: {avg_roc_auc:.4f}')
        print("*" * 50)


    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Random classifier line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Mean Receiver Operating Characteristic (ROC) Curve in {dataname}')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{dataname}_roc_curve.png', dpi=300)
    plt.show()


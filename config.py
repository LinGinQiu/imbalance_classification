import os.path
import pandas as pd

class Config:
    def __init__(self, run=False):
        """methods: over-sampling methods include 'ADASYN', 'RandomOverSampler', 'KMeansSMOTE', 'SMOTE',
        'BorderlineSMOTE', 'SVMSMOTE', 'SMOTENC', 'SMOTEN'"""

        # 数据相关

        self.root_path = '/scratch/cq2u24' if run else '/Users/qiuchuanhang/PycharmProjects'
        self.data_path = 'UCRArchive_112_imb'
        self.datasets_list = 'UCRArchive_2018_DataSetLists/classification.txt'
        self.num_workers = 4
        self.imbalance_ratio = 19
        # 模型相关
        self.oversampling = ['none_sampling', 'ros', 'rose', 'adasyn', 'smote']
        self.oversampling_methods = ['adasyn']
        self.classification_methods = ['hc2', 'multi_rocket_hydra', 'rotation_forest']
        self.classifier = 'rotation_forest'
        # 其他
        self.seed = 2024
        self.Kfold = 10
        self.results_csv_path = None
        self.img_path = None
        self.log_path = None
        self.check_path()

    def check_path(self):
        """
        create path if not exist and acclaim the path (log_path, img_path, results_csv_path)
        """
        self.log_path = f'results/log/model_UCR_112_data_{self.classifier}.log'
        self.img_path = f'results/img/{self.classifier}'
        self.results_csv_path = f'results/experiment_results_{self.classifier}.csv'
        log_dir = os.path.dirname(self.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.results_csv_path):
            # if not, create a new file
            df = pd.DataFrame(
                columns=['Dataset', 'Oversampler', 'Classifier', 'Accuracy', 'Precision', 'Recall', 'F1 Score',
                         'ROC AUC', 'test_distribution'
                         'Time Taken'])
            df.to_csv(self.results_csv_path, index=False)


import os.path


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
        self.oversampling_methods = ['none_sampling']
        self.classification_methods = ['tsf_classifier', 'logistic_regression',
                                       'hc2', 'multi_rocket_hydra', 'rotation_forest']
        self.classifier = 'multi_rocket_hydra'
        # 其他
        self.seed = 42
        self.log_path = f'results/log/model_UCR_112_data_{self.classifier}.log'
        self.img_path = f'results/img/{self.classifier}'
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        self.results_csv_path = f'results/experiment_results_{self.classifier}.csv'
        self.Kfold = 1




class Config:
    def __init__(self):
        """methods: over-sampling methods include 'ADASYN', 'RandomOverSampler', 'KMeansSMOTE', 'SMOTE',
        'BorderlineSMOTE', 'SVMSMOTE', 'SMOTENC', 'SMOTEN'"""

        # 数据相关
        self.data_path = '/scratch/cq2u24/UCRArchive_2018_imbalance'
        self.data_debug_path = '/Users/qiuchuanhang/PycharmProjects/UCRArchive_2018_imbalance'
        self.data_info = 'dataset_info.csv'
        self.num_workers = 4
        self.imbalance_ratio = 10
        # 其他
        self.seed = 42
        self.log_path = './log/model_UCR_data.log'

        # 模型相关
        self.oversampling_methods = ['ros', 'rose', 'adasyn', 'smote']
        self.classification_methods = ['tsf_classifier', 'logistic_regression']
        self.classifier = 'tsf_classifier'




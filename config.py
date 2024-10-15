
class Config:
    def __init__(self):

        # 数据相关

        self.train_data_path = './data/train.csv'
        self.test_data_path = './data/test.csv'
        self.num_workers = 4

        # 其他
        self.seed = 42

        # 模型超参数
        self.learning_rate = 0.001
        self.batch_size = 32
        self.num_epochs = 10
        self.input_size = 784
        self.hidden_size = 500
        self.output_size = 10



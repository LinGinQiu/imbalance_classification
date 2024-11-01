import argparse

# 创建解析器
parser = argparse.ArgumentParser(description="Process some integers.")

# 添加参数
parser.add_argument("--debug", type=str, help="if debug mode", default="True")

# 解析参数
args = parser.parse_args()

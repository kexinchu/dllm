#/bin/sh

# 运行所有三个场景
# python motivation/test_three_scenarios.py --scenario 0 --num-runs 1000 --max-batch-size 64

# 运行单个场景
python motivation/test_three_scenarios.py --scenario 1 --num-runs 1000  # Exact tie + varying batch
python motivation/test_three_scenarios.py --scenario 2 --num-runs 1000  # Near tie + same batch
python motivation/test_three_scenarios.py --scenario 3 --num-runs 1000  # Near tie + varying batch

#!/usr/bin/env python3
"""简单的推理演示脚本"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import jax
from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config

def main():
    print("=== π₀-FAST-DROID 推理演示 ===\n")

    # 1. 获取配置
    print("1. 加载配置...")
    config = _config.get_config("pi0_fast_droid")

    # 2. 下载/加载模型
    print("2. 下载模型检查点...")
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
    print(f"   模型路径: {checkpoint_dir}")

    # 3. 创建策略
    print("3. 创建训练好的策略...")
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)

    # 4. 准备示例数据
    print("4. 准备DROID示例数据...")
    example = droid_policy.make_droid_example()
    # --- 修复代码开始 ---
    def get_info(x):
        # 如果有 shape 属性（如 numpy array），返回 shape
        if hasattr(x, "shape"):
            return x.shape
        # 如果是字符串或其他类型，返回类型名称
        return f"<{type(x).__name__}>"

    print(f"   输入形状: {jax.tree.map(get_info, example)}")
    # --- 修复代码结束 ---

    # 5. 运行推理
    print("5. 运行推理...")
    result = policy.infer(example)

    # 6. 输出结果
    print("\n=== 推理结果 ===")
    print(f"动作形状: {result['actions'].shape}")
    print(f"动作值示例: {result['actions'][0, :5]}")  # 打印前5个动作

    # 清理
    del policy
    print("\n✓ 推理完成!")

if __name__ == "__main__":
    main()
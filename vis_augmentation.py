import os
import random
import matplotlib
# 设置为非交互式后端，防止没有 GUI 的环境报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dsets import LunaDataset, getCtAugmentedCandidate, getCtRawCandidate

def main():
    print("正在寻找一个阳性结节样本...")
    # 初始化无增强的数据集以获取干净的阳性样本元数据
    ds = LunaDataset()
    pos_candidates = [c for c in ds.candidateInfo_list if c.isNodule_bool]

    if not pos_candidates:
        print("未找到任何阳性结节样本！")
        return

    candidate = pos_candidates[0]
    print(f"找到阳性样本，series_uid: {candidate.series_uid}")
    print(f"结节直径: {candidate.diameter_mm:.2f} mm")
    print(f"中心位置 xyz: {candidate.center_xyz}")

    width_irc = (32, 48, 48)

    # 2. 定义各个测试场景对应的增强字典
    augmentation_scenarios = {
        "Original": {},
        "Flip": {"flip": True},
        "Offset": {"offset": 0.1},
        "Scale": {"scale": 0.2},
        "Rotate": {"rotate": True},
        "Noise": {"noise": 25.0},
        "Full Aug 1": {
            "flip": True, "offset": 0.1, "scale": 0.2, "rotate": True, "noise": 25.0
        },
        "Full Aug 2": {
            "flip": True, "offset": 0.1, "scale": 0.2, "rotate": True, "noise": 25.0
        }
    }

    # 3. 生成各个增强场景对应的中心切片
    slices = {}
    for name, aug_dict in augmentation_scenarios.items():
        print(f"正在生成场景: {name} ...")
        # 如果是需要翻转的场景，设置随机种子以确保在 2D 切片上能明显看出翻转效果（X/Y 轴都被翻转）
        if "Flip" in name or "Full" in name:
            random.seed(10)  # 种子 10 对应 [True, True, True]，即三个轴全部翻转
        else:
            # 其它场景重置或使用其它随机种子
            random.seed()

        candidate_t, _ = getCtAugmentedCandidate(
            aug_dict,
            candidate.series_uid,
            candidate.center_xyz,
            width_irc,
        )
        chunk_a = candidate_t[0].numpy() # 恢复成 (32, 48, 48) 形状的 numpy 数组
        center_slice = chunk_a[chunk_a.shape[0] // 2] # 提取 Z 轴中心切片
        slices[name] = center_slice

    # 4. 使用 Matplotlib 绘制 2x4 格式化子图
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Nodule Augmentation Comparison (Z-Center Slice)\nUID: {candidate.series_uid}", fontsize=16)

    # 图像 HU 范围限制（CT 肺窗典型值）
    clim = (-1000.0, 300.0)

    for idx, (name, slice_img) in enumerate(slices.items()):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        # 显示切片图像
        im = ax.imshow(slice_img, cmap='gray', clim=clim)
        ax.set_title(name, fontsize=12)
        ax.axis('off') # 隐藏坐标轴

    # 调整布局
    plt.tight_layout()

    # 5. 保存结果
    output_filename = "augmentation_comparison.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()
    print(f"\n对比图可视化成功！已保存至本地: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    main()

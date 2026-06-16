import os
import numpy as np
import matplotlib
# 使用 Agg 后端，确保在无显示器的环境（如服务器或后台命令）下也能成功绘图和保存
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dsets import getCandidateInfoList, getCtRawCandidate

def main():
    print("=== 开始运行结节掩码 (Mask) 可视化脚本 ===")

    # 1. 获取所有的候选样本并寻找第一个正样本（真实结节）
    candidates = getCandidateInfoList()
    pos_candidates = [c for c in candidates if c.isNodule_bool]

    if not pos_candidates:
        print("未找到任何真实的结节样本，请确认 annotations_with_malignancy.csv 的配置。")
        return

    candidate = pos_candidates[0]
    print(f"\n找到结节样本:")
    print(f"  series_uid: {candidate.series_uid}")
    print(f"  center_xyz: {candidate.center_xyz}")
    print(f"  diameter_mm: {candidate.diameter_mm:.2f} mm")

    # 2. 裁剪结节所在的三维 CT 块和掩码块
    width_irc = (32, 48, 48)  # Z, Y, X 轴大小
    ct_chunk, pos_chunk, center_irc = getCtRawCandidate(
        candidate.series_uid,
        candidate.center_xyz,
        width_irc
    )

    # 3. 确定可视化的切片索引（选择 Z 轴中心切片及其前后各几张）
    center_z = ct_chunk.shape[0] // 2
    slices_to_show = [center_z - 4, center_z - 2, center_z, center_z + 2, center_z + 4]

    # 4. 创建 Matplotlib 画布：5行（代表不同的 Z 轴切片），2列（左列为原图，右列为带 Mask 覆盖的图）
    fig, axes = plt.subplots(len(slices_to_show), 2, figsize=(10, 3 * len(slices_to_show)))
    fig.suptitle(f"Nodule Mask Visualization (Z-Slices Around Center)\nUID: {candidate.series_uid}", fontsize=14, y=0.98)

    # 图像 HU 范围限制（CT 肺窗典型值）
    clim = (-1000.0, 300.0)

    for row_idx, slice_idx in enumerate(slices_to_show):
        ct_slice = ct_chunk[slice_idx]
        mask_slice = pos_chunk[slice_idx]

        # --- 左列：原始 CT 图像 ---
        ax_raw = axes[row_idx, 0]
        ax_raw.imshow(ct_slice, cmap='gray', clim=clim)
        ax_raw.set_title(f"Slice Z={slice_idx} (Raw)", fontsize=10)
        ax_raw.axis('off')

        # --- 右列：CT 图像 + Mask 红色半透明覆盖 & 黄色边缘轮廓 ---
        ax_mask = axes[row_idx, 1]
        # 绘制背景 CT 灰度图
        ax_mask.imshow(ct_slice, cmap='gray', clim=clim)

        # 绘制半透明红色掩码区域
        overlay = np.zeros((*ct_slice.shape, 4))  # H, W, RGBA
        overlay[mask_slice] = [1.0, 0.0, 0.0, 0.4]  # 红色，透明度 0.4
        ax_mask.imshow(overlay)

        # 绘制黄色边界线轮廓
        if mask_slice.any():
            ax_mask.contour(mask_slice, colors='yellow', linewidths=1.5, levels=[0.5])

        ax_mask.set_title(f"Slice Z={slice_idx} (Mask Overlay)", fontsize=10)
        ax_mask.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 5. 保存图像
    output_filename = "nodule_mask_visualization.png"
    plt.savefig(output_filename, dpi=150)
    plt.close()

    print(f"\n[成功] 结节及掩码对比图已成功生成，保存路径为:")
    print(f"  {os.path.abspath(output_filename)}")
    print("=== 可视化完成 ===")

if __name__ == "__main__":
    main()

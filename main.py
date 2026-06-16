import time
from dsets import Ct, LunaDataset, getCandidateInfoList

def test():
    t0 = time.time()
    candidates = getCandidateInfoList()
    t1 = time.time()

    print(f"解析耗时：{t1 - t0:.2f}秒")
    print(f"共找到 {len(candidates)} 个候选样本")

    nodule_count = sum(1 for c in candidates if c.isNodule_bool)
    non_nodule_count = len(candidates) - nodule_count
    print(f"真实结节数量：{nodule_count}")
    print(f"非结节数量：{non_nodule_count}")

    print("\n前 5 个候选点情况：")
    for idx, c in enumerate(candidates[:5]):
        print(f"  [{idx}] 是否为结节: {c.isNodule_bool}, 直径: {c.diameter_mm:.2f}mm, center_xyz: {c.center_xyz}")

    # 验证单张 CT 加载
    sample_candidate = candidates[0]
    print(f"选择候选样本 series_uid：{sample_candidate.series_uid}")
    t2 = time.time()
    ct = Ct(sample_candidate.series_uid)
    t3 = time.time()
    print(f"成功加载 CT 扫描，耗时: {t3 - t2:.2f} 秒")
    print(f"CT 数组形状 (Z, Y, X): {ct.hu_a.shape}")
    print(f"CT 数组 HU 最小值: {ct.hu_a.min()}")
    print(f"CT 数组 HU 最大值: {ct.hu_a.max()}")
    print(f"CT 数组 HU 平均值: {ct.hu_a.mean():.2f}")

    # 验证裁剪结节候选块
    width_irc = (48, 48, 48)
    t4 = time.time()
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(sample_candidate.center_xyz, width_irc)
    t5 = time.time()
    print(f"裁剪耗时: {t5 - t4:.2f} 秒")
    print(f"裁剪块形状 (I, R, C): {ct_chunk.shape}")
    print(f"结节中心体素坐标 (I, R, C): {center_irc}")

    ct_chunk_padded, pos_chunk, center_irc_padded = ct.getRawCandidate(ct.origin_xyz, width_irc)
    print(f"原点中心体素坐标 (I, R, C): {center_irc_padded}")
    print(f"填充后裁剪块形状 (I, R, C): {ct_chunk_padded.shape}")
    # 前 24 个体素应该都是填充的 -1000.0 (空气)
    print(f"裁剪块边界填充部分的值 (第一维前3个元素): {ct_chunk_padded[:3, 0, 0]}")

    # 验证 LunaDataset
    ds = LunaDataset()
    print(f"初始化完整数据集，样本数: {len(ds)}")

    val_ds = LunaDataset(val_stride=10, isValSet_bool=True)
    train_ds = LunaDataset(val_stride=10, isValSet_bool=False)
    print(f"验证集样本数 (val_stride=10): {len(val_ds)}")
    print(f"训练集样本数 (val_stride=10): {len(train_ds)}")

    print("\n读取第一个样本 (触发第一次缓存)...")
    t_first = time.time()
    candidate_t, pos_t, series_uid, center_irc = ds[0]
    t_first_end = time.time()
    print(f"第一次读取耗时: {t_first_end - t_first:.4f} 秒")
    print(f"样本张量形状: {candidate_t.shape}, 类别标签: {pos_t}, series_uid: {series_uid}, 中心坐标: {center_irc}")

    print("\n再次读取同一个样本 (触发缓存命中)...")
    t_second = time.time()
    candidate_t2, pos_t2, series_uid2, center_irc2 = ds[0]
    t_second_end = time.time()
    print(f"第二次读取耗时: {t_second_end - t_second:.4f} 秒")

    # 验证数据可视化功能
    import matplotlib
    # 设置为非交互式后端，防止弹窗阻塞命令行运行
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from vis import findPositiveSamples, showCandidate

    pos_samples = findPositiveSamples(limit=2)
    print(f"找到 {len(pos_samples)} 个阳性样本，第一个样本的 series_uid: {pos_samples[0].series_uid}")

    print("正在生成结节可视化图像...")
    showCandidate(pos_samples[0].series_uid)

    # 将绘制的图像保存到本地文件
    output_img = 'candidate_visualization.png'
    plt.savefig(output_img)
    plt.close()
    print(f"可视化图像成功保存至: {output_img}")


if __name__ == '__main__':
    test()
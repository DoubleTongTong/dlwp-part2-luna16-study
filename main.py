import time
from dsets import Ct, getCandidateInfoList

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


if __name__ == '__main__':
    test()
import os
import glob
import csv
import functools
from collections import namedtuple

CandidateInfoTuple = namedtuple(
	'CandidateInfoTuple',
	'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
	# 根据用户数据集实际路径 E:\LUNA16 进行读取
	mhd_list = glob.glob('E:/LUNA16/subset*/*.mhd')
	presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

	# 构建真实结节字典
	diameter_dict = {}
	with open('E:/LUNA16/annotations.csv', "r", encoding='utf-8') as f:
		for row in list(csv.reader(f))[1:]:
			series_uid = row[0]
			annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
			annotationDiameter_mm = float(row[4])
			diameter_dict.setdefault(series_uid, []).append(
				(annotationCenter_xyz, annotationDiameter_mm)
			)

	# 遍历候选名单并进行模糊匹配
	candidateInfo_list = []
	with open('E:/LUNA16/candidates.csv', "r", encoding='utf-8') as f:
		for row in list(csv.reader(f))[1:]:
			series_uid = row[0]
			if series_uid not in presentOnDisk_set and requireOnDisk_bool:
				continue

			isNodule_bool = bool(int(row[4]))
			candidateCenter_xyz = tuple([float(x) for x in row[1:4]])
			candidateDiameter_mm = 0.0

			for annotation_tup in diameter_dict.get(series_uid, []):
				annotationCenter_xyz, annotationDiameter_mm = annotation_tup

				# 检查 X, Y, Z 三个维度的坐标差距
				for i in range(3):
					delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
					# 如果任一维度的偏差大于结节半径的一半 (直径/4)，则认为不是同一个结节
					if delta_mm > annotationDiameter_mm / 4:
						break
				else:
					# 匹配成功
					candidateDiameter_mm = annotationDiameter_mm
					break

			candidateInfo_list.append(CandidateInfoTuple(
				isNodule_bool,
				candidateDiameter_mm,
				series_uid,
				candidateCenter_xyz
			))

	# 排序与返回
	candidateInfo_list.sort(reverse=True)
	return candidateInfo_list

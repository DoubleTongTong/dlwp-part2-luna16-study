import os
import glob
import csv
import functools
from collections import namedtuple

import numpy as np
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset
import diskcache

from util import XyzTuple, xyz2irc

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

class Ct:
	def __init__(self, series_uid):
		mhd_path = glob.glob(
			'E:/LUNA16/subset*/{}.mhd'.format(series_uid)
		)[0]
		ct_mhd = sitk.ReadImage(mhd_path)
		ct_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
		ct_a.clip(-1000, 1000, ct_a)

		self.series_uid = series_uid
		self.hu_a = ct_a

		self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
		self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
		self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

	def getRawCandidate(self, center_xyz, width_irc):
		center_irc = xyz2irc(
			center_xyz,
			self.origin_xyz,
			self.vxSize_xyz,
			self.direction_a
		)

		slice_list = []
		pad_list = []
		for axis, center_val in enumerate(center_irc):
			start_ndx = int(round(center_val - width_irc[axis] / 2))
			end_ndx = int(start_ndx + width_irc[axis])

			max_size = self.hu_a.shape[axis]
			pad_left = max(0, -start_ndx)
			pad_right = max(0, end_ndx - max_size)

			start_clamped = max(0, start_ndx)
			end_clamped = min(max_size, end_ndx)

			slice_list.append(slice(start_clamped, end_clamped))
			pad_list.append((pad_left, pad_right))

		ct_chunk = self.hu_a[tuple(slice_list)]

		if any(any(p) for p in pad_list):
			ct_chunk = np.pad(ct_chunk, pad_list, mode='constant', constant_values=-1000.0)

		return ct_chunk, center_irc


raw_cache = diskcache.FanoutCache('data-unversioned/cache', shards=64, timeout=60)

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
	return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
	ct = getCt(series_uid)
	ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
	return ct_chunk, center_irc

class LunaDataset(Dataset):
	def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None):
		self.candidateInfo_list = list(getCandidateInfoList())
		if series_uid:
			self.candidateInfo_list = [
				x for x in self.candidateInfo_list if x.series_uid == series_uid
			]

		if isValSet_bool:
			assert val_stride > 0, val_stride
			self.candidateInfo_list = self.candidateInfo_list[::val_stride]
			assert self.candidateInfo_list
		elif val_stride:
			del self.candidateInfo_list[::val_stride]
			assert self.candidateInfo_list

	def __len__(self):
		return len(self.candidateInfo_list)

	def __getitem__(self, ndx):
		candidateInfo_tup = self.candidateInfo_list[ndx]
		width_irc = (32, 48, 48)

		candidate_a, center_irc = getCtRawCandidate(
			candidateInfo_tup.series_uid,
			candidateInfo_tup.center_xyz,
			width_irc
		)

		candidate_t = torch.from_numpy(candidate_a)
		candidate_t = candidate_t.to(torch.float32)
		candidate_t = candidate_t.unsqueeze(0)

		pos_t = torch.tensor([
			not candidateInfo_tup.isNodule_bool,
			candidateInfo_tup.isNodule_bool
		], dtype=torch.long)

		return (
			candidate_t,
			pos_t,
			candidateInfo_tup.series_uid,
			torch.tensor(center_irc)
		)

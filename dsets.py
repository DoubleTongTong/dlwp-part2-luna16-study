import os
import glob
import csv
import functools
import random
import math
from collections import namedtuple

import numpy as np
import SimpleITK as sitk

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import diskcache

from util import XyzTuple, xyz2irc

# 数据集主路径常量
LUNA16_DIR = 'E:/LUNA16'
# 磁盘缓存前缀路径，默认为当前目录
CACHE_DIR = '.'

CandidateInfoTuple = namedtuple(
	'CandidateInfoTuple',
	'isNodule_bool, diameter_mm, series_uid, center_xyz'
)

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
	# 根据用户数据集实际路径 E:\LUNA16 进行读取
	mhd_list = glob.glob(f'{LUNA16_DIR}/subset*/*.mhd')
	presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

	# 构建真实结节字典
	diameter_dict = {}
	with open(f'{LUNA16_DIR}/annotations.csv', "r", encoding='utf-8') as f:
		for row in list(csv.reader(f))[1:]:
			series_uid = row[0]
			annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
			annotationDiameter_mm = float(row[4])
			diameter_dict.setdefault(series_uid, []).append(
				(annotationCenter_xyz, annotationDiameter_mm)
			)

	# 遍历候选名单并进行模糊匹配
	candidateInfo_list = []
	with open(f'{LUNA16_DIR}/candidates.csv', "r", encoding='utf-8') as f:
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
			f'{LUNA16_DIR}/subset*/{series_uid}.mhd'
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


raw_cache = diskcache.FanoutCache(os.path.join(CACHE_DIR, 'data-unversioned/cache'), shards=64, timeout=60, size_limit=2**40)

@functools.lru_cache(1, typed=True)
def getCt(series_uid):
	return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
	ct = getCt(series_uid)
	ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)
	return ct_chunk, center_irc

def getCtAugmentedCandidate(
	augmentation_dict,
	series_uid, center_xyz, width_irc,
	use_cache=True
):
	if use_cache:
		ct_chunk, center_irc = getCtRawCandidate(series_uid, center_xyz, width_irc)
	else:
		ct = getCt(series_uid)
		ct_chunk, center_irc = ct.getRawCandidate(center_xyz, width_irc)

	ct_t = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

	transform_t = torch.eye(4)
	for i in range(3):
		if 'flip' in augmentation_dict and random.random() > 0.5:
			transform_t[i, i] *= -1
		if 'offset' in augmentation_dict:
			offset_float = augmentation_dict['offset']
			random_float = random.random() * 2 - 1
			transform_t[i, 3] = offset_float * random_float
		if 'scale' in augmentation_dict:
			scale_float = augmentation_dict['scale']
			random_float = random.random() * 2 - 1
			transform_t[i, i] *= 1.0 + scale_float * random_float

	if 'rotate' in augmentation_dict:
		angle_rad = random.random() * math.pi * 2
		s = math.sin(angle_rad)
		c = math.cos(angle_rad)
		rotation_t = torch.tensor([
			[c, -s, 0, 0],
			[s, c, 0, 0],
			[0, 0, 1, 0],
			[0, 0, 0, 1]
		], dtype=torch.float32)
		transform_t = transform_t @ rotation_t

	affine_t = F.affine_grid(
		transform_t[:3].unsqueeze(0).to(torch.float32),
		ct_t.size(),
		align_corners=False
	)
	augmented_chunk = F.grid_sample(
		ct_t,
		affine_t,
		padding_mode='border',
		align_corners=False
	).to('cpu')

	if 'noise' in augmentation_dict:
		noise_t = torch.randn_like(augmented_chunk) * augmentation_dict['noise']
		augmented_chunk += noise_t

	return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
	def __init__(self, val_stride=0, isValSet_bool=None, series_uid=None, ratio_int=0, limit=None, augmentation_dict=None):
		self.ratio_int = ratio_int
		self.limit = limit
		self.augmentation_dict = augmentation_dict
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

		if self.limit and not self.ratio_int:
			self.candidateInfo_list = self.candidateInfo_list[:self.limit]

		if self.ratio_int:
			self.negative_list = [
				nt for nt in self.candidateInfo_list if not nt.isNodule_bool
			]
			self.pos_list = [
				nt for nt in self.candidateInfo_list if nt.isNodule_bool
			]

			if self.limit:
				neg_limit = int(self.limit * self.ratio_int / (self.ratio_int + 1))
				pos_limit = self.limit - neg_limit
				self.negative_list = self.negative_list[:neg_limit]
				self.pos_list = self.pos_list[:pos_limit]

			assert self.pos_list
			assert self.negative_list
			self.shuffleSamples()

	def shuffleSamples(self):
		if self.ratio_int:
			import random
			random.shuffle(self.negative_list)
			random.shuffle(self.pos_list)

	def __len__(self):
		if self.ratio_int:
			return self.limit if self.limit else 200000
		else:
			return len(self.candidateInfo_list)

	def __getitem__(self, ndx):
		if self.ratio_int:
			pos_ndx = ndx // (self.ratio_int + 1)
			if ndx % (self.ratio_int + 1):
				neg_ndx = ndx - pos_ndx - 1
				neg_ndx %= len(self.negative_list)
				candidateInfo_tup = self.negative_list[neg_ndx]
			else:
				pos_ndx %= len(self.pos_list)
				candidateInfo_tup = self.pos_list[pos_ndx]
		else:
			candidateInfo_tup = self.candidateInfo_list[ndx]

		width_irc = (32, 48, 48)

		if self.augmentation_dict:
			candidate_t, center_irc = getCtAugmentedCandidate(
				self.augmentation_dict,
				candidateInfo_tup.series_uid,
				candidateInfo_tup.center_xyz,
				width_irc
			)
		else:
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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import itertools
import math
import pdb
import random
import numpy as np
from collections import defaultdict
from typing import Optional
import torch
from torch.utils.data.sampler import Sampler

from utils import comm


class InfiniteTrainingSampler(Sampler):
    """
    In training, we only care about the "infinite stream" of training data.
    So this sampler produces an infinite stream of indices and
    all workers cooperate to correctly shuffle the indices and sample different indices.

    The samplers in each worker effectively produces `indices[worker_id::num_workers]`
    where `indices` is an infinite stream of indices consisting of
    `shuffle(range(size)) + shuffle(range(size)) + ...` (if shuffle is True)
    or `range(size) + range(size) + ...` (if shuffle is False)
    """

    def __init__(self, size: int, shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._size = size
        assert size > 0
        if seed is None:
            seed = comm.shared_random_seed()

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        
        self.data_idxs = list(range(self._size))
        if seed is not None: random.seed(seed)
        if shuffle: random.shuffle(self.data_idxs)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        while True:
            for idx in self.data_idxs:
                yield idx
                
                
class InfiniteBalanceTrainingSampler(Sampler):
    '''
    Description:
        A sampler that provides more balanced sample among different data subsets.
    '''
    def __init__(self, sub_cumulative_sizes: int,):
        self.sub_cumulative_sizes = sub_cumulative_sizes
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        
        subset_lens = [(self.sub_cumulative_sizes[i+1] - self.sub_cumulative_sizes[i]) for i in range(len(self.sub_cumulative_sizes) - 1)]
        max_subset_len = max(subset_lens)
        subset_repeat_factors = [math.sqrt(max_subset_len / subset_len) for subset_len in subset_lens]
        self.data_idxs = []
        for i in range(len(subset_lens)):
            subset_idxs = random.choices(list(range(self.sub_cumulative_sizes[i], self.sub_cumulative_sizes[i + 1])), k = int(subset_repeat_factors[i] * subset_lens[i]))
            self.data_idxs += subset_idxs

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        while True:
            for idx in self.data_idxs:
                yield idx

class RepeatFactorTrainingSampler(Sampler):
    """
    Similar to TrainingSampler, but suitable for training on class imbalanced datasets
    like LVIS. In each epoch, an image may appear multiple times based on its "repeat
    factor". The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1] is defined
    as the fraction of images in the training set (without repeats) in which category c
    appears.

    See https://arxiv.org/abs/1908.03195 (>= v2) Appendix B.2.
    """

    def __init__(self, dataset_dicts, repeat_thresh, shuffle=True, seed=None):
        """
        Args:
            dataset_dicts (list[dict]): annotations in Detectron2 dataset format.
            repeat_thresh (float): frequency threshold below which data is repeated.
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        # Get fractional repeat factors and split into whole number (_int_part)
        # and fractional (_frac_part) parts.
        rep_factors = self._get_repeat_factors(dataset_dicts, repeat_thresh)
        self._int_part = torch.trunc(rep_factors)
        self._frac_part = rep_factors - self._int_part

    def _get_repeat_factors(self, dataset_dicts, repeat_thresh):
        """
        Compute (fractional) per-image repeat factors.

        Args:
            See __init__.

        Returns:
            torch.Tensor: the i-th element is the repeat factor for the dataset image
                at index i.
        """
        # 1. For each category c, compute the fraction of images that contain it: f(c)
        category_freq = defaultdict(int)
        for dataset_dict in dataset_dicts:  # For each image (without repeats)
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            for cat_id in cat_ids:
                category_freq[cat_id] += 1
        num_images = len(dataset_dicts)
        for k, v in category_freq.items():
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t / f(c)))
        category_rep = {
            cat_id: max(1.0, math.sqrt(repeat_thresh / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I, compute the image-level repeat factor:
        #    r(I) = max_{c in I} r(c)
        rep_factors = []
        for dataset_dict in dataset_dicts:
            cat_ids = {ann["category_id"] for ann in dataset_dict["annotations"]}
            rep_factor = max({category_rep[cat_id] for cat_id in cat_ids})
            rep_factors.append(rep_factor)

        return torch.tensor(rep_factors, dtype=torch.float32)

    def _get_epoch_indices(self, generator):
        """
        Create a list of dataset indices (with repeats) to use for one epoch.

        Args:
            generator (torch.Generator): pseudo random number generator used for
                stochastic rounding.

        Returns:
            torch.Tensor: list of dataset indices to use in one epoch. Each index
                is repeated based on its calculated repeat factor.
        """
        # Since repeat factors are fractional, we use stochastic rounding so
        # that the target repeat factor is achieved in expectation over the
        # course of training
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        # Construct a list of indices in which we repeat images as specified
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __iter__(self):
        start = self._rank
        yield from itertools.islice(self._infinite_indices(), start, None, self._world_size)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            # Sample indices with repeats determined by stochastic rounding; each
            # "epoch" may have a slightly different size due to the rounding.
            indices = self._get_epoch_indices(g)
            if self._shuffle:
                randperm = torch.randperm(len(indices), generator=g)
                yield from indices[randperm]
            else:
                yield from indices


class InferenceSampler(Sampler):
    """
    Produce indices for inference.
    Inference needs to run on the __exact__ set of samples,
    therefore when the total number of samples is not divisible by the number of workers,
    this sampler produces different number of samples on different workers.
    """

    def __init__(self, size: int):
        """
        Args:
            size (int): the total number of data of the underlying dataset to sample from
        """
        self._size = size
        assert size > 0
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()

        shard_size = (self._size - 1) // self._world_size + 1
        begin = shard_size * self._rank
        end = min(shard_size * (self._rank + 1), self._size)
        self._local_indices = range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    
class MemorySampler(Sampler):
    def __init__(self, size: int, ids_map_dict: dict, batch_size = int, sample_interval = int, shuffle: bool = True, seed: Optional[int] = None):
        self._size = size
        assert size > 0
        self.ids_map_dict = ids_map_dict
        self.ids_key_list = list(ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in ids_map_dict.values()])
        self.batch_size = batch_size
        self.batch_idxs = [None for _ in range(self.batch_size)]
        self.batch_idxs_end = [None for _ in range(self.batch_size)]
        self.sample_interval = sample_interval
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs_id = -1

    def __iter__(self):
        while True:
            self.bs_id = (self.bs_id + 1) % self.batch_size
            
            if self.batch_idxs[self.bs_id] is None or self.is_id_segment_end(self.batch_idxs[self.bs_id]):
                self.batch_idxs[self.bs_id] = random.choice(list(range(self._rank, self._size, self._world_size)))
                idx_range_id = np.searchsorted(self.ids_start_array, self.batch_idxs[self.bs_id], side = 'right') - 1
                idx_range = self.ids_map_dict[self.ids_key_list[idx_range_id]]
                self.batch_idxs_end[self.bs_id] = random.randint(self.batch_idxs[self.bs_id], idx_range[1])
            else:
                self.batch_idxs[self.bs_id] += self.sample_interval
            yield self.batch_idxs[self.bs_id]
        
    def is_id_segment_end(self, id):
        return (id + self.sample_interval) > self.batch_idxs_end[self.bs_id]
    
class MemoryClipSampler(Sampler):
    def __init__(self, size: int, ids_map_dict: dict, batch_size = int, sample_interval = int, shuffle: bool = True, seed: Optional[int] = None):
        self._size = size
        assert size > 0
        self.ids_map_dict = ids_map_dict
        self.ids_key_list = list(ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in ids_map_dict.values()])
        self.batch_size = batch_size
        self.batch_idxs = [None for _ in range(self.batch_size)]
        self.batch_idxs_end = [None for _ in range(self.batch_size)]
        self.sample_interval = sample_interval
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs_id = -1

    def __iter__(self):
        while True:
            self.bs_id = (self.bs_id + 1) % self.batch_size
            if self.batch_idxs[self.bs_id] is None or self.is_id_segment_end(self.batch_idxs[self.bs_id]):
                self.batch_idxs[self.bs_id] = random.choice(list(range(self._rank, self._size, self._world_size)))
                idx_range_id = np.searchsorted(self.ids_start_array, self.batch_idxs[self.bs_id], side = 'right') - 1
                idx_range = self.ids_map_dict[self.ids_key_list[idx_range_id]]
                self.batch_idxs_end[self.bs_id] = random.randint(self.batch_idxs[self.bs_id], idx_range[1])
            else:
                self.batch_idxs[self.bs_id] += self.sample_interval
            yield self.batch_idxs[self.bs_id]
        
    def is_id_segment_end(self, id):
        return (id + self.sample_interval) > self.batch_idxs_end[self.bs_id]
    
class MultiDatasetMemorySampler(Sampler):
    def __init__(self, sub_cumulative_sizes: list, ids_map_dict_list: list, batch_size = int, sample_interval = int, shuffle: bool = True, seed: Optional[int] = None):
        self._size = sub_cumulative_sizes[-1]
        assert self._size > 0
        # Merge the ids_map_dict of different datasets.
        self.ids_map_dict = {} 
        for dataset_id, dataset_ids_map_dict in enumerate(ids_map_dict_list):
            for seg_name, frame_range in dataset_ids_map_dict.items():
                merge_segname = f"set{dataset_id}_{seg_name}"
                merge_frame_range = (frame_range[0] + sub_cumulative_sizes[dataset_id], frame_range[1] + sub_cumulative_sizes[dataset_id])
                self.ids_map_dict[merge_segname] = merge_frame_range
        
        self.ids_key_list = list(self.ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in self.ids_map_dict.values()])
        self.batch_size = batch_size
        self.batch_idxs = [None for _ in range(self.batch_size)]
        self.batch_idxs_end = [None for _ in range(self.batch_size)]
        self.sample_interval = sample_interval
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs_id = -1

    def __iter__(self):
        while True:
            self.bs_id = (self.bs_id + 1) % self.batch_size
            if self.batch_idxs[self.bs_id] is None or self.is_id_segment_end(self.batch_idxs[self.bs_id]):
                self.batch_idxs[self.bs_id] = random.choice(list(range(self._rank, self._size, self._world_size)))
                idx_range_id = np.searchsorted(self.ids_start_array, self.batch_idxs[self.bs_id], side = 'right') - 1
                idx_range = self.ids_map_dict[self.ids_key_list[idx_range_id]]
                self.batch_idxs_end[self.bs_id] = random.randint(self.batch_idxs[self.bs_id], idx_range[1])
            else:
                self.batch_idxs[self.bs_id] += self.sample_interval
            yield self.batch_idxs[self.bs_id]
        
    def is_id_segment_end(self, id):
        return (id + self.sample_interval) > self.batch_idxs_end[self.bs_id]
    
class MultiDatasetMemoryClipSampler(Sampler):
    def __init__(self, sub_cumulative_sizes: list, ids_map_dict_list: list, batch_size = int, sample_interval = int, shuffle: bool = True, seed: Optional[int] = None):
        self._size = sub_cumulative_sizes[-1]
        assert self._size > 0
        # Merge the ids_map_dict of different datasets.
        self.ids_map_dict = {} 
        for dataset_id, dataset_ids_map_dict in enumerate(ids_map_dict_list):
            for seg_name, frame_range in dataset_ids_map_dict.items():
                merge_segname = f"set{dataset_id}_{seg_name}"
                merge_frame_range = (frame_range[0] + sub_cumulative_sizes[dataset_id], frame_range[1] + sub_cumulative_sizes[dataset_id])
                self.ids_map_dict[merge_segname] = merge_frame_range
        
        self.ids_key_list = list(self.ids_map_dict.keys())
        self.ids_start_array = np.array([ele[0] for ele in self.ids_map_dict.values()])
        self.batch_size = batch_size
        self.batch_idxs = [None for _ in range(self.batch_size)]
        self.batch_idxs_end = [None for _ in range(self.batch_size)]
        self.sample_interval = sample_interval
        self._shuffle = shuffle
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)

        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        self.bs_id = -1

    def __iter__(self):
        while True:
            self.bs_id = (self.bs_id + 1) % self.batch_size
            if self.batch_idxs[self.bs_id] is None or self.is_id_segment_end(self.batch_idxs[self.bs_id]):
                seg_range = random.choice(list(self.ids_map_dict.values()))
                self.batch_idxs[self.bs_id] = random.randint(seg_range[0], int(seg_range[0] + 0.05 * (seg_range[1] - seg_range[0])))
                self.batch_idxs_end[self.bs_id] = random.randint(int(seg_range[1] - 0.05 * (seg_range[1] - seg_range[0])), seg_range[1])
            else:
                self.batch_idxs[self.bs_id] += self.sample_interval
            yield self.batch_idxs[self.bs_id]
        
    def is_id_segment_end(self, id):
        return (id + self.sample_interval) > self.batch_idxs_end[self.bs_id]

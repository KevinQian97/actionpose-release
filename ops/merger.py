from collections import defaultdict

import numpy as np
import torch

class Filter(object):

    def __init__(self):
        pass

    def __call__(self, cube_acts, **kwargs):
        '''
        Perform filter on an `activity_spec.CubeActivities` object,
        returns a filtered `activity_spec.CubeActivities` object.
        Other parameters can be added if needed.
        '''
        return cube_acts

    def __repr__(self):
        return '%s.%s' % (self.__module__, self.__class__.__name__)


class OverlapCubeMerger(Filter):

    '''
    Merge overlap cubes into non-overlap ones.
    '''

    def __init__(self, cube_length: int = 64, stride: int = 16):
        self.stride = stride
        self.cube_length = cube_length
        self.n_overlaps = self.cube_length // self.stride
        assert self.n_overlaps * self.stride == self.cube_length, \
            'Cube length must be divisible by stride'

    def __call__(self, cube_acts):
        if len(cube_acts) == 0:
            return cube_acts
        columns = cube_acts.columns
        grouped_acts = self._group_acts(cube_acts.cubes.numpy(), columns)
        splited_acts = self._split_acts(grouped_acts, columns)
        merged_acts_list = []
        for acts in splited_acts:
            if acts.shape[0] == 1:
                merged_acts_list.append(acts)
                continue
            interpolated_acts = self._interpolate_acts(acts, columns)
            intervaled_acts = self._interval_acts(interpolated_acts, columns)
            merged_acts = self._merge_acts(intervaled_acts, columns)
            merged_acts_list.append(merged_acts)
        merged_acts = np.concatenate(merged_acts_list, axis=0)
        merged_cube_acts = cube_acts.__class__(
            torch.as_tensor(merged_acts), cube_acts.video_name,
            cube_acts.type_names)
        return merged_cube_acts

    def _group_acts(self, cube_acts, columns):
        grouped_acts = defaultdict(list)
        for cube in cube_acts:
            act_id = int(round(cube[columns.id]))
            act_type = int(round(cube[columns.type]))
            grouped_acts[(act_type, act_id)].append(cube)
        grouped_acts = [np.stack(acts) for acts in grouped_acts.values()]
        return grouped_acts

    def _split_acts(self, grouped_acts, columns):
        splited_acts = []
        for acts in grouped_acts:
            time_order = np.argsort(acts[:, columns.t0])
            acts = acts[time_order]
            times = acts[:, columns.t0]
            split_points = np.where(
                times[1:] - times[:-1] >= self.cube_length)[0]
            prev, split = 0, 0
            for split in split_points + 1:
                splited_acts.append(acts[prev:split])
                prev = split
            splited_acts.append(acts[split:])
        return splited_acts

    def _interpolate_acts(self, acts, columns):
        interpolated_acts = [acts[0]]
        for act in acts[1:]:
            prev_act = interpolated_acts[-1]
            inter_slots = int(round(
                act[columns.t0] - prev_act[columns.t0])) // self.stride
            assert inter_slots > 0
            if inter_slots == 1:
                interpolated_acts.append(act)
                continue
            box_delta_stride = act[columns.x0:columns.y1 + 1] - \
                prev_act[columns.x0:columns.y1 + 1] / inter_slots
            for inter_i in range(1, inter_slots):
                inter_act = prev_act.copy()
                inter_act[[columns.t0, columns.t1]] += \
                    self.stride * inter_i
                inter_act[columns.x0:columns.y1 + 1] += \
                    box_delta_stride * inter_i
                interpolated_acts.append(inter_act)
            interpolated_acts.append(act)
        interpolated_acts = np.stack(interpolated_acts)
        return interpolated_acts

    def _interval_acts(self, interpolated_acts, columns):
        overlaped_acts = np.zeros((
            self.n_overlaps, interpolated_acts.shape[0] + self.n_overlaps - 1,
            interpolated_acts.shape[1]), dtype=np.float32)
        for i_row in range(self.n_overlaps):
            start = i_row
            end = start + interpolated_acts.shape[0]
            overlaped_acts[i_row, :start] = interpolated_acts[0:1]
            overlaped_acts[i_row, :start, columns.t0:columns.t1 + 1] -= \
                np.arange(i_row, 0, -1)[:, None] * self.stride
            overlaped_acts[i_row, start: end] = interpolated_acts
            overlaped_acts[i_row, end:] = interpolated_acts[-1:]
            overlaped_acts[i_row, end:, columns.t0:columns.t1 + 1] += \
                np.arange(1, self.n_overlaps - i_row)[:, None] * self.stride
        intervaled_acts = overlaped_acts[0].copy()
        intervaled_acts[:, columns.score] = overlaped_acts[
            ..., columns.score].mean(axis=0)
        max_columns = [columns.t0, columns.x0, columns.y0]
        min_columns = [columns.t1, columns.x1, columns.y1]
        intervaled_acts[:, max_columns] = overlaped_acts[
            ..., max_columns].max(axis=0)
        intervaled_acts[:, min_columns] = overlaped_acts[
            ..., min_columns].min(axis=0)
        return intervaled_acts

    def _merge_acts(self, intervaled_acts, columns):
        length = intervaled_acts.shape[0] - self.n_overlaps + 1
        connected_acts = np.stack([intervaled_acts[start:start + length]
                                   for start in range(self.n_overlaps)])
        scores = connected_acts[:, :, columns.score].mean(axis=0)
        max_index = scores.argmax(axis=0)
        selected_acts = connected_acts[
            :, max_index % self.n_overlaps::self.n_overlaps]
        merged_acts = selected_acts[0].copy()
        merged_acts[:, columns.score] = scores[
            max_index % self.n_overlaps::self.n_overlaps]
        max_columns = [columns.t1, columns.x1, columns.y1]
        min_columns = [columns.t0, columns.x0, columns.y0]
        merged_acts[:, max_columns] = selected_acts[
            ..., max_columns].max(axis=0)
        merged_acts[:, min_columns] = selected_acts[
            ..., min_columns].min(axis=0)
        return merged_acts
import cv2
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_img_patches(img):
    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    stride_ratio = 0.5
    stride = int(input_width * stride_ratio)

    normalization_map = np.zeros((img_height, img_width), dtype=np.int16)

    patches = []
    patch_locs = []

    if img_height < img_width:
        assert img_height < 2 * input_height
        y_corner = [0, img_height - input_height, int(0.5 * (img_height - input_height))]
        for y in y_corner:
            for x in range(0, img_width - input_width + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if x != img_width - input_width:
                x = img_width - input_width
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
    else:
        assert img_width < 2 * input_width
        x_corner = [0, img_width - input_width, int(0.5 * (img_width - input_width))]
        for x in x_corner:
            for y in range(0, img_height - input_height + 1, stride):
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))
            if y != img_height - input_height:
                y = img_height - input_height
                segment = img[y:y + input_height, x:x + input_width]
                normalization_map[y:y + input_height, x:x + input_width] += 1
                patches.append(segment)
                patch_locs.append((x, y))

    assert np.all(normalization_map >= 1)

    patches.append(cv2.resize(img, (input_height, input_width), interpolation=cv2.INTER_CUBIC))

    patches = np.array(patches)

    return patches, patch_locs


def merge_pred_patches(img, preds, patch_locs):
    img_height, img_width, _ = img.shape

    input_height = input_width = 256

    probability_map = np.zeros((img_height, img_width), dtype=float)
    num1 = np.zeros((img_height, img_width), dtype=np.int16)

    for i, response in enumerate(preds):
        if i < len(preds) - 1:
            coords = patch_locs[i]
            probability_map[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += response
            num1[coords[1]:coords[1] + input_height, coords[0]:coords[0] + input_width] += 1
        else:
            mskp = cv2.resize(response, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

    assert np.all(num1 != 0)
    probability_map = probability_map / num1

    msk_pred = 0.5 * probability_map + 0.5 * mskp

    return msk_pred


class Visualizer(object):
    def __init__(self, loss_filename, training=False):
        self.log_name = os.path.join('./checkpoints', loss_filename)
        self.training = training
        if self.training:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training loss (%s) ================\n' % now)
        else:
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Testing begin (%s) ================\n' % now)

    def print_current_losses(self, epoch=0, iters=0, loss=0., lr=0., is_val=False):
        """print current losses on console; also save the losses to the disk
        """
        if not is_val:  # train
            message = '(epoch: %d, iters: %d) mean_loss: %6f lr: %6f' % (epoch, iters, loss, lr)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('%s\n' % message)  # save the message
        elif is_val:  # val
            message = 'validation on epoch>> %d, mean tloss>> %6f ' % (epoch, loss)
            print(message)  # print the message
            with open(self.log_name, "a") as log_file:
                log_file.write('val_mode:%s\n' % message)  # save the message

    def print_end(self, best=0, best_val_loss=0.):
        message = 'best model appear in epoch%d and best val_loss is %6f' % (best, best_val_loss)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Training End (%s) ================\n' % end_now)

    def print_val(self, tn, fp, fn, tp, precision, recll, f1):
        message = 'TN=%d, FP= %d, FN=%d, TP=%d\nprecision:%6f, recall:%6f, F1_score:%6f' % (
        tn, fp, fn, tp, precision, recll, f1)
        end_now = time.strftime("%c")
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)
            log_file.write('================ Testing End (%s) ================\n' % end_now)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight = torch.zeros_like(targets)
        weight = torch.fill_(weight, 0.04)
        weight[targets > 0] = 0.96

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def dice_loss(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)
    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    weight = torch.zeros_like(targets, device=targets.device)
    weight = torch.fill_(weight, 0.04)
    weight[targets > 0] = 0.96

    intersection = (inputs * targets).sum()
    dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
    BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
    return BCE + dice_loss


def get_img_patches_adjusted(img, crop_size, stride_ratio=0.5):
    c, img_height, img_width = img.shape
    input_height = input_width = crop_size
    stride = int(input_width * stride_ratio)

    patches = []
    patch_locs = []

    # 遍历图像，按stride移动窗口
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            # 如果块超出图像边界，则向左或向上调整
            adjusted_x = min(x, img_width - input_width)
            adjusted_y = min(y, img_height - input_height)

            segment = img[:,adjusted_y:adjusted_y + input_height, adjusted_x:adjusted_x + input_width]
            patches.append(segment.unsqueeze(0))
            patch_locs.append((adjusted_x, adjusted_y))
    patches = torch.cat(patches, dim=0)
    return patches, patch_locs


def merge_pred_patches_adjusted(img, preds, patch_locs):
    _, img_height, img_width = img.shape
    patch_num, _, patch_height, patch_width, = preds.shape
    assert patch_num == len(patch_locs)

    probability_map = torch.zeros((img_height, img_width), dtype=torch.float32)
    num1 = torch.zeros((img_height, img_width), dtype=torch.float32)

    for i, response in enumerate(preds):
        response = response.squeeze()
        coords = patch_locs[i]
        probability_map[coords[1]:coords[1] + patch_height, coords[0]:coords[0] + patch_width] += response
        num1[coords[1]:coords[1] + patch_height, coords[0]:coords[0] + patch_width] += 1

    probability_map = probability_map / num1
    msk_pred = probability_map

    return msk_pred


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    np.seterr(divide="ignore", invalid="ignore")
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    np.seterr(divide="warn", invalid="warn")
    res[np.isnan(res)] = 0.
    return res


class ComputeIoU(object):
    """
    IoU: Intersection over Union
    """

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.cfsmatrix = np.zeros((num_classes, num_classes), dtype="uint64")  # confusion matrix
        self.ious = dict()

    def get_cfsmatrix(self):
        return self.cfsmatrix

    def get_ious(self):
        self.ious = dict(zip(range(self.num_classes), per_class_iu(self.cfsmatrix)))  # {0: iou, 1: iou, ...}
        return self.ious

    def get_miou(self, ignore=None):
        self.get_ious()
        total_iou = 0
        count = 0
        for key, value in self.ious.items():
            if isinstance(ignore, list) and key in ignore or \
                    isinstance(ignore, int) and key == ignore:
                continue
            total_iou += value
            count += 1
        return total_iou / count

    def __call__(self, pred, label):
        """
        :param pred: [N, H, W]
        :param label:  [N, H, W}
        Channel == 1
        """

        pred = pred.cpu().numpy()
        label = label.cpu().numpy()

        assert pred.shape == label.shape

        self.cfsmatrix += fast_hist(pred.reshape(-1), label.reshape(-1), self.num_classes).astype("uint64")


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs


class CosineLRScheduler(Scheduler):
    """
    Cosine decay with restarts.
    This is described in the paper https://arxiv.org/abs/1608.03983.

    Inspiration from
    https://github.com/allenai/allennlp/blob/master/allennlp/training/learning_rate_schedulers/cosine.py
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            print("Cosine annealing scheduler will have no effect on the learning "
                  "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))

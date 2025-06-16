import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

from util.io import load_json
from .transform import RandomGaussianNoise, RandomHorizontalFlipFLow, \
    RandomOffsetFlow, SeedableRandomSquareCrop, ThreeCrop


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameReader:
    """ Reads frames using TorchVision. """
    IMG_NAME = '{:06d}.jpg'

    def __init__(self, frame_dir, modality, crop_transform, img_transform,
                 same_transform):
        self._frame_dir = frame_dir
        self._is_flow = modality == 'flow'
        self._crop_transform = crop_transform
        self._img_transform = img_transform
        self._same_transform = same_transform

    def read_frame(self, frame_path):
        """ Reads a single frame. """
        img = torchvision.io.read_image(frame_path).float() / 255
        if self._is_flow:
            img = img[1:, :, :] # GB channels contain data for flownet data
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1,
                    randomize=False):
        """ Loads frames for a clip. """
        rand_crop_state = None
        rand_state_backup = None
        ret = []
        n_pad_start = 0
        n_pad_end = 0
        processed_video_name = video_name # Expect pre-processed name here

        for frame_num in range(start, end, stride):
            if randomize and stride > 1:
                frame_num += random.randint(0, stride - 1)

            if frame_num < 0:
                n_pad_start += 1
                continue

            frame_path = os.path.join(
                self._frame_dir, processed_video_name,
                FrameReader.IMG_NAME.format(frame_num))

            try:
                img = self.read_frame(frame_path)
                if self._crop_transform:
                    if self._same_transform:
                        if rand_crop_state is None:
                            rand_crop_state = random.getstate()
                        else:
                            rand_state_backup = random.getstate()
                            random.setstate(rand_crop_state)

                    img = self._crop_transform(img)

                    if rand_state_backup is not None:
                        random.setstate(rand_state_backup)
                        rand_state_backup = None

                if not self._same_transform:
                    img = self._img_transform(img)
                ret.append(img)
            except RuntimeError as e:
                # Check specific error types that indicate missing file/decode error
                if 'No such file or directory' in str(e) or \
                   'could not be decoded' in str(e) or \
                   'Error opening file' in str(e) or \
                   'failed to load' in str(e): # Added another common error msg
                    n_pad_end += 1
                else:
                    print(f"RuntimeError processing {frame_path}: {e}")
                    # Consider logging or raising a specific exception if it's not a missing file
                    # For now, we treat other RuntimeErrors as padding triggers too
                    n_pad_end += 1 # Treat other errors as padding for robustness

        if not ret:
             # print(f"WARNING: No frames loaded for video '{video_name}' (processed: '{processed_video_name}') range {start}-{end}.")
             return None # Return None instead of empty tensor to signal failure

        # Determine stack dimension based on tensor shapes in ret
        try:
             # If individual items are (C, H, W), stack dim 0 creates (T, C, H, W)
             # If individual items are (N, C, H, W) (e.g. ThreeCrop), stack dim 1 creates (N, T, C, H, W)
             stack_dim = 1 if len(ret[0].shape) == 4 else 0
             ret_tensor = torch.stack(ret, dim=stack_dim)
        except Exception as stack_err:
             print(f"Error stacking frames for {video_name}: {stack_err}")
             # print(f"Shapes in list 'ret': {[f.shape for f in ret if isinstance(f, torch.Tensor)]}")
             return None # Return None on stacking error

        if self._same_transform:
            ret_tensor = self._img_transform(ret_tensor)

        if n_pad_start > 0 or (pad and n_pad_end > 0):
            # Adjust padding logic based on stack_dim
            pad_dims = [0] * (ret_tensor.ndim * 2)
            pad_idx_start = (ret_tensor.ndim - 1 - stack_dim) * 2 # Target the time dimension
            pad_dims[pad_idx_start] = n_pad_end if pad else 0
            pad_dims[pad_idx_start + 1] = n_pad_start

            try:
                ret_tensor = nn.functional.pad(ret_tensor, tuple(pad_dims))
            except Exception as pad_err:
                print(f"Error padding tensor for {video_name}: {pad_err}")
                # print(f"Tensor shape: {ret_tensor.shape}, Pad dims: {pad_dims}")
                return None # Return None on padding error

        return ret_tensor


DEFAULT_PAD_LEN = 5 # Pad the start/end of videos with empty frames


def _get_deferred_rgb_transform():
    img_transforms = [
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return nn.Sequential(*img_transforms)


def _get_deferred_bw_transform():
    img_transforms = [
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
        transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        RandomGaussianNoise()
    ]
    return nn.Sequential(*img_transforms)


def _load_frame_deferred(gpu_transform, batch, device):
    """ Applies deferred transforms (typically on GPU). """
    if 'frame' not in batch or batch['frame'] is None or batch['frame'].nelement() == 0:
         return None
    frame = batch['frame'].to(device)
    with torch.no_grad():
        # Assuming frame shape is (B, T, C, H, W) or similar for non-same_transform
        # Or (B, C, T, H, W) for same_transform? Check FrameReader stacking logic.
        # Let's assume (B, T, C, H, W) for now.
        for i in range(frame.shape[0]): # Iterate through batch
            frame_item = frame[i] # Shape (T, C, H, W)
            # Apply transform to each frame in the sequence if needed
            # This assumes gpu_transform works on (C, H, W) or (T, C, H, W)
            # If gpu_transform expects (B, C, T, H, W), this needs adjustment
            frame[i] = gpu_transform(frame_item) # Apply transform

        if 'mix_weight' in batch and batch['mix_weight'] is not None:
            weight = batch['mix_weight'].to(device)
            # Adjust weight shape for broadcasting
            weight_shape = [-1] + [1] * (frame.ndim - 1)
            frame = frame * weight.view(weight_shape)

            if 'mix_frame' in batch and batch['mix_frame'] is not None:
                 mix_frame = batch['mix_frame']
                 if isinstance(mix_frame, torch.Tensor) and mix_frame.nelement() > 0:
                     mix_frame = mix_frame.to(device)
                     # Apply transform to the mix_frame as well
                     for i in range(mix_frame.shape[0]):
                          mix_frame[i] = gpu_transform(mix_frame[i])
                     frame = frame + (1. - weight.view(weight_shape)) * mix_frame
                 # else: print warning about invalid mix_frame?
    return frame


def _get_img_transforms(
        is_eval,
        crop_dim,
        modality,
        same_transform,
        defer_transform=False,
        multi_crop=False
):
    """ Gets the appropriate image transforms. """
    crop_transform = None
    if crop_dim is not None:
        if multi_crop:
            assert is_eval, "Multi-crop only supported during evaluation"
            crop_transform = ThreeCrop(crop_dim)
        elif is_eval:
            crop_transform = transforms.CenterCrop(crop_dim)
        elif same_transform:
            crop_transform = SeedableRandomSquareCrop(crop_dim)
        else:
            crop_transform = transforms.RandomCrop(crop_dim)

    img_transforms = []
    if modality == 'rgb':
        if not is_eval:
            img_transforms.append(transforms.RandomHorizontalFlip())
            if not defer_transform:
                img_transforms.extend([
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(saturation=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=(0.7, 1.2))]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25)
                ])
        if not defer_transform:
            img_transforms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    elif modality == 'bw':
        if not is_eval:
            img_transforms.extend([transforms.RandomHorizontalFlip(), transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(hue=0.2)]), p=0.25)])
        img_transforms.append(transforms.Grayscale())
        if not defer_transform:
            if not is_eval:
                img_transforms.extend([
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(brightness=0.3)]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.ColorJitter(contrast=0.3)]), p=0.25),
                    transforms.RandomApply(nn.ModuleList([transforms.GaussianBlur(5)]), p=0.25),
                ])
            img_transforms.append(transforms.Normalize(mean=[0.5], std=[0.5]))
            if not is_eval:
                img_transforms.append(RandomGaussianNoise())
    elif modality == 'flow':
        assert not defer_transform, "Deferred transform not supported for flow"
        img_transforms.append(transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]))
        if not is_eval:
            img_transforms.extend([RandomHorizontalFlipFLow(), RandomOffsetFlow(), RandomGaussianNoise()])
    else:
        raise NotImplementedError(modality)

    img_transform = nn.Sequential(*img_transforms)
    return crop_transform, img_transform


def _print_info_helper(src_file, labels):
        """ Helper to print dataset statistics. """
        num_frames = sum([x.get('num_frames', 0) for x in labels])
        num_events = sum([len(x.get('events', [])) for x in labels])
        non_bg_perc = (num_events / num_frames * 100) if num_frames > 0 else 0
        print(f'{src_file} : {len(labels)} videos, {num_frames} frames, {non_bg_perc:0.5f}% non-bg')


class ActionSpotDataset(Dataset):
    """ Dataset for training action spotting models frame-by-frame. """
    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            dataset_len,
            is_eval=True,
            crop_dim=None,
            stride=1,
            same_transform=True,
            dilate_len=0,
            mixup=False,
            pad_len=DEFAULT_PAD_LEN,
            fg_upsample=-1,
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        # Use consistent key processing
        self._video_idxs = {}
        for i, x in enumerate(self._labels):
             original_key = x.get('video')
             if isinstance(original_key, str):
                  processed_key = original_key.replace('/', '_').strip()
                  self._video_idxs[processed_key] = i
             else:
                  print(f"Warning: Invalid or missing video key in label file {label_file} at index {i}")


        num_frames_list = [v.get('num_frames', 0) for v in self._labels]
        total_frames = np.sum(num_frames_list)
        if total_frames > 0:
            self._weights_by_length = np.array(num_frames_list, dtype=float) / total_frames
        else:
             # Handle case with no frames to avoid division by zero
             self._weights_by_length = np.ones(len(self._labels)) / max(1, len(self._labels))


        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0
        self._is_eval = is_eval

        self._dilate_len = dilate_len
        self._fg_upsample = fg_upsample

        if self._fg_upsample > 0:
            self._flat_labels = []
            for i, x in enumerate(self._labels):
                 original_key = x.get('video')
                 if not isinstance(original_key, str): continue # Skip if key invalid
                 proc_key = original_key.replace('/', '_').strip()
                 num_frames_video = x.get('num_frames', 0)
                 for event in x.get('events', []):
                     if 'frame' in event and 'label' in event and 0 <= event['frame'] < num_frames_video:
                         self._flat_labels.append((proc_key, event['frame']))
            if not self._flat_labels:
                 print("Warning: fg_upsample > 0 but no valid events found. Disabling FG sampling.")
                 self._fg_upsample = -1

        self._mixup = mixup
        self._gpu_transform = None
        if not is_eval and same_transform:
            if modality == 'rgb': self._gpu_transform = _get_deferred_rgb_transform()
            elif modality == 'bw': self._gpu_transform = _get_deferred_bw_transform()

        crop_transform, img_transform = _get_img_transforms(
            is_eval, crop_dim, modality, same_transform,
            defer_transform=self._gpu_transform is not None)

        self._frame_reader = FrameReader(frame_dir, modality, crop_transform, img_transform, same_transform)

    def load_frame_gpu(self, batch, device):
        """ Load frame batch to GPU and apply deferred transforms if any. """
        if 'frame' not in batch or batch['frame'] is None or not isinstance(batch['frame'], torch.Tensor) or batch['frame'].nelement() == 0:
             return None # Skip if frame data is missing or empty
        if self._gpu_transform is None:
            frame = batch['frame'].to(device)
        else:
            frame = _load_frame_deferred(self._gpu_transform, batch, device)
        return frame

    def _sample_uniform(self):
        """ Samples a clip uniformly based on video length. """
        video_idx = random.choices(range(len(self._labels)), weights=self._weights_by_length)[0]
        video_meta = self._labels[video_idx]
        original_key = video_meta.get('video')
        if not isinstance(original_key, str): # Handle potential missing key
             print(f"Warning: Missing or invalid video key at index {video_idx} during uniform sampling.")
             # Potentially re-sample or return None/raise error
             return None, None, None # Indicate failure
        video_key = original_key.replace('/', '_').strip() # Processed key

        video_len = video_meta.get('num_frames', 0)
        effective_clip_len_frames = (self._clip_len -1) * self._stride + 1
        # Calculate valid range for starting frame index
        min_start_frame = -self._pad_len * self._stride
        max_start_frame = video_len -1 + self._pad_len * self._stride - (self._clip_len -1) * self._stride

        if max_start_frame < min_start_frame:
             # Handle cases where video is shorter than clip len even with padding
             start_idx = min_start_frame
        else:
             start_idx = random.randint(min_start_frame, max_start_frame)

        return video_key, start_idx, video_meta

    def _sample_foreground(self):
        """ Samples a clip centered around a foreground event. """
        if not hasattr(self, '_flat_labels') or not self._flat_labels:
             return self._sample_uniform() # Fallback if no foreground labels

        video_key, frame_idx = random.choice(self._flat_labels) # Get processed key and frame
        if video_key not in self._video_idxs:
            print(f"Warning: Foreground sample key '{video_key}' not found in main index. Falling back.")
            return self._sample_uniform() # Fallback if key somehow missing

        video_meta = self._labels[self._video_idxs[video_key]] # Look up original meta using processed key
        video_len = video_meta.get('num_frames', 0)

        # Calculate sampling bounds for start_idx
        # We want the event frame (frame_idx) to be within the clip [start_idx, start_idx + (clip_len-1)*stride]
        lower_bound = frame_idx - (self._clip_len - 1) * self._stride
        upper_bound = frame_idx

        # Adjust bounds by padding
        min_possible_start = -self._pad_len * self._stride
        max_possible_start = video_len - 1 + self._pad_len * self._stride - (self._clip_len - 1) * self._stride

        final_lower = max(min_possible_start, lower_bound)
        final_upper = min(max_possible_start, upper_bound)


        if final_upper < final_lower:
             # Fallback if bounds are invalid (e.g., event too close to edge)
             # Center the clip on the frame_idx as much as possible within valid range
             potential_start = frame_idx - (self._clip_len // 2) * self._stride
             start_idx = max(min_possible_start, min(potential_start, max_possible_start))
             # Ensure start_idx is actually valid if max_possible_start < min_possible_start
             if max_possible_start < min_possible_start: start_idx = min_possible_start

        else:
             start_idx = random.randint(final_lower, final_upper)

        return video_key, start_idx, video_meta

    def _get_one(self):
        """ Gets a single sample (clip and labels). """
        use_foreground_sampling = (hasattr(self, '_flat_labels') and self._flat_labels and
                                   self._fg_upsample > 0 and random.random() < self._fg_upsample)

        video_key, base_idx, video_meta = None, None, None
        if use_foreground_sampling:
            video_key, base_idx, video_meta = self._sample_foreground()
        else:
            video_key, base_idx, video_meta = self._sample_uniform()

        if video_key is None: # Handle sampling failure
             print("Warning: Failed to sample video key.")
             return None

        labels = np.zeros(self._clip_len, np.int64)
        num_classes = len(self._class_dict) # Actual number of foreground classes

        for event in video_meta.get('events', []):
            if 'frame' not in event or 'label' not in event: continue

            event_frame = event['frame']
            # Calculate the indices in the 'labels' array that this event influences
            start_label_idx = (event_frame - self._dilate_len - base_idx + self._stride -1) // self._stride
            end_label_idx = (event_frame + self._dilate_len - base_idx) // self._stride

            clip_start_label_idx = max(0, start_label_idx)
            clip_end_label_idx = min(self._clip_len -1, end_label_idx)

            if clip_start_label_idx <= clip_end_label_idx: # Check if any part overlaps with the clip
                 label = self._class_dict.get(event['label'])
                 if label is not None: # Ensure label exists in class dict
                     mapped_label_index = label # Use 0-based index from dict now
                     if 0 <= mapped_label_index < num_classes:
                         # Assign 1-based index (+1) to the labels array
                         labels[clip_start_label_idx : clip_end_label_idx + 1] = mapped_label_index + 1

        frames = self._frame_reader.load_frames(
            video_key, base_idx, base_idx + self._clip_len * self._stride,
            pad=True, stride=self._stride, randomize=not self._is_eval)

        if frames is None: # Check if frame loading failed
             print(f"Warning: Frame loading failed for {video_key}, start {base_idx}. Skipping sample.")
             return None

        return {'frame': frames, 'contains_event': int(np.sum(labels) > 0), 'label': labels, 'video_key': video_key}

    def __getitem__(self, unused):
        """ Returns a single item or a mixed-up item for training. """
        ret = None
        attempts = 0
        max_attempts = 10 # Increased attempts
        while ret is None and attempts < max_attempts:
            ret = self._get_one()
            attempts += 1
        if ret is None:
            # Instead of raising error, return a dummy sample or skip?
            # For now, raise error as it indicates a persistent problem.
            raise RuntimeError(f"Failed to load a valid sample after {max_attempts} attempts.")

        if self._mixup and not self._is_eval:
            mix = None
            mix_attempts = 0
            while mix is None and mix_attempts < max_attempts: # Try to get mix sample
                 mix = self._get_one()
                 mix_attempts += 1

            if mix is None: # If mixup sample fails, return original without mixup
                 print("Warning: Failed to get mixup sample, returning original.")
                 ret['label'] = torch.from_numpy(ret['label']).long()
                 # Ensure frame is tensor
                 if isinstance(ret['frame'], np.ndarray): ret['frame'] = torch.from_numpy(ret['frame'])
                 elif not isinstance(ret['frame'], torch.Tensor): ret['frame'] = torch.as_tensor(ret['frame'])
                 return ret

            l = random.betavariate(0.2, 0.2)
            num_total_classes = len(self._class_dict) + 1 # Includes background class 0
            label_dist = np.zeros((self._clip_len, num_total_classes), dtype=np.float32)

            # ret['label'] and mix['label'] contain 0 for background, 1..N for foreground classes
            label_dist[np.arange(self._clip_len), ret['label']] += l
            label_dist[np.arange(self._clip_len), mix['label']] += (1. - l)

            # Ensure frames are tensors
            ret_frame_tensor = ret['frame'] if isinstance(ret['frame'], torch.Tensor) else torch.as_tensor(ret['frame'])
            mix_frame_tensor = mix['frame'] if isinstance(mix['frame'], torch.Tensor) else torch.as_tensor(mix['frame'])

            if self._gpu_transform is None: # Mix on CPU
                if ret_frame_tensor.shape == mix_frame_tensor.shape:
                    ret['frame'] = l * ret_frame_tensor + (1. - l) * mix_frame_tensor
                else:
                    print(f"Warning: Mixup frame shape mismatch ({ret_frame_tensor.shape} vs {mix_frame_tensor.shape}), skipping frame mixing.")
                    ret['frame'] = ret_frame_tensor # Keep original frame
            else: # Defer mixing to GPU
                ret['frame'] = ret_frame_tensor
                ret['mix_frame'] = mix_frame_tensor
                ret['mix_weight'] = float(l)

            ret['contains_event'] = max(ret['contains_event'], mix['contains_event'])
            ret['label'] = torch.from_numpy(label_dist).float() # Labels are now distributions
        else:
             ret['label'] = torch.from_numpy(ret['label']).long() # Labels are class indices

        # Ensure frame is tensor before returning
        if isinstance(ret['frame'], np.ndarray): ret['frame'] = torch.from_numpy(ret['frame'])
        elif not isinstance(ret['frame'], torch.Tensor): ret['frame'] = torch.as_tensor(ret['frame'])

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)


class ActionSpotVideoDataset(Dataset):
    """ Dataset for evaluating frame-by-frame predictions on whole videos. """
    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            crop_dim=None,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            flip=False,
            multi_crop=False,
            skip_partial_end=True
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        # Use consistent key processing
        self._video_idxs = {}
        for i, x in enumerate(self._labels):
             original_key = x.get('video')
             if isinstance(original_key, str):
                  processed_key = original_key.replace('/', '_').strip()
                  self._video_idxs[processed_key] = i
             else:
                  print(f"Warning: Invalid or missing video key in label file {label_file} at index {i}")

        self._clip_len = clip_len
        self._stride = stride
        assert stride > 0
        self._pad_len = pad_len

        self._flip = flip
        self._multi_crop = multi_crop

        crop_transform, img_transform = _get_img_transforms(
            is_eval=True, crop_dim=crop_dim, modality=modality, same_transform=True,
            defer_transform=False, multi_crop=self._multi_crop)

        self._frame_reader = FrameReader(frame_dir, modality, crop_transform, img_transform, False)

        self._clips = []
        for label_entry in self._labels:
            original_key = label_entry.get('video')
            if not isinstance(original_key, str): continue # Skip if key invalid
            video_key = original_key.replace('/', '_').strip() # Use processed key

            num_frames = label_entry.get('num_frames', 0)
            if num_frames <= 0 and self._pad_len <= 0: continue # Skip videos with no frames and no padding

            has_clip = False
            step = (self._clip_len - overlap_len) * self._stride
            # Ensure step is at least stride to avoid infinite loops with overlap >= clip_len
            step = max(step, self._stride)

            start_range_begin = -pad_len * self._stride
            # Calculate the last possible start index
            last_frame_index = num_frames -1
            last_possible_start = last_frame_index + self._pad_len * self._stride - (self._clip_len -1) * self._stride

            if skip_partial_end:
                # Only include clips that *start* within the valid frame range (0 to num_frames - clip_len_frames)
                 effective_clip_len_frames = (self._clip_len -1) * self._stride + 1
                 stop_frame_for_range = max(start_range_begin, num_frames - effective_clip_len_frames + 1)
            else:
                 # Include clips even if they extend past the end with padding
                 stop_frame_for_range = last_possible_start + step # Go one step past the last possible start


            for i in range(start_range_begin, stop_frame_for_range, step):
                 # Check if clip starts before video ends (considering padding)
                 if i < num_frames + pad_len * self._stride :
                     has_clip = True
                     self._clips.append((video_key, i)) # Store processed key

            # Ensure at least one clip is added if the video has frames or padding is enabled
            if not has_clip and (num_frames > 0 or pad_len > 0):
                 self._clips.append((video_key, start_range_begin))

        if not self._clips:
             print(f"WARNING: No clips generated for ActionSpotVideoDataset from {label_file}.")


    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        if idx >= len(self._clips):
             raise IndexError("Index out of range for dataset clips")

        video_key, start = self._clips[idx] # video_key is processed name
        frames = self._frame_reader.load_frames(
            video_key, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        # Handle frame loading failure from FrameReader
        if frames is None:
             print(f"Warning: Frame loading returned None for video {video_key} start {start}. Returning placeholder.")
             # Return placeholder or potentially raise error / filter in collate_fn
             return {'video': video_key, 'start': start, 'frame': torch.empty(0)} # Empty tensor placeholder

        ret_dict = {'video': video_key, 'start': start, 'frame': frames}

        if self._flip:
             current_frames = ret_dict['frame']
             if current_frames is not None and isinstance(current_frames, torch.Tensor) and current_frames.ndim >= 4:
                 try:
                      frames_flipped = torch.flip(current_frames, dims=(-1,)) # Flip width dim
                      # Check if already stacked (e.g., from multi-crop)
                      if current_frames.ndim == 5 and self._multi_crop: # Shape (N, T, C, H, W)
                           frames_flipped = torch.flip(current_frames, dims=(-1,))
                           # Stack along a new dimension or combine? Stacking might be simpler.
                           ret_dict['frame'] = torch.stack((current_frames, frames_flipped), dim=0) # (2, N, T, C, H, W)
                      elif current_frames.ndim == 4 : # Shape (T, C, H, W)
                           frames_flipped = torch.flip(current_frames, dims=(-1,))
                           ret_dict['frame'] = torch.stack((current_frames, frames_flipped), dim=0) # (2, T, C, H, W)
                      # else: potentially handle other dimensions or issue warning
                 except Exception as flip_err:
                      print(f"Warning: Error during flip augmentation for {video_key}: {flip_err}")
                      # Keep original frames if flip fails, ensure stacking dimension is handled downstream
                      if current_frames.ndim == 4:
                           ret_dict['frame'] = current_frames.unsqueeze(0) # Add a dimension to indicate no flip pair


        # Handle multi_crop output stacking
        if self._multi_crop and not self._flip: # If flip is True, stacking already happened
            # FrameReader with ThreeCrop returns (N, C, H, W) per frame, stacked to (N, T, C, H, W)
            # We need to make sure the dim order is consistent, maybe (B, N_crops, T, C, H, W) later
            # For now, let's assume the dataloader handles the batch dim, so output is (N_crops, T, C, H, W)
            pass # Already in correct shape from FrameReader stack_dim=1

        elif self._multi_crop and self._flip:
            # Already stacked to (2, N, T, C, H, W) where dim 0 is orig/flipped
            pass # Keep shape

        elif not self._multi_crop and self._flip:
             # Already stacked to (2, T, C, H, W)
             pass

        # If no augmentation, frame is (T, C, H, W)

        return ret_dict


    def get_labels(self, video):
        """ Retrieves ground truth labels for a given processed video key. """
        # Ensure lookup key uses same processing as init
        lookup_key = str(video).replace('/', '_').strip()

        if lookup_key not in self._video_idxs:
            # This was the source of the error. Return empty array gracefully.
            # print(f"Warning: Processed video key '{lookup_key}' not found in _video_idxs during get_labels.")
            return np.zeros(0, dtype=np.int64)

        original_label_index = self._video_idxs[lookup_key]
        meta = self._labels[original_label_index] # Get original metadata

        num_frames = meta.get('num_frames', 0)
        # Calculate number of labels based on stride
        safe_stride = max(1, self._stride)
        num_labels = max(0, (num_frames + safe_stride - 1) // safe_stride) # Ceiling division

        labels = np.zeros(num_labels, dtype=np.int64) # Background is 0
        num_fg_classes = len(self._class_dict) # Number of foreground classes

        for event in meta.get('events', []):
            if 'frame' not in event or 'label' not in event: continue

            frame = event['frame']
            if 0 <= frame < num_frames: # Ensure event frame is within bounds
                 label_idx = frame // safe_stride # Calculate strided index
                 if 0 <= label_idx < num_labels: # Ensure calculated index is valid
                      label_val = self._class_dict.get(event['label']) # Get 0-based index
                      if label_val is not None:
                           # Labels array uses 0 for background, 1 to N for classes
                           mapped_label_index = label_val + 1
                           if 1 <= mapped_label_index <= num_fg_classes:
                                labels[label_idx] = mapped_label_index
                           # else: print warning?
        return labels

    @property
    def augment(self):
        """ Returns true if flip or multi-crop augmentation is enabled. """
        return self._flip or self._multi_crop

    @property
    def videos(self):
         """ Returns list of (processed_video_name, num_strided_frames, strided_fps). """
         safe_stride = max(1, self._stride)
         video_list = []
         for i in range(len(self._labels)):
             v = self._labels[i]
             original_key = v.get('video')
             if not isinstance(original_key, str): continue # Skip invalid entries

             processed_video_name = original_key.replace('/', '_').strip()
             original_num_frames = v.get('num_frames', 0)
             # Calculate strided frames correctly using ceiling division
             num_strided_frames = max(0, (original_num_frames + safe_stride - 1) // safe_stride)
             strided_fps = v.get('fps', 25.0) / safe_stride
             video_list.append((processed_video_name, num_strided_frames, strided_fps))
         return sorted(video_list)

    @property
    def labels(self):
        """ Returns label info list potentially adjusted for evaluation stride. """
        labels_adjusted = []
        safe_stride = max(1, self._stride)
        for x in self._labels:
             x_copy = copy.deepcopy(x)
             original_key = x_copy.get('video')
             if not isinstance(original_key, str): continue # Skip invalid

             processed_key = original_key.replace('/', '_').strip()
             x_copy['video'] = processed_key # Store processed key

             original_num_frames = x_copy.get('num_frames', 0)
             x_copy['fps'] = x_copy.get('fps', 25.0) / safe_stride
             # Calculate strided frames correctly
             num_strided = max(0, (original_num_frames + safe_stride - 1) // safe_stride)
             x_copy['num_frames'] = num_strided

             labels_adjusted.append(x_copy)
        return sorted(labels_adjusted, key=lambda v: v.get('video', ''))

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)
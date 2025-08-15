import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    """
    Data Transformer for training U-Net models.
    """
    def __init__(self, isforward, max_key, augment=False, augmentation_prob=0.5, noise_level=0.01):
        """
        Args:
            isforward (bool): Whether this is for forward pass (testing) or training.
            max_key (str): Key to retrieve maximum value from attributes.
            augment (bool): Whether to apply data augmentation.
            augmentation_prob (float): Probability of applying each augmentation.
            noise_level (float): Standard deviation of Gaussian noise to add to k-space.
        """
        self.isforward = isforward
        self.max_key = max_key
        self.augment = augment
        self.augmentation_prob = augmentation_prob
        self.noise_level = noise_level

    def __call__(self, mask, input_kspace, target, attrs, fname, slice_num):
        """
        Args:
            mask (np.array): Sampling mask.
            input_kspace (np.array): Input k-space data.
            target (np.array): Target image.
            attrs (dict): Attributes from the h5 file.
            fname (str): File name.
            slice_num (int): Slice number.
        """
        # === Data Augmentation Block (for training only) ===
        if self.augment and not self.isforward:
            # 1. Random horizontal flip
            if np.random.uniform() < self.augmentation_prob:
                input_kspace = np.flip(input_kspace, axis=-1).copy() # axis=-1 corresponds to width
                target = np.flip(target, axis=-1).copy()

            # 2. Random vertical flip
            if np.random.uniform() < self.augmentation_prob:
                input_kspace = np.flip(input_kspace, axis=-2).copy() # axis=-2 corresponds to height
                target = np.flip(target, axis=-2).copy()

            # 3. Add k-space noise
            if np.random.uniform() < self.augmentation_prob:
                noise = np.random.normal(scale=self.noise_level, size=input_kspace.shape).astype(np.complex64)
                input_kspace = input_kspace + noise

        # === Original Transform Logic ===
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = torch.tensor([-1]) # Use torch.tensor for consistency
            maximum = -1

        # Apply mask and convert to tensor
        kspace = to_tensor(input_kspace * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        
        # Reshape mask
        mask_tensor = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        
        return mask_tensor, kspace, target, maximum, fname, slice_num
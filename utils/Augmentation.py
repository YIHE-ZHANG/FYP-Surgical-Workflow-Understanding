from datasets.transforms_ss import *
from randaugment import RandAugment
import traceback



# class GroupTransform(object):
#     def __init__(self, transform):
#         self.worker = transform

#     def __call__(self, img_group):
#         return [self.worker(img) for img in img_group]

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        try:
            return [self.worker(img) for img in img_group]
        except AttributeError as e:
            if "numpy" in str(e) and "int" in str(e):
                # This is the NumPy deprecation error
                print(f"Warning: RandAugment error due to NumPy version incompatibility: {e}")
                # Fall back to just returning the img_group unchanged
                return img_group
            else:
                # If it's a different error, we should know about it
                raise e


# def get_augmentation(training, config):
#     input_mean = [0.48145466, 0.4578275, 0.40821073]
#     input_std = [0.26862954, 0.26130258, 0.27577711]
#     # scale_size = config.data.input_size * 256 // 224
#     scale_size = config.data.input_size
#     if training:

#         unique = torchvision.transforms.Compose([GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
#                                                  GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
#                                                  GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4,
#                                                                         saturation=0.2, hue=0.1),
#                                                  GroupRandomGrayscale(p=0.2),
#                                                  GroupGaussianBlur(p=0.0),
#                                                  GroupSolarization(p=0.0)]
#                                                 )
#     else:
#         unique = torchvision.transforms.Compose([GroupScale(scale_size),
#                                                  GroupCenterCrop(config.data.input_size)])

#     common = torchvision.transforms.Compose([Stack(roll=False),
#                                              ToTorchFormatTensor(div=True),
#                                              GroupNormalize(input_mean,
#                                                             input_std)])
#     return torchvision.transforms.Compose([unique, common])

def get_augmentation(training, config):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = config.data.input_size
    
    if training:
        if config.data.dataset == 'rarp50':
            # Special augmentations for surgical videos
            unique = torchvision.transforms.Compose([
                GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_sth=False),
                GroupSurgicalColorJitter(p=0.8, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
                GroupRandomGrayscale(p=0.1),
                GroupGaussianBlur(p=0.3),
                GroupSurgicalNoise(p=0.3),
                GroupSurgicalBlur(p=0.2),
                GroupSurgicalSpecularityHighlight(p=0.2)
            ])
        else:
            # Default augmentations for other datasets
            unique = torchvision.transforms.Compose([
                GroupMultiScaleCrop(config.data.input_size, [1, .875, .75, .66]),
                GroupRandomHorizontalFlip(is_sth='some' in config.data.dataset),
                GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                GroupRandomGrayscale(p=0.2),
                GroupGaussianBlur(p=0.0),
                GroupSolarization(p=0.0)
            ])
    else:
        unique = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(config.data.input_size)
        ])

    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(input_mean, input_std)
    ])
    
    return torchvision.transforms.Compose([unique, common])


# def randAugment(transform_train, config):
#     print('Using RandAugment!')
#     transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
#     return transform_train


def randAugment(transform_train, config):
    print('Using RandAugment!')
    try:
        # Try the original way first
        transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    except TypeError:
        # If that fails, try alternate initialization approaches
        try:
            # Some versions of RandAugment only take a single argument
            transform_train.transforms.insert(0, GroupTransform(RandAugment(n=config.data.randaug.N)))
            print('Using RandAugment with n parameter only')
        except TypeError:
            # As a last resort, try with no arguments
            transform_train.transforms.insert(0, GroupTransform(RandAugment()))
            print('Using RandAugment with default parameters')
    
    return transform_train

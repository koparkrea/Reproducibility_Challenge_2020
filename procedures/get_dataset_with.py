import numpy as np
import torch
import torchvision.transforms as transform
import torchvision.datasets as dset



Dataset_Numclasses = {'cifar10' : 10, 'cifar100' : 100, 'imagenet-1k' : 1000}

class CUTOUT(object):
    def __init__(self, cutout_length):
        self.length = cutout_length
    
    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h,w), dtype = np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length //2, 0, h)
        y2 = np.clip(y + self.length //2, 0, h)
        x1 = np.clip(x - self.length //2, 0, w)
        x2 = np.clip(x + self.length //2, 0, w)

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img += mask
        return img


def get_datasets(name, root, cutout_length):

    if name == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif name == 'cifar100':
        mean = [x /255 for x in [129.3, 124.1, 112.4]]
        std = [x /255 for x in [68.2, 65.4, 70.4]]
    elif name.startswith('imagenet-1k'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    
    else:
        raise ValueError('invalid name = {:}'.format(name))

    # data augumentation
    if name == 'cifar10' or name == 'cifar100':
        lists = [transform.RandomHorizontalFlip(), transform.RandomCrop(32, padding = 4), transform.ToTensor(), transform.Normalize(mean, std)]
        if cutout_length > 0:
            lists += CUTOUT(cutout_length)
        train_transform = transform.Compose(lists)
        test_transform = transform.Compose([transform.ToTensor(), transform.Normalize(mean, std)])
        xshape = (1, 3, 32, 32)    
    else:
        raise ValueError('invalid dataset = {:}'.format(name))

    
    # data load
    if name == 'cifar10':
        train_data = dset.CIFAR10(root, train = True, transform = train_transform, download= True)
        test_data = dset.CIFAR10(root, train = False, transform = test_transform, download= True)
        assert len(train_data) == 50000 and len(test_data) == 10000
    elif name == 'cifar100':
        train_data = dset.CIFAR100(root, train = True, transform = train_transform, download= True)
        test_data = dset.CIFAR100(root, train = False, transform = test_transform, download = True)
    
    else:
        raise ValueError('invalid dataset = {:}'.format(name))

    class_num = Dataset_Numclasses[name]
    return train_data, test_data, xshape, class_num


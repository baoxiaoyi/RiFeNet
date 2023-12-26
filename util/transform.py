import random
import math
import numpy as np
import numbers
import collections
import cv2
import torch.nn.functional as F

import torch

manual_seed = 123
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
torch.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)

class Compose(object):
    # Composes segtransforms: segtransform.Compose([segtransform.RandScale([0.5, 2.0]), segtransform.ToTensor()])
    def __init__(self, segtransform):
        self.segtransform = segtransform

    def __call__(self, image, label, padding_mask=None):
        for t in self.segtransform:
            image, label, padding_mask = t(image, label, padding_mask)
        return image, label, padding_mask

import time
class ToTensor(object):
    # Converts numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).
    def __call__(self, image, label, padding_mask=None):
        if not isinstance(image, np.ndarray) or not isinstance(label, np.ndarray):
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray"
                                "[eg: data readed by cv2.imread()].\n"))
        if len(image.shape) > 3 or len(image.shape) < 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray with 3 dims or 2 dims.\n"))
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)
        if not len(label.shape) == 2:
            raise (RuntimeError("segtransform.ToTensor() only handle np.ndarray labellabel with 2 dims.\n"))

        image = torch.from_numpy(image.transpose((2, 0, 1)))
        if not isinstance(image, torch.FloatTensor):
            image = image.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        
        if padding_mask is not None:
            padding_mask = torch.from_numpy(padding_mask)
            if not isinstance(padding_mask, torch.LongTensor):
                padding_mask = padding_mask.long()
        return image, label, padding_mask


class Normalize(object):
    # Normalize tensor with mean and standard deviation along channel: channel = (channel - mean) / std
    def __init__(self, mean, std=None):
        if std is None:
            assert len(mean) > 0
        else:
            assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image, label, padding_mask=None):
        if self.std is None:
            for t, m in zip(image, self.mean):
                t.sub_(m)
        else:
            for t, m, s in zip(image, self.mean, self.std):
                t.sub_(m).div_(s)
        return image, label, padding_mask


class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, padding_mask=None):

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        def find_new_hw(ori_h, ori_w, test_size):
            if ori_h >= ori_w:
                ratio = test_size*1.0 / ori_h
                new_h = test_size
                new_w = int(ori_w * ratio)
            elif ori_w > ori_h:
                ratio = test_size*1.0 / ori_w
                new_h = int(ori_h * ratio)
                new_w = test_size

            if new_h % 8 != 0:
                new_h = (int(new_h /8))*8
            else:
                new_h = new_h
            if new_w % 8 != 0:
                new_w = (int(new_w /8))*8
            else:
                new_w = new_w    
            return new_h, new_w           

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        #new_h, new_w = test_size, test_size
        image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        back_crop = np.zeros((test_size, test_size, 3)) 
        # back_crop[:,:,0] = mean[0]
        # back_crop[:,:,1] = mean[1]
        # back_crop[:,:,2] = mean[2]
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop 

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        #new_h, new_w = test_size, test_size
        s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        if padding_mask is not None:
            s_mask = padding_mask
            new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
            #new_h, new_w = test_size, test_size
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),interpolation=cv2.INTER_NEAREST)
            back_crop_s_mask = np.ones((test_size, test_size)) * 255
            back_crop_s_mask[:new_h, :new_w] = s_mask
            padding_mask = back_crop_s_mask            

        return image, label, padding_mask


class test_Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label, padding_mask=None):

        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]

        def find_new_hw(ori_h, ori_w, test_size):
            if max(ori_h, ori_w) > test_size:
                if ori_h >= ori_w:
                    ratio = test_size*1.0 / ori_h
                    new_h = test_size
                    new_w = int(ori_w * ratio)
                elif ori_w > ori_h:
                    ratio = test_size*1.0 / ori_w
                    new_h = int(ori_h * ratio)
                    new_w = test_size

                if new_h % 8 != 0:
                    new_h = (int(new_h /8))*8
                else:
                    new_h = new_h
                if new_w % 8 != 0:
                    new_w = (int(new_w /8))*8
                else:
                    new_w = new_w    
                return new_h, new_w     
            else:
                return ori_h, ori_w      

        test_size = self.size
        new_h, new_w = find_new_hw(image.shape[0], image.shape[1], test_size)
        if new_w != image.shape[0] or new_h != image.shape[1]:
            image_crop = cv2.resize(image, dsize=(int(new_w), int(new_h)), interpolation=cv2.INTER_LINEAR)
        else:
            image_crop = image.copy()
        back_crop = np.zeros((test_size, test_size, 3)) 
        back_crop[:new_h, :new_w, :] = image_crop
        image = back_crop 

        s_mask = label
        new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
        if new_w != s_mask.shape[0] or new_h != s_mask.shape[1]:
            s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),interpolation=cv2.INTER_NEAREST)
        back_crop_s_mask = np.ones((test_size, test_size)) * 255
        back_crop_s_mask[:new_h, :new_w] = s_mask
        label = back_crop_s_mask

        if padding_mask is not None:
            s_mask = padding_mask
            new_h, new_w = find_new_hw(s_mask.shape[0], s_mask.shape[1], test_size)
            if new_w != s_mask.shape[0] or new_h != s_mask.shape[1]:
                s_mask = cv2.resize(s_mask.astype(np.float32), dsize=(int(new_w), int(new_h)),interpolation=cv2.INTER_NEAREST)
            back_crop_s_mask = np.ones((test_size, test_size)) * 255
            back_crop_s_mask[:new_h, :new_w] = s_mask
            padding_mask = back_crop_s_mask

        return image, label, padding_mask


class RandScale(object):
    # Randomly resize image & label with scale factor in [scale_min, scale_max]
    def __init__(self, scale, aspect_ratio=None):
        assert (isinstance(scale, collections.Iterable) and len(scale) == 2)
        if isinstance(scale, collections.Iterable) and len(scale) == 2 \
                and isinstance(scale[0], numbers.Number) and isinstance(scale[1], numbers.Number) \
                and 0 < scale[0] < scale[1]:
            self.scale = scale
        else:
            raise (RuntimeError("segtransform.RandScale() scale param error.\n"))
        if aspect_ratio is None:
            self.aspect_ratio = aspect_ratio
        elif isinstance(aspect_ratio, collections.Iterable) and len(aspect_ratio) == 2 \
                and isinstance(aspect_ratio[0], numbers.Number) and isinstance(aspect_ratio[1], numbers.Number) \
                and 0 < aspect_ratio[0] < aspect_ratio[1]:
            self.aspect_ratio = aspect_ratio
        else:
            raise (RuntimeError("segtransform.RandScale() aspect_ratio param error.\n"))

    def __call__(self, image, label, padding_mask=None):
        temp_scale = self.scale[0] + (self.scale[1] - self.scale[0]) * random.random()
        temp_aspect_ratio = 1.0
        if self.aspect_ratio is not None:
            temp_aspect_ratio = self.aspect_ratio[0] + (self.aspect_ratio[1] - self.aspect_ratio[0]) * random.random()
            temp_aspect_ratio = math.sqrt(temp_aspect_ratio)
        scale_factor_x = temp_scale * temp_aspect_ratio
        scale_factor_y = temp_scale / temp_aspect_ratio
        image = cv2.resize(image, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        if padding_mask is not None:
            padding_mask = cv2.resize(padding_mask, None, fx=scale_factor_x, fy=scale_factor_y, interpolation=cv2.INTER_NEAREST)
        return image, label, padding_mask


class Crop(object):
    """Crops the given ndarray image (H*W*C or H*W).
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
        int instead of sequence like (h, w), a square crop (size, size) is made.
    """
    def __init__(self, size, crop_type='center', padding=None, ignore_label=255):
        self.size = size
        if isinstance(size, int):
            self.crop_h = size
            self.crop_w = size
        elif isinstance(size, collections.Iterable) and len(size) == 2 \
                and isinstance(size[0], int) and isinstance(size[1], int) \
                and size[0] > 0 and size[1] > 0:
            self.crop_h = size[0]
            self.crop_w = size[1]
        else:
            raise (RuntimeError("crop size error.\n"))
        if crop_type == 'center' or crop_type == 'rand':
            self.crop_type = crop_type
        else:
            raise (RuntimeError("crop type error: rand | center\n"))
        if padding is None:
            self.padding = padding
        elif isinstance(padding, list):
            if all(isinstance(i, numbers.Number) for i in padding):
                self.padding = padding
            else:
                raise (RuntimeError("padding in Crop() should be a number list\n"))
            if len(padding) != 3:
                raise (RuntimeError("padding channel is not equal with 3\n"))
        else:
            raise (RuntimeError("padding in Crop() should be a number list\n"))
        if isinstance(ignore_label, int):
            self.ignore_label = ignore_label
        else:
            raise (RuntimeError("ignore_label should be an integer number\n"))

    def __call__(self, image, label, padding_mask=None):
        h, w = label.shape

        
        pad_h = max(self.crop_h - h, 0)
        pad_w = max(self.crop_w - w, 0)
        pad_h_half = int(pad_h / 2)
        pad_w_half = int(pad_w / 2)
        if pad_h > 0 or pad_w > 0:
            if self.padding is None:
                raise (RuntimeError("segtransform.Crop() need padding while padding argument is None\n"))
            image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
            if padding_mask is not None:
                padding_mask = cv2.copyMakeBorder(padding_mask, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half, cv2.BORDER_CONSTANT, value=self.ignore_label)
        h, w = label.shape
        raw_label = label
        raw_image = image
        raw_padding_mask = padding_mask

        if self.crop_type == 'rand':
            h_off = random.randint(0, h - self.crop_h)
            w_off = random.randint(0, w - self.crop_w)
        else:
            h_off = int((h - self.crop_h) / 2)
            w_off = int((w - self.crop_w) / 2)
        image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        if padding_mask is not None:
            padding_mask = padding_mask[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
        raw_pos_num = np.sum(raw_label == 1)
        pos_num = np.sum(label == 1)
        crop_cnt = 0
        while(pos_num < 0.85*raw_pos_num and crop_cnt<=30):
            image = raw_image
            label = raw_label
            padding_mask = raw_padding_mask
            if self.crop_type == 'rand':
                h_off = random.randint(0, h - self.crop_h)
                w_off = random.randint(0, w - self.crop_w)
            else:
                h_off = int((h - self.crop_h) / 2)
                w_off = int((w - self.crop_w) / 2)
            image = image[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]
            label = label[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]   
            if padding_mask is not None:
                padding_mask = padding_mask[h_off:h_off+self.crop_h, w_off:w_off+self.crop_w]   
            raw_pos_num = np.sum(raw_label == 1)
            pos_num = np.sum(label == 1)  
            crop_cnt += 1
        if crop_cnt >= 50:
            image = cv2.resize(raw_image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(raw_label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)            
            if padding_mask is not None:
                padding_mask = cv2.resize(raw_padding_mask, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)    
                               
        if image.shape != (self.size[0], self.size[0], 3):
            image = cv2.resize(image, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)
            if padding_mask is not None:
                padding_mask = cv2.resize(padding_mask, (self.size[0], self.size[0]), interpolation=cv2.INTER_NEAREST)

        return image, label, padding_mask


class RandRotate(object):
    # Randomly rotate image & label with rotate factor in [rotate_min, rotate_max]
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        assert (isinstance(rotate, collections.Iterable) and len(rotate) == 2)
        if isinstance(rotate[0], numbers.Number) and isinstance(rotate[1], numbers.Number) and rotate[0] < rotate[1]:
            self.rotate = rotate
        else:
            raise (RuntimeError("segtransform.RandRotate() scale param error.\n"))
        assert padding is not None
        assert isinstance(padding, list) and len(padding) == 3
        if all(isinstance(i, numbers.Number) for i in padding):
            self.padding = padding
        else:
            raise (RuntimeError("padding in RandRotate() should be a number list\n"))
        assert isinstance(ignore_label, int)
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label, padding_mask=None):
        if random.random() < self.p:
            angle = self.rotate[0] + (self.rotate[1] - self.rotate[0]) * random.random()
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
            if padding_mask is not None:
                padding_mask = cv2.warpAffine(padding_mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label, padding_mask


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, padding_mask=None):
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)
            if padding_mask is not None:
                padding_mask = cv2.flip(padding_mask, 1)
        return image, label, padding_mask


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label, padding_mask=None):
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)
            if padding_mask is not None:
                padding_mask = cv2.flip(padding_mask, 0)
        return image, label, padding_mask

class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
            To jitter hue, the pixel values of the input image has to be non-negative for conversion to HSV space;
            thus it does not work if you normalize your image to an interval with negative values,
            or use an interpolation that generates negative values before using this function.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")
        self.hue = self._check_input(hue, "hue", center=0, bound=(-0.5, 0.5), clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True):
            if isinstance(value, numbers.Number):
                if value < 0:
                    raise ValueError(f"If {name} is a single number, it must be non negative.")
                value = [center - float(value), center + float(value)]
                if clip_first_on_zero:
                    value[0] = max(value[0], 0.0)
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                if not bound[0] <= value[0] <= value[1] <= bound[1]:
                    raise ValueError(f"{name} values should be between {bound}")
            else:
                raise TypeError(f"{name} should be a single number or a list/tuple with length 2.")

            # if value is 0 or (1., 1.) for brightness/contrast/saturation
            # or (0., 0.) for hue, do nothing
            if value[0] == value[1] == center:
                value = None
            return value

    @ staticmethod
    def get_params(
            brightness,
            contrast,
            saturation,
            hue):
        """Get the parameters for the randomized transform to be applied on image.

        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.

        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img = img / 255.
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:

                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        img = img * 255.
        return img

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"brightness={self.brightness}"
            f", contrast={self.contrast}"
            f", saturation={self.saturation}"
            f", hue={self.hue})"
        )
        return s


class RandomGaussianBlur(object):
    def __init__(self, radius=5):
        self.radius = radius

    def __call__(self, image, label, padding_mask=None):
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label, padding_mask


class RGB2BGR(object):
    # Converts image from RGB order to BGR order, for model initialized from Caffe
    def __call__(self, image, label, padding_mask=None):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, label, padding_mask


class BGR2RGB(object):
    # Converts image from BGR order to RGB order, for model initialized from Pytorch
    def __call__(self, image, label, padding_mask=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, label, padding_mask

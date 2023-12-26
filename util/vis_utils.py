import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import json
import cv2
import os.path as osp


softmax = nn.Softmax(dim=1)

def get_coordinate_tensors(x_max, y_max):
    x_map = np.tile(np.arange(x_max), (y_max,1))/x_max*2 - 1.0
    y_map = np.tile(np.arange(y_max), (x_max,1)).T/y_max*2 - 1.0

    x_map_tensor = torch.from_numpy(x_map.astype(np.float32)).cuda()
    y_map_tensor = torch.from_numpy(y_map.astype(np.float32)).cuda()

    return x_map_tensor, y_map_tensor


def get_center(part_map, self_referenced=False):

    h,w = part_map.shape
    x_map, y_map = get_coordinate_tensors(h,w)

    x_center = (part_map * x_map).sum()
    y_center = (part_map * y_map).sum()

    if self_referenced:
        x_c_value = float(x_center.cpu().detach())
        y_c_value = float(y_center.cpu().detach())
        x_center = (part_map * (x_map - x_c_value)).sum() + x_c_value
        y_center = (part_map * (y_map - y_c_value)).sum() + y_c_value

    return x_center, y_center

def get_centers(part_maps, detach_k=True, epsilon=1e-3, self_ref_coord=False):
    C,H,W = part_maps.shape
    centers = []
    for c in range(C):
        part_map = part_maps[c,:,:] + epsilon
        k = part_map.sum()
        part_map_pdf = part_map/k
        x_c, y_c = get_center(part_map_pdf, self_ref_coord)
        centers.append(torch.stack((x_c, y_c), dim=0).unsqueeze(0))
    return torch.cat(centers, dim=0)

def batch_get_centers(pred_softmax):
    B,C,H,W = pred_softmax.shape

    centers_list = []
    for b in range(B):
        centers_list.append(get_centers(pred_softmax[b]).unsqueeze(0))
    return torch.cat(centers_list, dim=0)

def color_map(N=256, normalized=True):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def Batch_Draw_Landmarks(imgs, pred, sm=True):

    B,C,H,W = pred.shape
    cmap = color_map(40,normalized=False)

    if sm:
        pred_softmax = softmax(pred)
    else:
        pred_softmax = pred

    imgs_cv2 = imgs.detach().cpu().numpy().transpose(0,2,3,1).astype(np.uint8)

    centers = np.zeros((B,C-1,2))

    part_response = np.zeros((B,C-1,H,W,3)).astype(np.uint8)
    part_response_gradient =np.zeros((B,C-1,H,W,3)).astype(np.uint8)

    for b in range(B):
        for c in range(1,C):

            # normalize part map as spatial pdf
            part_map = pred_softmax[b,c,:,:]
            k = float(part_map.sum())
            part_map_pdf = part_map/k

            response_map = part_map_pdf.detach().cpu().numpy()
            response_map = response_map/response_map.max()

            response_map = cv2.applyColorMap((response_map*255.0).astype(np.uint8), cv2.COLORMAP_HOT)[:,:,::-1] # BGR->RGB

            part_response[b,c-1,:,:,:] = response_map.astype(np.uint8)

            x_c, y_c = get_center(part_map_pdf)


            centers[b,c-1,:] = [x_c/2,y_c/2]

            x_c = (x_c+1.0)/2*W
            y_c = (y_c+1.0)/2*H


            img = imgs_cv2[b].copy()
            cv2.drawMarker(img, (x_c,y_c), (int(cmap[c][0]), int(cmap[c][1]), int(cmap[c][2])), markerType=cv2.MARKER_CROSS, markerSize = 10, thickness=2)
            imgs_cv2[b] = img

    return imgs_cv2.transpose(0,3,1,2), centers, part_response.transpose(0,1,4,2,3), part_response_gradient.transpose(0,1,4,2,3)



class Colorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image


class BatchColorize(object):
    def __init__(self, n=40):
        self.cmap = color_map(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((size[0], 3, size[1], size[2]), dtype=np.float32)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[:,0][mask] = self.cmap[label][0]
            color_image[:,1][mask] = self.cmap[label][1]
            color_image[:,2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[:,0][mask] = color_image[:,1][mask] = color_image[:,2][mask] = 255

        return color_image

if __name__ == "__main__":
    print(color_map(10))
import pdb
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

class HSVColorInstSegment():
    '''
    Description:
        Given an image batch (b, h, w, 3), segment all instances in this batch based on color filtering (there is at most one object for each color in the same image).
    '''
    def __init__(self, ):
        filter_cfg = {
            'red': [(-15, 0.3, 0.2), (15, 1.0, 1.0)],   # The correct hue range is 0~15 and 345~360
            'green': [(100, 0.3, 0.2), (140, 1.0, 1.0)],
            'blue': [(220, 0.3, 0.2), (260, 1.0, 1.0)],
            'purple': [(280, 0.3, 0.2), (320, 1.0, 1.0)],
            'yellow': [(40, 0.3, 0.2), (80, 1.0, 1.0)],
            'cyan': [(160, 0.3, 0.2), (200, 1.0, 1.0)],
        }
        self.filter_cfg = {k: torch.Tensor(v).cuda() for k, v in filter_cfg.items()}
        
    def __call__(self, vision_obs_dict):
        top_rgb, hand_rgb = vision_obs_dict['top_rgb'], vision_obs_dict['hand_rgb']
        imgs = torch.cat((top_rgb, hand_rgb), dim  = 0) # Left shape: (2, h, w, 3)
        inst_seg_results = self.forward(imgs)
        
        result_dict = dict(
            top_inst_seg = inst_seg_results[0],   # Left: A list with each element as a segmentation instance. Each element is a turple in the format of (cls, seg_mask)
            hand_inst_seg = inst_seg_results[1],
        )
        return result_dict

    def forward(self, imgs):
        '''
        Input:
            img: normalized torch tensor with the shape of (bs, img_h, img_w, 3)
        '''
        norm_img = imgs / 255   # Left shape: (bs, img_h, img_w, 3)
        hsv_imgs = torch_rgb_to_hsv(norm_img.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(1)  # Left shape: (bs, 1, img_h, img_w, 3)
        hsv_imgs[[hsv_imgs[..., 0] > 345]] -= 360   # Convert the hue range of red from 345~360 to -15~0
        
        color_thre_min = torch.stack([ele[0] for ele in self.filter_cfg.values()])[None, :, None, None, :]  # Left shape: (1, num_color, 1, 1, 3)
        color_thre_max = torch.stack([ele[1] for ele in self.filter_cfg.values()])[None, :, None, None, :]   # Left shape: (1, num_color, 1, 1, 3)
        obj_mask = (hsv_imgs >= color_thre_min) & (hsv_imgs <= color_thre_max) # Left shape: (bs, num_color, img_h, img_w, 3)
        obj_mask = torch.all(obj_mask, dim = -1) # Left shape: (bs, num_color, img_h, img_w)
        
        inst_seg_results = []
        for bs_id in range(obj_mask.shape[0]):
            inst_seg_results.append([])
            for color_id in range(obj_mask.shape[1]):
                if obj_mask[bs_id, color_id].sum() > 3:
                    inst_seg_results[bs_id].append((list(self.filter_cfg.keys())[color_id], obj_mask[bs_id, color_id]))
        return inst_seg_results

def torch_rgb_to_hsv(image):
    """
    Input:
        image: normalized image tensor with the shape of (B, 3, H, W). Pixel value range: (0, 1)
    Output:
        hsv: The converted HSV image with the shape of (B, 3, H, W). Hue range is (0, 360) and the ranges of value and Saturation are (0, 1)
    """
    max_val, _ = torch.max(image, dim=1, keepdim=True)  # (B, 1, H, W)
    min_val, _ = torch.min(image, dim=1, keepdim=True)  # (B, 1, H, W)
    # Compute Value
    V = max_val  # V = max(R, G, B)
    # Compute Saturation
    delta = max_val - min_val
    S = delta / (max_val + 1e-7)
    S[delta == 0] = 0
    # Compute Hue
    R, G, B = image[:, 0:1, :, :], image[:, 1:2, :, :], image[:, 2:3, :, :]
    H = torch.zeros_like(V)

    mask = (max_val == R)
    H[mask] = 60 * ((G[mask] - B[mask]) / (delta[mask] + 1e-7) % 6)
    mask = (max_val == G)
    H[mask] = 60 * ((B[mask] - R[mask]) / (delta[mask] + 1e-7) + 2)
    mask = (max_val == B)
    H[mask] = 60 * ((R[mask] - G[mask]) / (delta[mask] + 1e-7) + 4)
    H[delta == 0] = 0
    hsv_image = torch.cat([H, S, V], dim=1)
    return hsv_image

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
            and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)
        if len(x) == 0:
            bounding_boxes[index, 0] = 0
            bounding_boxes[index, 1] = 0
            bounding_boxes[index, 2] = 0
            bounding_boxes[index, 3] = 0
        else:
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes

def rescale_box(det_box, img_shape, scale_ratio = 1.0):
    '''
    Input:
        det_box: torch tensor with the shape of (bs, 4)
        img_shape: (img_w, img_h)
    Output:
        det_box: torch tensor with the shape of (bs, 4)
    '''
    bs = det_box.shape[0]
    img_w, img_h = img_shape

    det_box = det_box.view(bs, 2, 2)
    det_box_center = det_box.mean(dim = 1)[:, None]   # Left shape: (bs, 1, 2)
    det_box_wh = (det_box[:, 1] - det_box[:, 0])[:, None] # Left shape: (bs, 1, 2)
    det_box = torch.cat((det_box_center - det_box_wh / 2 * scale_ratio, det_box_center + det_box_wh / 2 * scale_ratio), dim = 1)    # Left shape: (bs, 2, 2)

    det_box[:, :, 0] = torch.clamp(det_box[:, :, 0], min = 0, max = img_w - 1)
    det_box[:, :, 1] = torch.clamp(det_box[:, :, 1], min = 0, max = img_h - 1)
    return det_box.view(bs, 4)

if __name__ == '__main__':
    video_path = '/home/cvte/twilight/code/act/datasets/isaac_singlecolorbox/exterior_camera1/episode_0.mp4'
    filter_cfg = {
        'red': [(0.2, -0.01, -0.01), (1.01, 0.1, 0.1)],
        'green': [(-0.01, 0.2, -0.01), (0.1, 1.01, 0.1)],
        'blue': [(-0.01, -0.01, 0.2), (0.1, 0.1, 1.2)],
        'purple': [(0.2, -0.01, 0.2), (1.01, 0.1, 1.01)],
        'yellow': [(0.2, 0.2, -0.01), (1.01, 1.01, 0.1)],
    }
    color_thre = filter_cfg['yellow']

    read_cap = cv2.VideoCapture(video_path)
    write_cap = cv2.VideoWriter('vis.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (640, 240))

    while True:
        ret, frame = read_cap.read()
        if not ret:
            break
        rgb_img = frame[:, :, ::-1]
        norm_img = torch.Tensor(rgb_img.astype(np.float32) / 255)
        color_min_thre = torch.Tensor(color_thre[0])[None][None]
        color_max_thre = torch.Tensor(color_thre[1])[None][None]
        obj_mask = (norm_img >= color_min_thre) & (norm_img <= color_max_thre) # Left shape: (img_h, img_w, 3)
        ch, img_h, img_w = obj_mask.shape
        obj_mask = torch.all(obj_mask, dim = 2) # Left shape: (img_h, img_w)
        det_box = masks_to_boxes(obj_mask[None])[0]  # Left shape: (4,)

        vis_img = frame
        cv2.rectangle(vis_img, (int(det_box[0]), int(det_box[1])), (int(det_box[2]), int(det_box[3])), (255, 255, 0), 1)
        vis_mask = obj_mask.numpy()
        vis_mask = np.concatenate((vis_mask[:, :, None].astype(np.uint8) * 255, np.zeros((vis_mask.shape[0], vis_mask.shape[1], 2), dtype = np.uint8)), axis = 2)
        vis = np.concatenate((vis_img, vis_mask), axis = 1)
        write_cap.write(vis)
        
    write_cap.release()
    write_cap.release()
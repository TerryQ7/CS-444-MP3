import torch
import numpy as np
import random

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    
def get_detections(outs):
    B, BB, _, _ = outs[0][0].shape
    _, A, _, _ = outs[0][2].shape
    A = A // 4
    num_classes = BB // A
    
    pred_bboxes, pred_clss, anchors = [], [], []
    for pred_cls, pred_bbox, anchor in outs:
        # Get all the anchors, pred and bboxes
        H, W = pred_cls.shape[-2:]
        pred_cls = pred_cls.reshape(B, A, -1, H, W)
        pred_bbox = pred_bbox.reshape(B, A, -1, H, W)

        pred_clss.append(pred_cls.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes))
        pred_bboxes.append(pred_bbox.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))

        anchor = anchor.reshape(B, A, -1, H, W)
        anchors.append(anchor.permute(0, 1, 3, 4, 2).reshape(B, -1, 4))
    pred_clss = torch.cat(pred_clss, dim=1)
    pred_bboxes = torch.cat(pred_bboxes, dim=1)
    anchors = torch.cat(anchors, dim=1)
    return pred_clss, pred_bboxes, anchors

def compute_bbox_iou(bbox1, bbox2, dim=1):
    """
    Args:
        bbox1: (N, 4) tensor of (x1, y1, x2, y2)
        bbox2: (M, 4) tensor of (x1, y1, x2, y2)
    Returns
        iou: (N, M) tensor of IoU values
    """
    bbox1 = bbox1.unsqueeze(1)
    bbox2 = bbox2.unsqueeze(0)
    max_min_x = torch.max(bbox1[...,0], bbox2[...,0])
    min_max_x = torch.min(bbox1[...,2], bbox2[...,2])
    max_min_y = torch.max(bbox1[...,1], bbox2[...,1])
    min_max_y = torch.min(bbox1[...,3], bbox2[...,3])
    intersection = torch.clamp(min_max_x - max_min_x, min=0) * torch.clamp(min_max_y - max_min_y, min=0)
    area1 = (bbox1[...,2] - bbox1[...,0]) * (bbox1[...,3] - bbox1[...,1])
    area2 = (bbox2[...,2] - bbox2[...,0]) * (bbox2[...,3] - bbox2[...,1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

def compute_targets(anchor, cls, bbox):
    """
    Args:
        anchor: batch of anchors in the format (x1, y1, x2, y2) or in other words (xmin, ymin, xmax, ymax); shape is (B, A, 4), where B denotes image batch size and A denotes the number of anchors
        cls: groundtruth object classes of shape (B, number of objects in the image, 1)
        bbox: groundtruth bounding boxes of shape (B, number of objects in the image, 4)
    Returns:
        gt_clss: groundtruth class targets of shape (B, A, 1)
        gt_bboxes: groundtruth bbox targets of shape (B, A, 4)
    
    Hint: remember if the max_iou for that bounding box is between [0, 0.4) then the gt_cls should equal 0(because it is being assigned as background) and the
    gt_bbox should be all zero(it can be anything since it will be ignored however our tests set them to zero so you should too).
    Also, if the max iou is between [0.4, 0.5) then the gt_cls should be equal to -1(since it's neither background or assigned to a class. This is basically tells the model to ignore this box) 
    and the gt_bbox should again arbitrarilarly be set to all zeros).
    Otherwise if the max_iou > 0.5, you should assign the anchor to the gt_box with the max iou, and the gt_cls will be the ground truth class of that max_iou box
    Hint: use torch.max to get both the max iou and the index of the max iou.

    Hint: We recommend using the compute_bbox_iou function which efficently computes the ious between two lists of bounding boxes as a helper function.

    Hint: make sure that the returned gt_clss tensor is of type int(since it will be used as an index in the loss function). Also make sure that both the gt_bboxes and gt_clss are on the same device as the anchor. 
    You can do this by calling .to(anchor.device) on the tensor you want to move to the same device as the anchor.

    VECTORIZING CODE: Again, you can use for loops initially to make the tests pass, but in order to make your code efficient 
    during training, you should only have one for loop over the batch dimension and everything else should be vectorized. We recommend using boolean masks to do this. i.e
    you can compute the max_ious for all the anchor boxes and then do gt_cls[max_iou < 0.4] = 0 to access all the anchor boxes that should be set to background and setting their gt_cls to 0. 
    This will remove the need for a for loop over all the anchor boxes. You can then do the same for the other cases. This will make your code much more efficient and faster to train.
    """
    # TODO(student): Complete this function
    
    B, A, _ = anchor.shape
    device = anchor.device

    gt_clss = torch.zeros(B, A, 1, dtype=torch.long, device=device)
    gt_bboxes = torch.zeros(B, A, 4, device=device)

    for b in range(B):
        anchors_per_image = anchor[b]  # (A, 4)
        cls_per_image = cls[b].to(torch.long)  # (num_objs, 1)
        bbox_per_image = bbox[b]  # (num_objs, 4)

        if bbox_per_image.shape[0] == 0:
            # 如果图像中没有目标
            max_iou = torch.zeros(A, device=device)
            max_idx = torch.zeros(A, dtype=torch.long, device=device)
        else:
            # 计算每个锚点与真实边界框的IoU
            iou = compute_bbox_iou(anchors_per_image, bbox_per_image)  # (A, num_objs)
            max_iou, max_idx = torch.max(iou, dim=1)  # (A,), (A,)

        # 创建掩码
        mask_bg = max_iou < 0.4
        mask_ignore = (max_iou >= 0.4) & (max_iou < 0.5)
        mask_fg = max_iou >= 0.5

        # 背景锚点，类别设为0
        gt_clss[b][mask_bg] = 0

        # 忽略的锚点，类别设为-1
        gt_clss[b][mask_ignore] = -1

        if bbox_per_image.shape[0] > 0:
            # 前景锚点，设置类别和边界框
            gt_clss[b][mask_fg] = cls_per_image[max_idx[mask_fg]]
            gt_bboxes[b][mask_fg] = bbox_per_image[max_idx[mask_fg]]

    return gt_clss.to(torch.int), gt_bboxes

def compute_bbox_targets(anchors, gt_bboxes):
    """
    Args:
        anchors: anchors of shape (A, 4)
        gt_bboxes: groundtruth object classes of shape (A, 4)
    Returns:
        bbox_reg_target: regression offset of shape (A, 4)
    
    Remember that the delta_x and delta_y we compute are with respect to the center of the anchor box. I.E, we're seeing how much that center of the anchor box changes. 
    We also need to normalize delta_x and delta_y which means that we need to divide them by the width or height of the anchor box respectively. This is to make
    our regression targets more invariant to the size of the original anchor box. So, this means that:
    delta_x = (gt_bbox_center_x - anchor_center_x) / anchor_width  and delta_y would be computed in a similar manner.

    When computing delta_w and delta_h, there are a few things to note.
    1. We also want to normalize these with respect to the width and height of the anchor boxes. so delta_w = gt_bbox_width / anchor_width
    2. Logarithm: In order to make our regresssion targets better handle varying sizees of the bounding boxes, we use the logarithmic scale for our delta_w and delta_h
       This is to ensure that if for example the gt_width is twice or 1/2 the size of the anchor_width, the magnitude in the log scale would stay the same but only the sign of
       our regression target would be different. Therefore our formula changes to delta_w = log(gt_bbox_width / anchor_width)
    3. Clamping: Remember that logarithms can't handle negative values and that the log of values very close to zero will have very large magnitudes and have extremly 
       high gradients which might make training unstable. To mitigate this we use clamping to ensure that the value that we log isn't too small. Therefore, our final formula will be
       delta_w = log(max(gt_bbox_width,1) / anchor_width)
       
    """
    # TODO(student): Complete this function

    # 锚点框的中心和尺寸
    anchor_center_x = (anchors[:, 0] + anchors[:, 2]) / 2.0
    anchor_center_y = (anchors[:, 1] + anchors[:, 3]) / 2.0
    anchor_width = anchors[:, 2] - anchors[:, 0]
    anchor_height = anchors[:, 3] - anchors[:, 1]

    # 真实边界框的中心和尺寸
    gt_center_x = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_center_y = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]

    # 计算回归目标
    delta_x = (gt_center_x - anchor_center_x) / anchor_width
    delta_y = (gt_center_y - anchor_center_y) / anchor_height

    # 防止宽度和高度过小导致数值不稳定
    gt_width = torch.clamp(gt_width, min=1.0)
    gt_height = torch.clamp(gt_height, min=1.0)

    delta_w = torch.log(gt_width / anchor_width)
    delta_h = torch.log(gt_height / anchor_height)

    bbox_reg_target = torch.stack([delta_x, delta_y, delta_w, delta_h], dim=-1)

    return bbox_reg_target

def apply_bbox_deltas(boxes, deltas):
    """
    Args:
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        deltas: (N, 4) tensor of (dxc, dyc, dlogw, dlogh)
    Returns
        boxes: (N, 4) tensor of (x1, y1, x2, y2)
        
    """
    # TODO(student): Complete this function
    # 锚点框的中心和尺寸
    anchor_center_x = (boxes[:, 0] + boxes[:, 2]) / 2.0
    anchor_center_y = (boxes[:, 1] + boxes[:, 3]) / 2.0
    anchor_width = boxes[:, 2] - boxes[:, 0]
    anchor_height = boxes[:, 3] - boxes[:, 1]

    # 提取偏移量
    delta_x = deltas[:, 0]
    delta_y = deltas[:, 1]
    delta_w = deltas[:, 2]
    delta_h = deltas[:, 3]

    # 计算新的中心和尺寸
    pred_center_x = delta_x * anchor_width + anchor_center_x
    pred_center_y = delta_y * anchor_height + anchor_center_y
    pred_width = anchor_width * torch.exp(delta_w)
    pred_height = anchor_height * torch.exp(delta_h)

    # 计算新的边界框坐标
    x1 = pred_center_x - pred_width / 2.0
    y1 = pred_center_y - pred_height / 2.0
    x2 = pred_center_x + pred_width / 2.0
    y2 = pred_center_y + pred_height / 2.0

    new_boxes = torch.stack([x1, y1, x2, y2], dim=-1)

    return new_boxes

def nms(bboxes, scores, threshold=0.5):
    """
    Args:
        bboxes: (N, 4) tensor of (x1, y1, x2, y2)
        scores: (N,) tensor of scores
    Returns:
        keep: (K,) tensor of indices to keep
    
    Remember that nms is used to prevent having many boxes that overlap each other. To do this, if multiple boxes overlap each other beyond a
    threshold iou, nms will pick the "best" box(the one with the highest score) and remove the rest. One way to implement this is to
    first compute the ious between all pairs of bboxes. Then loop over the bboxes from highest score to lowest score. Since this is the 
    best bbox(the one with the highest score), It will be choosen over all overlapping boxes. Therefore, you should add this bbox to your final 
    resulting bboxes and remove all the boxes that overlap with it from consideration. Then repeat until you've gone through all of the bboxes.

    make sure that the indices tensor that you return is of type int or long(since it will be used as an index to select the relevant bboxes to output)
    """
    # TODO(student): Complete this function
    scores, idxs = scores.sort(descending=True)
    bboxes = bboxes[idxs]

    keep = []

    while idxs.numel() > 0:
        i = idxs[0].item()
        keep.append(i)

        if idxs.numel() == 1:
            break

        # 计算当前框与剩余框的IoU
        ious = compute_bbox_iou(bboxes[0].unsqueeze(0), bboxes[1:])[0]

        # 保留IoU小于阈值的框
        idxs = idxs[1:][ious <= threshold]
        bboxes = bboxes[1:][ious <= threshold]

    keep = torch.tensor(keep, dtype=torch.long, device=bboxes.device)

    return keep

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'UncertaintyLoss', 'UncertaintyRegularizationLoss',
           'BoundaryLoss', 'DeepSupervisionLoss', 'CombinedLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)
        return loss


class BoundaryLoss(nn.Module):
    """Boundary Loss - uses a Sobel operator to extract boundaries and compute an
    MSE between predicted and ground-truth boundary maps."""
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, input, target):
        if self.sobel_x.device != input.device:
            self.sobel_x = self.sobel_x.to(input.device)
            self.sobel_y = self.sobel_y.to(input.device)
        input_prob = torch.sigmoid(input)
        ibx = F.conv2d(input_prob, self.sobel_x, padding=1)
        iby = F.conv2d(input_prob, self.sobel_y, padding=1)
        input_boundary = torch.sqrt(ibx ** 2 + iby ** 2 + 1e-6)
        tbx = F.conv2d(target, self.sobel_x, padding=1)
        tby = F.conv2d(target, self.sobel_y, padding=1)
        target_boundary = torch.sqrt(tbx ** 2 + tby ** 2 + 1e-6)
        return F.mse_loss(input_boundary, target_boundary)


class UncertaintyRegularizationLoss(nn.Module):
    """Encourages low uncertainty in confident-correct regions, high uncertainty in
    incorrect regions, and moderate uncertainty along boundaries."""
    def __init__(self, boundary_weight=2.0):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def forward(self, prediction, target, uncertainty_map):
        if self.sobel_x.device != prediction.device:
            self.sobel_x = self.sobel_x.to(prediction.device)
            self.sobel_y = self.sobel_y.to(prediction.device)
        pred_prob = torch.sigmoid(prediction)
        prediction_error = torch.abs(pred_prob - target)
        tbx = F.conv2d(target, self.sobel_x, padding=1)
        tby = F.conv2d(target, self.sobel_y, padding=1)
        boundary_mask = torch.sqrt(tbx ** 2 + tby ** 2 + 1e-6)
        boundary_mask = (boundary_mask > 0.1).float()
        non_boundary_mask = 1.0 - boundary_mask
        correct_mask = (prediction_error < 0.2).float() * non_boundary_mask
        correct_uncertainty_loss = (uncertainty_map * correct_mask).mean()
        incorrect_mask = (prediction_error >= 0.2).float() * non_boundary_mask
        incorrect_uncertainty_loss = ((1.0 - uncertainty_map) * incorrect_mask * prediction_error).mean()
        boundary_uncertainty_target = 0.6
        boundary_uncertainty_loss = F.mse_loss(
            uncertainty_map * boundary_mask,
            torch.ones_like(uncertainty_map) * boundary_uncertainty_target * boundary_mask
        )
        total_loss = (correct_uncertainty_loss +
                      0.5 * incorrect_uncertainty_loss +
                      self.boundary_weight * boundary_uncertainty_loss)
        return total_loss


class ThresholdRegularizationLoss(nn.Module):
    """Prevents learnable thresholds from degenerating to extreme values."""
    def __init__(self, target_min=0.2, target_max=0.8):
        super().__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, thresholds):
        total_loss = 0.0
        for threshold in thresholds:
            threshold_value = torch.sigmoid(threshold)
            if threshold_value < self.target_min:
                total_loss += (self.target_min - threshold_value) ** 2
            if threshold_value > self.target_max:
                total_loss += (threshold_value - self.target_max) ** 2
        return total_loss / len(thresholds)


class UncertaintyConsistencyLoss(nn.Module):
    """Ensures uncertainty maps from successive decoder stages are consistent
    (deeper uncertainty guides shallower uncertainty)."""
    def __init__(self):
        super().__init__()

    def forward(self, uncertainty_maps):
        if len(uncertainty_maps) < 2:
            return torch.tensor(0.0, device=uncertainty_maps[0].device)
        total_loss = 0.0
        for i in range(len(uncertainty_maps) - 1):
            deeper_map = uncertainty_maps[i]
            shallower_map = uncertainty_maps[i + 1]
            if deeper_map.shape != shallower_map.shape:
                deeper_map = F.interpolate(deeper_map, size=shallower_map.shape[2:],
                                           mode='bilinear', align_corners=True)
            total_loss += F.mse_loss(shallower_map, deeper_map)
        return total_loss / (len(uncertainty_maps) - 1)


class UncertaintyLoss(nn.Module):
    """Aggregates the uncertainty regularization, threshold regularization, and
    multi-stage consistency terms."""
    def __init__(self, use_regularization=True, use_threshold_reg=True, use_consistency=True,
                 reg_weight=0.1, threshold_reg_weight=0.01, consistency_weight=0.05):
        super().__init__()
        self.use_regularization = use_regularization
        self.use_threshold_reg = use_threshold_reg
        self.use_consistency = use_consistency
        self.reg_weight = reg_weight
        self.threshold_reg_weight = threshold_reg_weight
        self.consistency_weight = consistency_weight
        if use_regularization:
            self.uncertainty_reg_loss = UncertaintyRegularizationLoss()
        if use_threshold_reg:
            self.threshold_reg_loss = ThresholdRegularizationLoss()
        if use_consistency:
            self.consistency_loss = UncertaintyConsistencyLoss()

    def forward(self, prediction, target, uncertainty_maps, learnable_thresholds=None):
        total_loss = 0.0
        loss_dict = {}
        if not isinstance(uncertainty_maps, list):
            uncertainty_maps = [uncertainty_maps]
        if self.use_regularization:
            reg_loss = 0.0
            for uncertainty_map in uncertainty_maps:
                if uncertainty_map.shape != prediction.shape:
                    uncertainty_map = F.interpolate(uncertainty_map, size=prediction.shape[2:],
                                                    mode='bilinear', align_corners=True)
                reg_loss += self.uncertainty_reg_loss(prediction, target, uncertainty_map)
            reg_loss = reg_loss / len(uncertainty_maps)
            total_loss += self.reg_weight * reg_loss
            loss_dict['uncertainty_reg'] = reg_loss.item()
        if self.use_threshold_reg and learnable_thresholds is not None:
            threshold_loss = self.threshold_reg_loss(learnable_thresholds)
            total_loss += self.threshold_reg_weight * threshold_loss
            loss_dict['threshold_reg'] = threshold_loss.item()
        if self.use_consistency and len(uncertainty_maps) > 1:
            consistency_loss = self.consistency_loss(uncertainty_maps)
            total_loss += self.consistency_weight * consistency_loss
            loss_dict['uncertainty_consistency'] = consistency_loss.item()
        loss_dict['uncertainty_total'] = total_loss.item() if torch.is_tensor(total_loss) else total_loss
        return total_loss, loss_dict


class DeepSupervisionLoss(nn.Module):
    """Weighted loss over the main output and auxiliary deep-supervision outputs."""
    def __init__(self, base_loss=None, weights=None):
        super().__init__()
        self.base_loss = base_loss if base_loss is not None else BCEDiceLoss()
        self.weights = weights if weights is not None else [1.0, 0.8, 0.6, 0.4]

    def forward(self, outputs, target):
        if isinstance(outputs, (tuple, list)):
            main_output, aux_outputs = outputs[0], outputs[1:]
            total_loss = self.weights[0] * self.base_loss(main_output, target)
            for i, aux_output in enumerate(aux_outputs):
                weight_idx = min(i + 1, len(self.weights) - 1)
                total_loss += self.weights[weight_idx] * self.base_loss(aux_output, target)
            return total_loss
        else:
            return self.base_loss(outputs, target)


class CombinedLoss(nn.Module):
    """Composite objective: segmentation (BCE+Dice, with deep supervision) +
    boundary-aware loss + uncertainty-guided losses. Compatible with a plain
    single-output model (pass only outputs and target) and with KGC-UKAN
    (also pass uncertainty_maps)."""
    def __init__(self, boundary_weight=0.2, deep_supervision_weights=None,
                 use_boundary=True, use_uncertainty_loss=True,
                 uncertainty_loss_weight=0.1, uncertainty_config=None):
        super().__init__()
        self.use_boundary = use_boundary
        self.boundary_weight = boundary_weight
        self.use_uncertainty_loss = use_uncertainty_loss
        self.uncertainty_loss_weight = uncertainty_loss_weight
        self.base_loss = BCEDiceLoss()
        if self.use_boundary:
            self.boundary_loss = BoundaryLoss()
        self.deep_supervision_loss = DeepSupervisionLoss(base_loss=self.base_loss,
                                                         weights=deep_supervision_weights)
        if self.use_uncertainty_loss:
            uncertainty_config = uncertainty_config or {}
            self.uncertainty_loss = UncertaintyLoss(**uncertainty_config)

    def forward(self, outputs, target, uncertainty_maps=None, learnable_thresholds=None):
        loss_dict = {}
        ds_loss = self.deep_supervision_loss(outputs, target)
        total_loss = ds_loss
        loss_dict['segmentation'] = ds_loss.item()
        main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        if self.use_boundary:
            boundary_loss = self.boundary_loss(main_output, target)
            total_loss = total_loss + self.boundary_weight * boundary_loss
            loss_dict['boundary'] = boundary_loss.item()
        if self.use_uncertainty_loss and uncertainty_maps is not None:
            uncertainty_loss, uncertainty_loss_dict = self.uncertainty_loss(
                main_output, target, uncertainty_maps, learnable_thresholds)
            total_loss = total_loss + self.uncertainty_loss_weight * uncertainty_loss
            loss_dict.update(uncertainty_loss_dict)
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math
# from pytorch_msssim import ms_ssim


# # ============================================================
# # 1️⃣ Standard MSE-Based Rate–Distortion Loss
# # ============================================================
# class RateDistortionLoss(nn.Module):
#     """
#     Standard Rate–Distortion Loss used in learned compression.

#     L = λ * 255^2 * MSE + BPP

#     Use this when reporting PSNR.
#     """

#     def __init__(self, lmbda=0.01):
#         super().__init__()
#         self.lmbda = lmbda

#     def forward(self, output, target):
#         """
#         output: dict from model containing:
#             - "x_hat"
#             - "likelihoods"
#         target: ground truth image
#         """

#         N, C, H, W = target.size()
#         num_pixels = N * H * W

#         # -------------------------
#         # Rate (Bits Per Pixel)
#         # -------------------------
#         bpp_loss = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in output["likelihoods"].values()
#         )

#         # -------------------------
#         # Distortion (MSE)
#         # -------------------------
#         mse_loss = F.mse_loss(output["x_hat"], target)

#         # -------------------------
#         # Final RD Loss
#         # -------------------------
#         loss = self.lmbda * 255**2 * mse_loss + bpp_loss

#         return {
#             "loss": loss,
#             "mse_loss": mse_loss,
#             "bpp_loss": bpp_loss,
#         }


# # ============================================================
# # 2️⃣ MS-SSIM-Based Rate–Distortion Loss (Optional)
# # ============================================================
# class MSSSIMRateDistortionLoss(nn.Module):
#     """
#     Perceptual Rate–Distortion Loss.

#     L = λ * (1 - MS-SSIM) + BPP

#     Use this when reporting MS-SSIM as primary metric.
#     """

#     def __init__(self, lmbda=0.01):
#         super().__init__()
#         self.lmbda = lmbda

#     def forward(self, output, target):

#         N, C, H, W = target.size()
#         num_pixels = N * H * W

#         # -------------------------
#         # Rate (Bits Per Pixel)
#         # -------------------------
#         bpp_loss = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in output["likelihoods"].values()
#         )

#         # -------------------------
#         # Distortion (MS-SSIM)
#         # -------------------------
#         msssim_val = ms_ssim(
#             output["x_hat"].clamp(0, 1),
#             target,
#             data_range=1.0,
#             size_average=True,
#         )

#         msssim_loss = 1 - msssim_val

#         # -------------------------
#         # Final RD Loss
#         # -------------------------
#         loss = self.lmbda * msssim_loss + bpp_loss

#         return {
#             "loss": loss,
#             "bpp_loss": bpp_loss,
#             "msssim_loss": msssim_loss,
#         }

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_msssim import ms_ssim


# ============================================================
# Shared helpers
# ============================================================
def compute_bpp(likelihoods, num_pixels):
    """Bits per pixel from a dict of likelihood tensors."""
    return sum(
        (torch.log(lk).sum() / (-math.log(2) * num_pixels))
        for lk in likelihoods.values()
    )


def comparable_loss(x_hat, target, likelihoods, lmbda=0.01):
    """
    Unified, scale-comparable reporting loss for the results table.

        L_report = lmbda * (1 - MS-SSIM) + BPP

    Every method in the comparison table (JPEG, WebP, NIC, GAN,
    Scale Hyperprior, ECM, ...) should be scored with THIS function
    so the LOSS column is apples-to-apples. It does NOT use the
    lmbda * 255^2 * MSE weighting, which inflates the value by ~65000x
    and is only meaningful as a training signal, not a cross-method score.
    """
    N, C, H, W = target.size()
    num_pixels = N * H * W

    bpp = compute_bpp(likelihoods, num_pixels) if likelihoods is not None else 0.0

    msssim_val = ms_ssim(
        x_hat.clamp(0, 1), target, data_range=1.0, size_average=True
    )
    distortion = 1.0 - msssim_val

    return lmbda * distortion + bpp


# ============================================================
# 1) Standard MSE-Based Rate-Distortion Loss (TRAINING)
# ============================================================
class RateDistortionLoss(nn.Module):
    """
    Standard Rate-Distortion Loss used in learned compression.

        L = lmbda * 255^2 * MSE + BPP

    Use this to TRAIN when targeting PSNR. NOTE: the value of `loss`
    here is NOT comparable to the table's LOSS column because of the
    255^2 weighting. Use `report_loss` for the table.
    """

    def __init__(self, lmbda=0.1, report_lmbda=0.1):
        super().__init__()
        self.lmbda = lmbda
        self.report_lmbda = report_lmbda

    def forward(self, output, target):
        N, C, H, W = target.size()
        num_pixels = N * H * W

        bpp_loss = compute_bpp(output["likelihoods"], num_pixels)
        mse_loss = F.mse_loss(output["x_hat"], target)

        # Training objective (large scale, do not put in the table)
        loss = self.lmbda * 255 ** 2 * mse_loss + bpp_loss

        # Comparable reporting metric (same formula as every other row)
        report_loss = comparable_loss(
            output["x_hat"], target, output["likelihoods"], self.report_lmbda
        )

        return {
            "loss": loss,
            "mse_loss": mse_loss,
            "bpp_loss": bpp_loss,
            "report_loss": report_loss,
        }


# ============================================================
# 2) MS-SSIM-Based Rate-Distortion Loss (TRAINING)
# ============================================================
class MSSSIMRateDistortionLoss(nn.Module):
    """
    Perceptual Rate-Distortion Loss.

        L = lmbda * (1 - MS-SSIM) + BPP

    Use this to TRAIN when targeting MS-SSIM. Here `loss` already
    matches the table scale, so report_loss == loss.
    """

    def __init__(self, lmbda=0.1):
        super().__init__()
        self.lmbda = lmbda

    def forward(self, output, target):
        N, C, H, W = target.size()
        num_pixels = N * H * W

        bpp_loss = compute_bpp(output["likelihoods"], num_pixels)

        msssim_val = ms_ssim(
            output["x_hat"].clamp(0, 1), target, data_range=1.0, size_average=True
        )
        msssim_loss = 1 - msssim_val

        loss = self.lmbda * msssim_loss + bpp_loss

        return {
            "loss": loss,
            "bpp_loss": bpp_loss,
            "msssim_loss": msssim_loss,
            "report_loss": loss,  # already comparable
        }
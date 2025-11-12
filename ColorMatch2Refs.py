# /ComfyUI/custom_nodes/Comfyui-ColorMatchNodes/ColorMatch2Refs.py

import os
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ColorMatch2Refs:
    """
    Color-match a target image to TWO references, then blend the two matched results.
    - For each target frame: matched_A = transfer(target, ref_A, method)
                              matched_B = transfer(target, ref_B, method)
      Output = target + strength * ( blend(matched_A, matched_B, weight_a) - target )
    Where blend = weight_a * matched_A + (1 - weight_a) * matched_B
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref_a": ("IMAGE",),
                "image_ref_b": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
                    [
                        "mkl",
                        "hm",
                        "reinhard",
                        "mvgd",
                        "hm-mvgd-hm",
                        "hm-mkl-hm",
                    ],
                    {"default": "mkl"},
                ),
                # Weight for reference A. Reference B gets (1 - weight_a).
                "weight_a": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                # Like your original node: allow a stronger/softer pull toward the blended result.
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "Elyetis/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch_blend"

    DESCRIPTION = """"""

    def colormatch_blend(self, image_ref_a, image_ref_b, image_target, method, weight_a, strength=1.0, multithread=True):
        try:
            from color_matcher import ColorMatcher
        except Exception:
            raise Exception("Can't import color-matcher. Install it first: pip install color-matcher")

        # Ensure CPU tensors for numpy interop
        image_ref_a = image_ref_a.cpu()
        image_ref_b = image_ref_b.cpu()
        image_target = image_target.cpu()

        batch_size = image_target.size(0)

        # Squeeze potential singleton batch dims for numpy output, but keep indexing logic
        refs_a = image_ref_a.squeeze()
        refs_b = image_ref_b.squeeze()
        targs  = image_target.squeeze()

        ref_a_np = refs_a.numpy()
        ref_b_np = refs_b.numpy()
        targ_np  = targs.numpy()

        # Keep weight_b as 1 - weight_a
        weight_b = 1.0 - float(weight_a)

        def get_frame(arr, i, total_b, arr_b):
            """
            Helper to pick the right frame if 'arr' is batched (batch==total_b) or single (batch==1).
            - If arr batch == 1, reuse the single reference for all i.
            - Else, return arr[i].
            """
            if arr_b == 1:
                return arr
            # arr is expected shape [B, H, W, C] once squeezed might be [H, W, C] if B==1
            # Here we assume original (B, H, W, C), so when batch>1, no squeeze removed B.
            return arr[i]

        # Resolve batch sizes for refs
        ref_a_batch = image_ref_a.size(0)
        ref_b_batch = image_ref_b.size(0)

        def process(i):
            cm = ColorMatcher()

            targ_i = targ_np if batch_size == 1 else image_target[i].cpu().numpy()
            ref_a_i = ref_a_np if ref_a_batch == 1 else image_ref_a[i].cpu().numpy()
            ref_b_i = ref_b_np if ref_b_batch == 1 else image_ref_b[i].cpu().numpy()

            try:
                matched_a = cm.transfer(src=targ_i, ref=ref_a_i, method=method)
            except Exception as e:
                print(f"[ColorMatch2Refs] Thread {i} ref A failed: {e}")
                matched_a = targ_i  # fallback: no change

            try:
                matched_b = cm.transfer(src=targ_i, ref=ref_b_i, method=method)
            except Exception as e:
                print(f"[ColorMatch2Refs] Thread {i} ref B failed: {e}")
                matched_b = targ_i  # fallback: no change

            # Blend the two matched results
            blended = weight_a * matched_a + weight_b * matched_b

            # Apply 'strength' pull toward the blended result
            out = targ_i + strength * (blended - targ_i)

            return torch.from_numpy(out)

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                outs = list(executor.map(process, range(batch_size)))
        else:
            outs = [process(i) for i in range(batch_size)]

        out = torch.stack(outs, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)

class ColorMatchBlendAutoWeights:
    """
    Color-match a *batch* of target images to TWO references, with weight_a
    automatically changing per frame across the batch.

    Default ramp (batch size N):
        i = 0 .......... N-1
        weight_a(i) = 1 - i/(N-1)   (=> 1.00 ... 0.00)

    You can customize the ramp via start_weight_a/end_weight_a and easing.

    Output per frame:
        matched_a = transfer(target_i, ref_A, method)
        matched_b = transfer(target_i, ref_B, method)
        blended   = weight_a * matched_a + (1 - weight_a) * matched_b
        out_i     = target_i + strength * (blended - target_i)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref_a": ("IMAGE",),
                "image_ref_b": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
                    [
                        "mkl",
                        "hm",
                        "reinhard",
                        "mvgd",
                        "hm-mvgd-hm",
                        "hm-mkl-hm",
                    ],
                    {"default": "mkl"},
                ),
            },
            "optional": {
                # Pull strength toward the blended result
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "multithread": ("BOOLEAN", {"default": True}),

                # Auto-weight controls (defaults reproduce your examples):
                # - With N=3 -> [1.0, 0.5, 0.0]
                # - With N=10 -> [1.0, 0.9, 0.8, ..., 0.0]
                "start_weight_a": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_weight_a":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "easing": (["linear", "ease_in", "ease_out", "ease_in_out", "smoothstep"], {"default": "linear"}),
                # For ease_in/ease_out/ease_in_out, controls curve "power"
                "ease_power": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                # Debug print per-frame weights in the console
                "debug_print": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Elyetis/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch_blend_autoweights"

    DESCRIPTION = """
Auto-weighted two-reference color-match for batches.
- First frame leans to Ref A, last frame to Ref B (by default 1→0 linearly).
- Optional easing and start/end weights for custom ramps.
Requires: pip install color-matcher
"""

    @staticmethod
    def _ease(t, mode, p):
        # t in [0,1]
        if mode == "linear":
            return t
        if mode == "smoothstep":
            # classic smoothstep
            return t * t * (3 - 2 * t)
        # power-based easings
        if mode == "ease_in":
            return t ** p
        if mode == "ease_out":
            return 1.0 - (1.0 - t) ** p
        if mode == "ease_in_out":
            # symmetric ease with power p
            if t < 0.5:
                return 0.5 * (2 * t) ** p
            else:
                return 1.0 - 0.5 * (2 * (1.0 - t)) ** p
        return t

    def colormatch_blend_autoweights(
        self,
        image_ref_a,
        image_ref_b,
        image_target,
        method,
        strength=1.0,
        multithread=True,
        start_weight_a=1.0,
        end_weight_a=0.0,
        easing="linear",
        ease_power=2.0,
        debug_print=False,
    ):
        try:
            from color_matcher import ColorMatcher
        except Exception:
            raise Exception("Can't import color-matcher. Install it first: pip install color-matcher")

        # Ensure CPU tensors for numpy interop
        image_ref_a = image_ref_a.cpu()
        image_ref_b = image_ref_b.cpu()
        image_target = image_target.cpu()

        batch_size = int(image_target.size(0))
        if batch_size < 1:
            raise ValueError("image_target batch is empty.")

        # If a reference batch is 1, reuse it for all targets; if it equals target batch, match per-frame.
        ref_a_batch = int(image_ref_a.size(0))
        ref_b_batch = int(image_ref_b.size(0))

        # Precompute per-frame weights
        weights_a = []
        if batch_size == 1:
            # degenerate case: just use start_weight_a
            wa = float(start_weight_a)
            weights_a.append(max(0.0, min(1.0, wa)))
        else:
            for i in range(batch_size):
                t = i / (batch_size - 1)                    # 0 → 1 across the batch
                t = self._ease(t, easing, float(ease_power)) # apply easing
                wa = (1.0 - t) * float(start_weight_a) + t * float(end_weight_a)  # lerp
                wa = max(0.0, min(1.0, wa))
                weights_a.append(wa)

        if debug_print:
            print(f"[ColorMatchBlendAutoWeights] weights_a (len={batch_size}): {weights_a}")

        def process(i):
            cm = ColorMatcher()

            targ_i = image_target[i].cpu().numpy()
            # Select appropriate ref frame (broadcast if batch==1)
            ref_a_i = image_ref_a[0 if ref_a_batch == 1 else i].cpu().numpy()
            ref_b_i = image_ref_b[0 if ref_b_batch == 1 else i].cpu().numpy()

            try:
                matched_a = cm.transfer(src=targ_i, ref=ref_a_i, method=method)
            except Exception as e:
                print(f"[ColorMatchBlendAutoWeights] Thread {i} ref A failed: {e}")
                matched_a = targ_i  # fallback: no change

            try:
                matched_b = cm.transfer(src=targ_i, ref=ref_b_i, method=method)
            except Exception as e:
                print(f"[ColorMatchBlendAutoWeights] Thread {i} ref B failed: {e}")
                matched_b = targ_i  # fallback: no change

            wa = weights_a[i]
            wb = 1.0 - wa
            blended = wa * matched_a + wb * matched_b
            out = targ_i + float(strength) * (blended - targ_i)
            return torch.from_numpy(out)

        if multithread and batch_size > 1:
            max_threads = min(os.cpu_count() or 1, batch_size)
            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                outs = list(executor.map(process, range(batch_size)))
        else:
            outs = [process(i) for i in range(batch_size)]

        out = torch.stack(outs, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)

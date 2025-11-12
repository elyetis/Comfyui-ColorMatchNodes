# /ComfyUI/custom_nodes/Comfyui-ColorMatchNodes/__init__.py

# Import both of your node classes
from .ColorMatch2Refs import ColorMatch2Refs
from .ColorMatch2Refs import ColorMatchBlendAutoWeights

NODE_CLASS_MAPPINGS = {
    # Color Match 2refs
    "ColorMatch2Refs": ColorMatch2Refs,
    "ColorMatchBlendAutoWeights": ColorMatchBlendAutoWeights,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorMatch2Refs": "Color Match 2Refs",
    "ColorMatchBlendAutoWeights": "Color Match 2Refs Blend AutoWeights",
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("✅ Loaded Elyetis's Comfyui-ColorMatchNodes")
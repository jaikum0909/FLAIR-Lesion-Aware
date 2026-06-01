from PIL import Image
import numpy as np

try:
    from flair import FLAIRModel
except ImportError:
    from local_flair.modeling import FLAIRModel

# This auto-downloads pretrained weights from HuggingFace (~few hundred MB)
model = FLAIRModel.from_pretrained("jusiro2/FLAIR")

print("FLAIR loaded successfully!")
print(model)
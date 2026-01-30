import cv2
import numpy as np
import torch
from scipy import ndimage

def smart_preprocess(user_image):
    # 1. Convert to grayscale if needed
    if len(user_image.shape) == 3:
        user_image = cv2.cvtColor(user_image, cv2.COLOR_RGB2GRAY)

    # 2. Invert colors? 
    # Gradio canvas is usually Black ink on White background.
    # MNIST is White ink on Black background.
    # We check the corner pixel. If it's bright (255), we invert.
    if user_image[0, 0] > 128:
        user_image = 255 - user_image

    # 3. Find the Bounding Box (Crop empty space)
    coords = cv2.findNonZero(user_image)
    if coords is None:
        return None # Empty drawing
    x, y, w, h = cv2.boundingRect(coords)
    digit = user_image[y:y+h, x:x+w]

    # 4. Resize to fit in 20x20 box (keeping aspect ratio)
    # MNIST digits are roughly 20px tall/wide inside the 28px box
    max_dim = max(w, h)
    scale = 20.0 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 5. Center in 28x28 canvas
    final_img = np.zeros((28, 28), dtype=np.uint8)
    
    # Calculate padding to center it
    pad_x = (28 - new_w) // 2
    pad_y = (28 - new_h) // 2
    
    final_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = digit_resized

    # 6. Normalize to [-1, 1] for the model
    # Convert to float 0..1
    img_tensor = final_img.astype(np.float32) / 255.0
    # Scale to -1..1
    img_tensor = (img_tensor * 2) - 1
    
    return torch.tensor(img_tensor).unsqueeze(0) # (1, 28, 28)
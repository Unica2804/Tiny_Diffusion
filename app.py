import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import ndimage
import os
import sys

# --- IMPORT YOUR MODULES ---
# We reuse the files you already uploaded!
from src.network import tiny3Dunet
# from src.user_img_preprocessor import smart_preprocess
from src.inference import smooth_ink_video, generate_writing_sample

# --- 1. CONFIGURATION ---
MODEL_PATH = "model/mnist_motion_3d_v3.pth" # Ensure this path is correct
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. LOAD MODEL ---
print(f"Loading model on {device}...")
model = tiny3Dunet().to(device)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully!")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}")

model.eval()

# # --- 3. DIFFUSION SCHEDULES ---
# T = 1000
# beta = torch.linspace(1e-4, 0.02, T).to(device)
# alpha = 1.0 - beta
# alpha_bar = torch.cumprod(alpha, dim=0)

# # --- 4. INFERENCE LOGIC ---
# @torch.no_grad()
# def animate_digit(user_image):
#     if user_image is None: return None

#     if isinstance(user_image, dict):
#         user_image = user_image["composite"]

#     # 1. Use your existing Preprocessor
#     # This centers and resizes the drawing to look like MNIST
#     ref_tensor = smart_preprocess(user_image)
#     if ref_tensor is None: return None
#     ref_tensor = ref_tensor.to(device)

#     # 2. Setup Conditions
#     # Create Mask: 1 where ink is, 0 where empty
#     ink_mask = (ref_tensor > -0.9).float().unsqueeze(0).unsqueeze(2).repeat(1, 1, 16, 1, 1)
#     condition = ref_tensor.unsqueeze(0).unsqueeze(2).repeat(1, 1, 16, 1, 1)

#     # 3. Initialize Noise
#     x = torch.randn((1, 1, 16, 28, 28)).to(device)
    
#     # Pre-move constants for speed
#     alpha_gpu = alpha.to(device)
#     alpha_bar_gpu = alpha_bar.to(device)
#     beta_gpu = beta.to(device)

#     # 4. Fast Diffusion Loop (Stride 2)
#     steps = list(reversed(range(0, 1000, 2)))
    
#     for i in steps:
#         t = torch.tensor([i]).to(device)
        
#         # Predict
#         model_input = torch.cat([x, condition], dim=1)
#         eps_theta = model(model_input, t)
        
#         # Denoise
#         a_t = alpha_gpu[i].view(1,1,1,1,1)
#         ab_t = alpha_bar_gpu[i].view(1,1,1,1,1)
#         b_t = beta_gpu[i].view(1,1,1,1,1)
        
#         mean = (1 / torch.sqrt(a_t)) * (x - (b_t / torch.sqrt(1 - ab_t)) * eps_theta)
        
#         if i > 0:
#             z = torch.randn_like(x)
#             x_new = mean + torch.sqrt(b_t) * z
#         else:
#             x_new = mean
        
#         # Apply Mask (The trick that cleans up user drawings)
#         x = x_new * ink_mask + (1 - ink_mask) * -1.0

#     # 5. Post-Process (Smooth & Save)
#     vid_np = x.clamp(-1, 1).cpu().squeeze().numpy()
    
#     # Interpolate for smoothness
#     vid_smooth = ndimage.zoom(vid_np, (4, 1, 1), order=1)
#     vid_smooth = vid_smooth[::1] # Reverse: Empty -> Full
    
#     output_path = "output_animation.gif"
#     fig = plt.figure(figsize=(3, 3))
#     plt.axis('off')
#     ims = []
    
#     for frame in vid_smooth:
#         frame_uint8 = ((frame + 1) / 2 * 255).astype(np.uint8)
#         im = plt.imshow(frame_uint8, cmap='gray', animated=True)
#         ims.append([im])
    
#     ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True)
#     ani.save(output_path, writer='pillow')
#     plt.close()
    
#     return output_path
def animate_digit(inputs,model=model):
    generated_video, ref_image = generate_writing_sample(model, inputs)
    path = smooth_ink_video(generated_video, ref_image)
    return path

# --- 5. LAUNCH APP ---
with gr.Blocks(title="AI Writer") as demo:
    gr.Markdown("# 🖊️ 3D Diffusion Writer")
    with gr.Row():
        # UPDATED: Use Sketchpad for drawing
        inp = gr.Sketchpad(label="Draw Here", type="numpy")
        out = gr.Image(label="AI Animation")
    btn = gr.Button("Animate!", variant="primary")
    btn.click(fn=animate_digit, inputs=inp, outputs=out)

if __name__ == "__main__":
    demo.launch(share=False)
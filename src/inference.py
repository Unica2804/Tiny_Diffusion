import torch
from tqdm import tqdm
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from src.Data_preprocessing import DirectionalMNISTDataset
from src.network import tiny3Dunet
from src.user_img_preprocessor import smart_preprocess

T=1000
beta=torch.linspace(1e-4,0.02,T)
alpha=1-beta
alpha_bar=torch.cumprod(alpha,dim=0)
device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def generate_writing_sample(model, image, digit_index=None):
    # 1. Get a Test Image
    # test_ds = DirectionalMNISTDataset(train=False)
    
    # if digit_index is None:
    #     for i in range(len(test_ds)):
    #         _, label = test_ds[i]
    #         if label == 7:
    #             digit_index = i
    #             break
    if image is None: return None

    if isinstance(image, dict):
        image = image["composite"]
    
    # video_tensor, label = test_ds[digit_index]

    ref_frame= smart_preprocess(image).to(device)
    # ref_frame = video_tensor[:,-1, :, :].to(device)
    
    condition = ref_frame.unsqueeze(1).repeat(1, 16, 1, 1).unsqueeze(0) # (1, 1, 16, 28, 28)
    
    x = torch.randn((1, 1, 16, 28, 28)).to(device)
    
    alpha_gpu = alpha.to(device)
    alpha_bar_gpu = alpha_bar.to(device)
    beta_gpu = beta.to(device)

    # print(f"Generating animation for Digit {label}...")
    
    for i in tqdm(reversed(range(1000)), total=1000, desc="Writing"):
        t = torch.tensor([i]).to(device)
        
        model_input = torch.cat([x, condition], dim=1)
        eps_theta = model(model_input, t)
        
        a_t = alpha_gpu[i].view(1,1,1,1,1)
        ab_t = alpha_bar_gpu[i].view(1,1,1,1,1)
        b_t = beta_gpu[i].view(1,1,1,1,1)
        
        mean = (1 / torch.sqrt(a_t)) * (x - (b_t / torch.sqrt(1 - ab_t)) * eps_theta)
        
        if i > 0:
            z = torch.randn_like(x)
            x = mean + torch.sqrt(b_t) * z
        else:
            x = mean

    output = x.clamp(-1, 1).cpu().squeeze().numpy()
    return output, ref_frame.cpu().squeeze().numpy()

def smooth_ink_video(generated_video, ref_image, filename="hq_smooth_writing.gif"):

    smooth_video = scipy.ndimage.zoom(generated_video, (4, 1, 1), order=1)
    
    frames = smooth_video[::1]
    
    frames_uint8 = ((frames + 1) / 2 * 255).astype(np.float32)
    ref_uint8 = ((ref_image + 1) / 2 * 255).astype(np.uint8)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.set_title("Input")
    ax2.set_title("AI Writing (Smoothed)")
    ax1.axis('off')
    ax2.axis('off')
    
    ims = []
    for frame in frames_uint8:
        
        blurred = scipy.ndimage.gaussian_filter(frame, sigma=0.5)
        
        clean_frame = np.clip((blurred - 50) * 1.5, 0, 255).astype(np.uint8)
        
        im1 = ax1.imshow(ref_uint8, cmap='gray', animated=True)
        im2 = ax2.imshow(clean_frame, cmap='gray', animated=True)
        ims.append([im1, im2])

    ani = animation.ArtistAnimation(fig, ims, interval=150, blit=True)
    ani.save(filename, writer='pillow')
    plt.close()
    print(f"Saved smoothed video: {filename}")
    return filename


# model = tiny3Dunet().to(device)
# model.load_state_dict(torch.load("model/mnist_motion_3d_v3.pth", map_location=device))
# model.eval()
# print("Model loaded successfully.")
# generated_video, ref_image = generate_writing_sample(model)
# smooth_ink_video(generated_video, ref_image)

import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from .network import tiny3Dunet
from .Data_preprocessing import DirectionalMNISTDataset

torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"

T=1000
beta=torch.linspace(1e-4,0.02,T)
alpha=1-beta
alpha_bar=torch.cumprod(alpha,dim=0)
def q_sample(x0,t,noise=None):
    if noise is None:
        noise=torch.randn_like(x0)
    device=x0.device
    alpha_bar=torch.cumprod(alpha,dim=0).to(device)
    alpha_bart=alpha_bar[t][(...,)+ (None,)*4]
    return torch.sqrt(alpha_bart)*x0+torch.sqrt(1-alpha_bart)*noise

def p_loss_3d(denoise_model, x_start, t, noise=None):
    if noise is None:
        noise=torch.randn_like(x_start)

    # Take the first frame as a condition
    condition = x_start[:,:,-1,:,:].unsqueeze(2).repeat(1,1,16,1,1)
    # Add noise to the video (Forward)
    x_noisy= q_sample(x_start,t,noise)
    # Concatenate noise and condition to feed it as input
    model_input = torch.cat([x_noisy,condition], dim=1)
    # predicted noise
    predicted_noise = denoise_model(model_input,t)
    # MSE Loss
    return torch.nn.functional.mse_loss(predicted_noise,noise)

lr = 2e-4
epochs = 3
batch_size = 64 
model = tiny3Dunet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
criterion = p_loss_3d # Your custom 3D loss
ds = DirectionalMNISTDataset(train=True)
train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print("Starting 3D Training...")
losses = []

for epoch in range(epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch_video, _ in pbar:
        batch_video = batch_video.to(device)
        t = torch.randint(0, 1000, (batch_video.shape[0],), device=device).long()
        loss = criterion(model, batch_video, t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pbar.set_postfix(MSE=loss.item())
        
torch.save(model.state_dict(), "mnist_motion_3d_.pth")
print("Training Complete! Model Saved.")
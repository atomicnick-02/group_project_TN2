import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def genBeta(timesteps: int = 300, T: int = 1000):
	# generate beta schedule in (0,1], linearly from 0.0003 to 0.03
	start_point = 0.0003
	end_point = 0.03
	beta_array = torch.linspace(start_point, end_point, timesteps)
	return beta_array


# SECTION: define the diffusion hyperparams
timesteps = 100
batch_size = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

beta_array = genBeta(timesteps=timesteps).to(device)
alpha_array = torch.cumprod(1-beta_array, dim=0)
sqrt_one_minus_alpha = torch.sqrt(1 - alpha_array)

noise = torch.randn((batch_size, 2), device=device)
diffusion_dt = torch.randint(0,timesteps,(batch_size,), device = device).long()


# SECTION: define the dataset
class ToyDataset(Dataset):
	"""Very simple 2D point dataset"""
	def __init__(self, num_samples=1000):
		self.data = self.generate_toy_data(num_samples)
	
	def generate_toy_data(self, num_samples):
		"""Generate simple 2D points - circular shape"""
		# Generate points on a circle
		angles = torch.rand(num_samples) * 2 * np.pi
		radius = 1.0 + torch.randn(num_samples) * 0.05  # slight noise
		
		x = radius * torch.cos(angles)
		y = radius * torch.sin(angles)
		
		data = torch.stack([x, y], dim=1).to(torch.float32)
		return data
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]


dataset = ToyDataset(num_samples=5000)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# SECTION: the diffusion process
def forward_diffusion(x_0, t):
	"""
	Forward diffusion process: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
	"""
	epsilon = torch.randn_like(x_0)  # noise ~ N(0, I)
	
	z_t = torch.sqrt(alpha_array[t]) * x_0 + torch.sqrt( 1 - alpha_array[t]) * epsilon
	return z_t

# SECTION: visualize the forward diffusion process
def visualize_alpha_beta_schedules(beta_array, alpha_array):
	timesteps = len(beta_array)
	plt.figure(figsize=(12, 5))
	
	plt.subplot(1, 2, 1)
	plt.plot(range(timesteps), beta_array.cpu().numpy(), label='Beta Schedule')
	plt.title('Beta Schedule')
	plt.xlabel('Timestep')
	plt.ylabel('Beta Value')
	plt.grid()
	plt.legend()
	
	plt.subplot(1, 2, 2)
	plt.plot(range(timesteps), alpha_array.cpu().numpy(), label='Alpha Schedule', color='orange')
	plt.title('Alpha Schedule')
	plt.xlabel('Timestep')
	plt.ylabel('Alpha Value')
	plt.grid()
	plt.legend()

	
	plt.tight_layout()
	plt.savefig('/workspaces/group_project_TN2/diffusion_models/alpha_beta_schedules.png')
	print("Alpha and Beta schedules plot saved to alpha_beta_schedules.png")
	
def visualize_forward_diffusion(dataset, timesteps):
	plots_to_show = 10
	step = timesteps // plots_to_show
	timesteps_to_plot = [t * step for t in range(plots_to_show)]

	fig, axes = plt.subplots(1, plots_to_show, figsize=(plots_to_show * 3, 3))  # ← only plots_to_show axes

	for i, t in enumerate(timesteps_to_plot):
		x_0 = dataset.data[:batch_size].to(device)
		x_t = forward_diffusion(x_0, t).cpu().numpy()

		axes[i].scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=10)
		axes[i].set_title(f"t={t}")
		axes[i].set_xlim(-3, 3)
		axes[i].set_ylim(-3, 3)
		axes[i].set_aspect('equal')
		axes[i].grid()

	plt.tight_layout()
	plt.savefig('/workspaces/group_project_TN2/diffusion_models/forward_diffusion_visualization.png')


class MLP(nn.Module):
	def __init__(self, input_dim = 2, hidden_dim = 128, time_dim = 24):
		super().__init__()

		self.time_embed = nn.Sequential(
			nn.Linear(1,time_dim),
			nn.SiLU(),
			nn.Linear(time_dim, time_dim)
		)

		self.net = nn.Sequential(
			nn.Linear(input_dim + time_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, hidden_dim),
			nn.SiLU(),
			nn.Linear(hidden_dim, input_dim)
		)
	def forward(self, x, t):
		t_embed = self.time_embed(t.unsqueeze(-1))
		combined = self.net(torch.concat([x, t_embed], dim= -1))
		return combined


class Scheduler:
	def __init__(self, num_steps):
		self.num_steps = num_steps

		# Linear schedule from 0.0003 to 0.03
		self.beta_array = torch.linspace(0.0003, 0.03, num_steps).to(device)
		self.alphas = 1 - self.beta_array
		self.alpha_array = torch.cumprod(self.alphas, dim=0) # This is alpha_cumprod
		self.sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha_array)

	def add_noise(self, x_0, t):
		epsilon = torch.randn_like(x_0)  # noise ~ N(0, I)
		sqrt_alpha_bar = torch.sqrt(self.alpha_array[t]).view(-1, 1)
		sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha[t].view(-1, 1)

		z_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * epsilon
		return z_t, epsilon

	
myNet = MLP(input_dim=2, hidden_dim=128, time_dim=24).to(device)
scheduler = Scheduler(num_steps=timesteps)
loss_fn = nn.MSELoss()
optim = torch.optim.Adam(myNet.parameters(), lr=1e-3)
loss_hist = []

def train(dataloader, model, loss_fn, optimizer):
	model.train()

	size = len(dataloader.dataset)
	for batch, x_0 in enumerate(dataloader):
		x_0 = x_0.to(device)
		t = torch.randint(1, timesteps, (x_0.size(0),), device=device)
		z_t, epsilon = scheduler.add_noise(x_0, t)

		t_normalized = t.float() / timesteps
		epsilon_pred = model(z_t, t_normalized)

		loss = loss_fn(epsilon_pred, epsilon)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		loss_hist.append(loss.item())
		if batch % 10 == 0:
			loss, current = loss.item(), batch * len(x_0)
			# print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
	
epochs = 100

for epoch in range(epochs):
	# print(f"Epoch {epoch+1}\n-------------------------------")
	train(dataloader, myNet, loss_fn, optim)



def sample(model, num_samples, timesteps):
	model.eval()
	intermediates = []
	save_at = [timesteps-1, timesteps//2, timesteps//5, timesteps//10, 1]
	with torch.no_grad():
		x_t = torch.randn((num_samples, 2), device=device)
		for t in reversed(range(1, timesteps)):
			t_normalized = torch.full((num_samples,), t / timesteps, device=device, dtype=torch.float32)
			beta_t = scheduler.beta_array[t]
			alpha_t = scheduler.alpha_array[t]
			first = 1/torch.sqrt(1-beta_t) * x_t
			second = beta_t / (torch.sqrt(1-beta_t) * torch.sqrt(1-alpha_t)) * model(x_t, t_normalized)
			x_t = first - second
			epsilon = torch.randn_like(x_t) if t > 1 else 0
			x_t += torch.sqrt(beta_t) * epsilon
			if t in save_at:
				intermediates.append((t, x_t.cpu().numpy()))
		

	return x_t.cpu().numpy(), intermediates

# visualize sampling results
num_samples = 1000
samples, intermediates = sample(myNet, num_samples, timesteps)
original_data = dataset.data[:num_samples].cpu().numpy()

plt.figure(figsize=(20, 5))
for i, (t_val, sample_val) in enumerate(intermediates):
	plt.subplot(1, len(intermediates), i+1)
	plt.scatter(original_data[:, 0], original_data[:, 1], alpha=0.1, s=5, label='Original Data', color='gray')
	plt.scatter(sample_val[:, 0], sample_val[:, 1], alpha=0.5, s=5, label='Samples', color='blue')
	plt.title(f'Step {t_val}')
	plt.xlim(-3, 3)
	plt.ylim(-3, 3)
	plt.legend()
	plt.grid()
plt.tight_layout()

# plot the loss history
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, label='Training Loss')
plt.title('Training Loss History')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.show()
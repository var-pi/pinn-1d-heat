import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")

a = 0.003
n = 10 # Number of periods
x = np.linspace(0, 1, 500)
t = np.linspace(0, 1, 500)
X, T = np.meshgrid(x, t)

# 1. Fix seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Convert meshgrid to torch tensors
XT = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

def sol1(x,t,f=1/n,amp=1):
	return amp*np.exp(-a*4*(1/f)**2*np.pi**2*t)*np.sin(2*(1/f)*np.pi*x)

def sol(x,t):
	return sol1(x,t) #+sol1(x,t,3/n,2)

def ic(x):
	return sol(x, 0)

def bc_at_0(t):
	return sol(0, t)

def bc_at_1(t):
	return sol(1, t)

U = sol(X, T)

# Flatten meshgrid
X_flat = X.flatten()[:, None]
T_flat = T.flatten()[:, None]
U_flat = sol(X_flat, T_flat)

n_obs = 10
indices = np.random.choice(len(X_flat), size=n_obs, replace=False)

X_obs = X_flat[indices]
T_obs = T_flat[indices]

# Convert to torch tensors
XT_obs = torch.tensor(np.hstack([X_obs, T_obs]), dtype=torch.float32).to(device)
U_obs = torch.tensor(U_flat[indices], dtype=torch.float32).to(device)



n_samples = 1000
indices = np.random.choice(len(X_flat), size=n_samples, replace=False)

X_syn = X_flat[indices]
T_syn = T_flat[indices]

# Convert to torch tensors
XT_syn = torch.tensor(np.hstack([X_syn, T_syn]), dtype=torch.float32).to(device)

# Define a simple MLP
class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        return self.net(x)

# Fourier feature mapping
class FourierFeatures(nn.Module):
    def __init__(self, in_features, num_features=64, scale=2*n):
        super().__init__()
        self.B = nn.Parameter(scale * torch.randn((in_features, num_features)), requires_grad=False)
    
    def forward(self, x):
        # x: (N, in_features)
        x_proj = 2 * np.pi * x @ self.B  # (N, num_features)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (N, 2*num_features)

class MLP_Fourier(nn.Module):
    def __init__(self, in_features=2, hidden=64, out_features=1, fourier_features=64):
        super().__init__()
        self.ff = FourierFeatures(in_features-1, num_features=fourier_features)
        self.net = nn.Sequential(
            nn.Linear(fourier_features*2+1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_features)
        )
        
    def forward(self, x):
        x_spatial = x[:, 0:1]
        t_temporal = x[:, 1:2]
        x_feat = self.ff(x_spatial)
        return self.net(torch.cat([x_feat, t_temporal], dim=-1))	

# Instantiate MLP and evaluate
# mlp = MLP_Fourier(fourier_features=32).to(device)
mlp = MLP(hidden=64).to(device)

optimizer = optim.Adam(mlp.parameters())
loss_fn = nn.MSELoss()

# Enable gradients for input
XT_obs.requires_grad_(True)
XT_syn.requires_grad_(True)

def loss_phys(nn):
    u = nn(XT_syn)
    u_t = torch.autograd.grad(u, XT_syn, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
    u_x = torch.autograd.grad(u, XT_syn, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
    u_xx = torch.autograd.grad(u_x, XT_syn, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
    
    f = u_t - a * u_xx
    return torch.mean(f**2)

n_bc = 100
T_bc_left = np.random.choice(t, size=n_bc, replace=False)[:, None]  # shape (10,1)
X_bc_left = np.zeros_like(T_bc_left)
T_bc_right = np.random.choice(t, size=n_bc, replace=False)[:, None]  # shape (10,1)
X_bc_right = np.ones_like(T_bc_right)

# Left boundary x=0
XT_bc_left = torch.tensor(np.hstack([X_bc_left, T_bc_left]), dtype=torch.float32, requires_grad=True).to(device)
# Right boundary x=1
XT_bc_right = torch.tensor(np.hstack([X_bc_right, T_bc_right]), dtype=torch.float32, requires_grad=True).to(device)

n_ic = 100
X_ic = np.random.choice(x, size=n_ic, replace=False)[:, None]  # random x points
T_ic = np.zeros_like(X_ic)  # t=0

# Convert to torch tensor
XT_ic = torch.tensor(np.hstack([X_ic, T_ic]), dtype=torch.float32, requires_grad=True).to(device)

def loss_cond(nn):
	# Evaluate MLP at boundary points
	u_bc_left = nn(XT_bc_left)
	u_bc_left_target = torch.tensor(bc_at_0(T_bc_left), dtype=torch.float32, device=device)
	loss_bc_left = torch.mean((u_bc_left - u_bc_left_target)**2)

	u_bc_right = nn(XT_bc_right)
	u_bc_right_target = torch.tensor(bc_at_1(T_bc_right), dtype=torch.float32, device=device)
	loss_bc_right = torch.mean((u_bc_right - u_bc_right_target)**2)

	# Evaluate MLP at IC points
	u_ic = mlp(XT_ic)

	u_ic_target = torch.tensor(ic(X_ic), dtype=torch.float32, device=device)

	# IC loss
	loss_ic = torch.mean((u_ic - u_ic_target)**2)

	return loss_bc_left + loss_bc_right + loss_ic


n_epochs = 50000
for epoch in range(n_epochs):
    optimizer.zero_grad()

    # Data loss
    u = mlp(XT_obs)
    loss_data = loss_fn(u, U_obs) if n_obs > 0 else 0
    
    # Total loss
    loss = loss_data + loss_phys(mlp) + loss_cond(mlp)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

with torch.no_grad():
    M = mlp(XT).cpu().numpy().reshape(X.shape)



# --- Visualisation ---



# Create figure with three plots: exact, approximate, and difference
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Exact solution
pcm0 = axes[0].pcolormesh(X, T, U, shading='auto')
axes[0].set_title("Exact Solution $u_{exact}(x,t)$")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(pcm0, ax=axes[0], label="u_exact(x,t)")

# Approximate solution
pcm1 = axes[1].pcolormesh(X, T, M, shading='auto')
axes[1].set_title("Approximate Solution $u_{aprx}(x,t)$")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(pcm1, ax=axes[1], label="u_aprx(x,t)")

# Difference
pcm2 = axes[2].pcolormesh(X, T, U-M, shading='auto')
axes[2].set_title("Difference $u_{aprx} - u_{exact}$")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
axes[2].scatter(X_obs, T_obs, color='red', marker='x', s=50, label='Observations')
axes[2].scatter(X_syn, T_syn, color='green', marker='o', s=5, label='Synthetical')
axes[2].scatter(np.vstack([X_bc_left, X_bc_right]), np.vstack([T_bc_left, T_bc_right]), color='orange', marker='o', s=5, label='Boundary Conditions')
axes[2].scatter(X_ic, T_ic, color='yellow', marker='o', s=5, label='Initial Conditions')
axes[2].legend()
fig.colorbar(pcm2, ax=axes[2], label="Difference")

plt.tight_layout()
plt.show()

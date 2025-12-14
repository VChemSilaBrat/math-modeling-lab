import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Гиперпараметры ---
ITERATIONS = 3000   # Уменьшено для более быстрого тестирования
LR = 0.001
N_PHYS = 10000
N_BOUND = 2500
DS = 0.1
N_STEPS = 40
MODEL_NAME = "2_7_7_1_bif"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Оптимизация CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

class PINN(nn.Module):
    def __init__(self, init_lambda=0.0):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 7),
            nn.Tanh(),
            nn.Linear(7, 7),
            nn.Tanh(),
            nn.Linear(7, 1)
        )
        self.lam = nn.Parameter(torch.tensor([init_lambda], dtype=torch.float32))

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

def generate_data(n_physics, n_boundary):
    x_phys = torch.rand((n_physics, 1), device=device, requires_grad=True)
    y_phys = torch.rand((n_physics, 1), device=device, requires_grad=True)
    
    rand_vals = torch.rand((n_boundary, 1), device=device)
    zeros = torch.zeros((n_boundary, 1), device=device)
    ones = torch.ones((n_boundary, 1), device=device)
    
    left   = torch.cat([zeros, rand_vals], dim=1)
    right  = torch.cat([ones, rand_vals], dim=1)
    bottom = torch.cat([rand_vals, zeros], dim=1)
    top    = torch.cat([rand_vals, ones], dim=1)
    
    boundary_points = torch.cat([left, right, bottom, top], dim=0)
    
    return x_phys, y_phys, boundary_points

def compute_pde_residual(model, x, y):
    u = model(x, y)
    lambda_param = model.lam
    
    u_x = torch.autograd.grad(u, x, 
                              grad_outputs=torch.ones_like(u), 
                              create_graph=True, 
                              retain_graph=True)[0]
    
    u_y = torch.autograd.grad(u, y, 
                              grad_outputs=torch.ones_like(u), 
                              create_graph=True,
                              retain_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, 
                               grad_outputs=torch.ones_like(u_x), 
                               create_graph=True)[0]
                               
    u_yy = torch.autograd.grad(u_y, y, 
                               grad_outputs=torch.ones_like(u_y), 
                               create_graph=True)[0]
    
    residual = u_xx + u_yy + 1 + u + lambda_param * (u ** 2)
    
    return residual

def get_anchor_points(n=1000):
    x = torch.rand((n, 1), device=device)
    y = torch.rand((n, 1), device=device)
    return x, y

# Предварительно генерируем все данные
x_anchor, y_anchor = get_anchor_points(N_PHYS)
x_p, y_p, xy_b = generate_data(N_PHYS, N_BOUND)

history_lambda = []
history_u_max = []

print(">>> Step 0: Initial solution at lambda=0...")
model = PINN(init_lambda=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model.lam.requires_grad = False

for i in tqdm(range(ITERATIONS)):
    optimizer.zero_grad()
    
    res = compute_pde_residual(model, x_p, y_p)
    loss_phys = torch.mean(res**2)
    
    u_b = model(xy_b[:,0:1], xy_b[:,1:2])
    loss_b = torch.mean(u_b**2)
    
    loss = loss_phys + 10 * loss_b
    loss.backward()
    optimizer.step()

with torch.no_grad():
    u_prev = model(x_anchor, y_anchor).clone()
    lam_prev = model.lam.item()
    max_val = u_prev.max().item()

history_lambda.append(lam_prev)
history_u_max.append(max_val)
print(f"Start Point: Lambda={lam_prev:.4f}, Max|u|={max_val:.4f}")

# --- ШАГ 1: Вторая точка ---
print(">>> Step 1: Second solution at lambda=0.5...")
model.lam.data = torch.tensor([0.5], device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
for i in tqdm(range(ITERATIONS)):
    optimizer.zero_grad()
    res = compute_pde_residual(model, x_p, y_p)
    loss_phys = torch.mean(res**2)
    u_b = model(xy_b[:,0:1], xy_b[:,1:2])
    loss_b = torch.mean(u_b**2)
    loss = loss_phys + 10 * loss_b
    loss.backward()
    optimizer.step()

with torch.no_grad():
    u_curr = model(x_anchor, y_anchor).clone()
    lam_curr = model.lam.item()
    max_val = u_curr.max().item()

history_lambda.append(lam_curr)
history_u_max.append(max_val)
print(f"Second Point: Lambda={lam_curr:.4f}, Max|u|={max_val:.4f}")

# --- ШАГ 2: ЦИКЛ ПО ДЛИНЕ ДУГИ ---
print(f">>> Continuation process for {N_STEPS} steps...")

model.lam.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ИСПРАВЛЕНИЕ: правильные начальные значения для касательной
u_prev_vec = u_prev.clone()  # Первая точка
u_curr_vec = u_curr.clone()  # Вторая точка
lam_prev_val = history_lambda[0]  # Первое значение lambda
lam_curr_val = history_lambda[1]  # Второе значение lambda

for step in range(N_STEPS):
    # 1. Вычисляем касательную (ИСПРАВЛЕНО!)
    diff_u = u_curr_vec - u_prev_vec  # Было u_prev_vec - u_prev_vec (BUG!)
    diff_lam = lam_curr_val - lam_prev_val
    
    # Нормализация
    norm = torch.sqrt(torch.mean(diff_u**2) + diff_lam**2) + 1e-10
    
    tau_u = diff_u / norm
    tau_lam = diff_lam / norm

    # 2. ПРЕДСКАЗАНИЕ
    with torch.no_grad():
        model.lam.data = torch.tensor([lam_curr_val + DS * tau_lam.item()], device=device)

    # 3. КОРРЕКЦИЯ
    pbar = tqdm(range(ITERATIONS), leave=False, desc=f"Step {step}")

    for i in pbar:
        optimizer.zero_grad()
        
        # A. PDE Loss
        res = compute_pde_residual(model, x_p, y_p)
        loss_pde = torch.mean(res**2)
        
        # B. Boundary Loss
        u_b = model(xy_b[:,0:1], xy_b[:,1:2])
        loss_b = torch.mean(u_b**2)
        
        # C. Arclength Loss
        u_anchor_pred = model(x_anchor, y_anchor)
        
        # ИСПРАВЛЕНО: используем u_curr_vec как базовую точку
        dot_u = torch.mean((u_anchor_pred - u_curr_vec) * tau_u)
        dot_lam = (model.lam - lam_curr_val) * tau_lam
        
        loss_arc = ((dot_u + dot_lam) - DS) ** 2
        
        loss = loss_pde + 10 * loss_b + 100 * loss_arc  # Увеличен вес arclength
        
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            pbar.set_description(
                f"S:{step} λ:{model.lam.item():.3f} "
                f"PDE:{loss_pde.item():.2e} Arc:{loss_arc.item():.2e}"
            )

    # 4. Сохранение результатов
    with torch.no_grad():
        u_new_vec = model(x_anchor, y_anchor).clone()
        lam_new_val = model.lam.item()
        
        history_lambda.append(lam_new_val)
        history_u_max.append(u_new_vec.max().item())
        
        # Обновляем для следующего шага
        u_prev_vec = u_curr_vec.clone()
        lam_prev_val = lam_curr_val
        
        u_curr_vec = u_new_vec
        lam_curr_val = lam_new_val
        
    print(f"Step {step+1}: Lambda={lam_new_val:.4f}, Max|u|={u_new_vec.max().item():.4f}")

plt.figure(figsize=(8, 6))
plt.plot(history_lambda, history_u_max, '-o', markersize=4, label='PINN Bifurcation Path')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$max|u|$')
plt.title('Bifurcation Diagram via Pseudo-Arclength Continuation')
plt.grid(True)
plt.legend()
plt.savefig('bifurcation_diagram.png', dpi=150)
print("Диаграмма сохранена как bifurcation_diagram.png")

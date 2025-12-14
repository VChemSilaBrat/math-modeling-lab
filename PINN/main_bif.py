import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import pickle
import copy

# --- Гиперпараметры ---
ITERATIONS = 5000   # Итераций на каждую точку кривой
LR = 0.001
N_PHYS = 10000 # Точек внутри
N_BOUND = 2500 # Точек на одной стороне границы
DS = 0.1            # Длина шага по дуге (Arc Length step)
N_STEPS = 40        # Сколько точек бифуркационной кривой ищем
MODEL_NAME = "2_7_7_1_bif"

# Выбираем устройство (GPU если есть, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

class PINN(nn.Module):
    def __init__(self, init_lambda=0.0):
        super(PINN, self).__init__()
        # Полносвязная сеть:
        # Вход (2) -> Скрытый (32) -> ... -> Выход (1)
        self.net = nn.Sequential(
            nn.Linear(2, 7),
            nn.Tanh(),
            nn.Linear(7, 7),
            nn.Tanh(),
            # nn.Linear(1000, 1000),
            nn.Tanh(),
            nn.Linear(7, 1)
        )
        # веса были выбраны в соотвествии с Table 2 из статьи по PINN
        self.lam = nn.Parameter(torch.tensor([init_lambda], dtype=torch.float32))

    def forward(self, x, y):
        # Объединяем x и y в один тензор размером (N, 2)
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

def generate_data(n_physics, n_boundary):
    """
    n_physics: количество точек внутри области
    n_boundary: количество точек на каждой из 4-х границах
    """
    
    # 1. Точки внутри области [0, 1] x [0, 1]
    # Создаем случайные числа от 0 до 1
    x_phys = torch.rand((n_physics, 1), requires_grad=True).to(device)
    y_phys = torch.rand((n_physics, 1), requires_grad=True).to(device)
    
    # 2. Граничные точки (4 стороны)
    # Левая (x=0), Правая (x=1), Нижняя (y=0), Верхняя (y=1)
    
    # Просто случайные координаты вдоль границ
    rand_vals = torch.rand((n_boundary, 1)) 
    zeros = torch.zeros((n_boundary, 1))
    ones = torch.ones((n_boundary, 1))
    
    # Формируем координаты (x, y) для каждой границы
    # x=0, y=random
    left   = torch.cat([zeros, rand_vals], dim=1)
    # x=1, y=random
    right  = torch.cat([ones, rand_vals], dim=1)
    # x=random, y=0
    bottom = torch.cat([rand_vals, zeros], dim=1)
    # x=random, y=1
    top    = torch.cat([rand_vals, ones], dim=1)
    
    # Объединяем все границы в один массив
    boundary_points = torch.cat([left, right, bottom, top], dim=0).float().to(device)
    
    return x_phys, y_phys, boundary_points

def compute_pde_residual(model, x, y):
    # 1. Предсказание сети u(x, y)
    u = model(x, y)
    lambda_param = model.lam
    # 2. Первые производные du/dx и du/dy
    # grad возвращает кортеж, берем [0]
    u_x = torch.autograd.grad(u, x, 
                              grad_outputs=torch.ones_like(u), 
                              create_graph=True)[0]
    
    u_y = torch.autograd.grad(u, y, 
                              grad_outputs=torch.ones_like(u), 
                              create_graph=True)[0]
    
    # 3. Вторые производные d2u/dx2 и d2u/dy2
    u_xx = torch.autograd.grad(u_x, x, 
                               grad_outputs=torch.ones_like(u_x), 
                               create_graph=True)[0]
                               
    u_yy = torch.autograd.grad(u_y, y, 
                               grad_outputs=torch.ones_like(u_y), 
                               create_graph=True)[0]
    
    # 4. Собираем уравнение
    # Исходное: u_xx + u_yy = -1 - u - lambda*u^2
    # Переносим все влево: u_xx + u_yy + 1 + u + lambda*u^2 = 0
    residual = u_xx + u_yy + 1 + u + lambda_param * (u ** 2)
    
    return residual

def get_anchor_points(n=1000):
    x = torch.rand((n, 1)).to(device)
    y = torch.rand((n, 1)).to(device)
    return x, y

x_anchor, y_anchor = get_anchor_points(N_PHYS)

# Генерируем данные один раз (можно и на каждой итерации, но так быстрее)
x_p, y_p, xy_b = generate_data(N_PHYS, N_BOUND)

history_lambda = []
history_u_max = []

print(">>> Step 0: Initial solution at lambda=0...")
model = PINN(init_lambda=0.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# Замораживаем лямбду для первого шага, чтобы найти точное решение именно для lambda=0
model.lam.requires_grad = False

for i in tqdm(range(ITERATIONS)):
    optimizer.zero_grad()
    
    res = compute_pde_residual(model, x_p, y_p)
    loss_phys = torch.mean(res**2)
    
    u_b = model(xy_b[:,0:1], xy_b[:,1:2])
    loss_b = torch.mean(u_b**2)
    
    loss = loss_phys + 10 * loss_b # Вес граничных условий побольше
    loss.backward()
    optimizer.step()

# Сохраняем "предыдущее" состояние (u_prev, lambda_prev)
# Для u_prev мы просто запокаем предсказания на якорных точках
with torch.no_grad():
    u_prev = model(x_anchor, y_anchor)
    lam_prev = model.lam.item()
    max_val = u_prev.max().item()

history_lambda.append(lam_prev)
history_u_max.append(max_val)

print(f"Start Point: Lambda={lam_prev:.4f}, Max|u|={max_val:.4f}")

# --- ШАГ 1: Вторая точка (нужна для вектора направления) ---
# Чуть сдвинем лямбду и дообучим, чтобы получить направление касательной
print(">>> Step 1: Second solution at lambda=0.5 (to get direction)...")
model.lam.requires_grad = False
model.lam.data = torch.tensor([0.5]).to(device) # Принудительно ставим сдвиг

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
    u_curr = model(x_anchor, y_anchor)
    lam_curr = model.lam.item()
    max_val = u_curr.max().item()

history_lambda.append(lam_curr)
history_u_max.append(max_val)
print(f"Second Point: Lambda={lam_curr:.4f}, Max|u|={max_val:.4f}")

# Теперь у нас есть две точки, мы можем вычислить касательную (тангенс)
# Tangent vector: t = (u_curr - u_prev, lam_curr - lam_prev)
# Но нужно нормализовать

# --- ШАГ 2: ЦИКЛ ПО ДЛИНЕ ДУГИ (Pseudo-Arclength Loop) ---
print(f">>> Continuation process for {N_STEPS} steps...")

# Теперь Лямбда должна меняться
model.lam.requires_grad = True
# Создаем новый оптимизатор, который включает model.lam
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Переменные для хранения "предыдущего" шага для цикла
u_prev_vec = u_curr.clone()   # Вектор значений u на anchor точках
lam_prev_val = lam_curr

for step in range(N_STEPS):
    # 1. Вычисляем касательную (направление secant)
    # diff_u имеет размер (N_anchor, 1)
    diff_u = u_prev_vec - u_prev_vec 
    diff_lam = lam_prev_val - lam_prev_val

    # Нормализация касательной
    # Скалярное произведение для функций считаем как среднее (или интеграл)
    # Norm^2 = <du, du> + dlam^2
    norm = torch.sqrt(torch.mean(diff_u**2) + diff_lam**2)
    
    tau_u = diff_u / norm       # Компонента касательной по u
    tau_lam = diff_lam / norm   # Компонента касательной по lambda

    # 2. ПРЕДСКАЗАНИЕ (Predictor)
    # Делаем линейный шаг вдоль касательной
    # u_guess = u_prev + DS * tau
    # lam_guess = lam_prev + DS * tau
    
    # Прямая манипуляция весами модели сложна для предиктора, 
    # поэтому мы просто устанавливаем начальное значение лямбды, 
    # а веса сети оставим с прошлого шага (это работает как "близкое" приближение).
    # Но лямбду сдвинем явно:
    with torch.no_grad():
        model.lam.data += DS * tau_lam 
        # Если бы мы могли легко сдвинуть веса сети вдоль tau_u, мы бы сделали это,
        # но для NN это сложно, поэтому надеемся, что оптимизатор дотянет.

    # 3. КОРРЕКЦИЯ (Corrector) - Обучение PINN
    # Теперь ищем решение, которое удовлетворяет PDE и находится на расстоянии DS 
    # от (u_prev_vec, lam_prev_val) в направлении касательной.
    
    pbar = tqdm(range(ITERATIONS), leave=False)

    for i in pbar:
        optimizer.zero_grad()
        
        # A. PDE Loss
        res = compute_pde_residual(model, x_p, y_p)
        loss_pde = torch.mean(res**2)
        
        # B. Boundary Loss
        u_b = model(xy_b[:,0:1], xy_b[:,1:2])
        loss_b = torch.mean(u_b**2)
        
        # C. Arclength Loss (Constraint)
        # Уравнение: <u - u_prev, tau_u> + (lam - lam_prev)*tau_lam - DS = 0
        u_anchor_pred = model(x_anchor, y_anchor)
        
        # Скалярное произведение функций (approximation via mean)
        dot_u = torch.mean((u_anchor_pred - u_prev_vec) * tau_u)
        dot_lam = (model.lam - lam_prev_val) * tau_lam
        
        # Мы хотим, чтобы (dot_u + dot_lam) == DS
        loss_arc = ((dot_u + dot_lam) - DS) ** 2
        
        # Суммарный лосс
        loss = loss_pde + 10 * loss_b + 10 * loss_arc
        
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            pbar.set_description(f"S:{step} L:{model.lam.item():.2f} Loss:{loss.item():.5f}")

    # 4. Сохранение результатов шага
    with torch.no_grad():
        u_new_vec = model(x_anchor, y_anchor)
        lam_new_val = model.lam.item()
        
        # Обновляем историю
        history_lambda.append(lam_new_val)
        history_u_max.append(u_new_vec.max().item())
        
        # Сдвигаем буферы для следующего шага
        u_pprev_vec = u_prev_vec.clone()
        lam_pprev_val = lam_prev_val
        
        u_prev_vec = u_new_vec.clone()
        lam_prev_val = lam_new_val
        
    print(f"Step {step+1}: Lambda={lam_new_val:.4f}, Max|u|={u_new_vec.max().item():.4f}")

plt.figure(figsize=(8, 6))
plt.plot(history_lambda, history_u_max, '-o', markersize=4, label='PINN Bifurcation Path')
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$max|u|$')
plt.title('Bifurcation Diagram via Pseudo-Arclength Continuation')
plt.grid(True)
plt.savefig('bifurcation_diagram.png')
print("Диаграмма сохранена как bifurcation_diagram.png")
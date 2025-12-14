import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Выбираем устройство (GPU если есть, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

class PINN(nn.Module):
    def __init__(self):
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

def compute_pde_residual(model, x, y, lambda_param=1.0):
    # 1. Предсказание сети u(x, y)
    u = model(x, y)
    
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

# --- Гиперпараметры ---
LAMBDA = 5.0  # Параметр из твоего уравнения
ITERATIONS = 5000
LR = 0.001
N_PHYS = 10000 # Точек внутри
N_BOUND = 2500 # Точек на одной стороне границы

# Инициализация модели и оптимизатора
model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Генерируем данные один раз (можно и на каждой итерации, но так быстрее)
x_p, y_p, xy_b = generate_data(N_PHYS, N_BOUND)

loss_history = []

print("Начинаем обучение...")

for epoch in tqdm(range(ITERATIONS)):
    optimizer.zero_grad()
    
    # --- 1. Потеря на уравнении (Physics Loss) ---
    res = compute_pde_residual(model, x_p, y_p, LAMBDA)
    loss_physics = torch.mean(res ** 2)
    
    # --- 2. Потеря на границах (Boundary Loss) ---
    # Граничные точки у нас в одном тензоре (N, 2), разделим для подачи в сеть
    x_b = xy_b[:, 0:1]
    y_b = xy_b[:, 1:2]
    
    u_boundary_pred = model(x_b, y_b)
    # Мы хотим, чтобы u на границе было равно 0
    u_boundary_target = torch.zeros_like(u_boundary_pred)
    # loss_boundary = torch.mean((u_boundary_pred - u_boundary_target) ** 2)
    loss_boundary = torch.mean((u_boundary_pred) ** 2)
    
    # --- 3. Общая потеря ---
    # Иногда граничному условию дают больший вес, например 10 * loss_boundary
    loss = loss_physics + loss_boundary
    
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())
    
    # if epoch % 500 == 0:
    #     print(f"Epoch {epoch}: Loss = {loss.item():.6f} (Phys: {loss_physics.item():.6f}, BC: {loss_boundary.item():.6f})")

print("Обучение завершено.")

# Создаем сетку для рисования
x_vals = np.linspace(0, 1, 100)
y_vals = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Преобразуем в тензоры для PyTorch
x_grid = torch.tensor(X.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)
y_grid = torch.tensor(Y.flatten(), dtype=torch.float32).reshape(-1, 1).to(device)

# Предсказываем (отключаем расчет градиентов для скорости)
with torch.no_grad():
    u_pred = model(x_grid, y_grid)
    u_pred = u_pred.cpu().numpy().reshape(100, 100)

final_loss = loss_history[-1]

# Рисуем
plt.figure(figsize=(12, 5))

# График 1: Тепловая карта решения
plt.subplot(1, 2, 1)
plt.pcolormesh(X, Y, u_pred, cmap='jet', shading='auto')
plt.colorbar(label='u(x, y)')
plt.title(f'PINN Solution (lambda={LAMBDA})')
plt.xlabel('x')
plt.ylabel('y')

# График 2: Кривая падения ошибки (Loss)
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.yscale('log')
plt.title('Loss History')
plt.text(0.5, 0.9, f'Final Loss: {final_loss:.6f}', transform=plt.gca().transAxes, fontsize=12, ha='center')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')

plt.tight_layout()
# plt.show()
plt.savefig('pinn_solution.png')

PATH = '2_7_7_1.pth'
torch.save(model.state_dict(), PATH)

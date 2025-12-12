import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. Определение Нейросети
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Вход: 2 нейрона (x, y)
        # Скрытые слои: 4 слоя по 50 нейронов
        # Выход: 1 нейрон (u)
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(), # Tanh хорош для вторых производных
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )
        
    def forward(self, x, y):
        # Объединяем x и y в один тензор
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

# 2. Функция для вычисления PDE Loss
def physics_loss(model, x, y, lambda_param):
    # Важно: включаем create_graph=True для вычисления высших производных
    u = model(x, y)
    
    # Первые производные du/dx и du/dy
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    
    # Вторые производные d2u/dx2 и d2u/dy2
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    
    # Уравнение: u_xx + u_yy + u + lambda*u^2 + 1 = 0
    residual = u_xx + u_yy + u + lambda_param * (u ** 2) + 1
    
    loss_f = torch.mean(residual ** 2)
    return loss_f

# 3. Параметры обучения
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lambda_param = 1.0 # Ваша лямбда
iterations = 5000 # Количество эпох

# 4. Цикл обучения
for epoch in range(iterations):
    optimizer.zero_grad()
    
    # --- A. Точки внутри области (для физики) ---
    # Генерируем случайные x, y в диапазоне [0, 1]
    x_collocation = torch.rand(2000, 1, requires_grad=True)
    y_collocation = torch.rand(2000, 1, requires_grad=True)
    
    loss_physics = physics_loss(model, x_collocation, y_collocation, lambda_param)
    
    # --- B. Граничные условия (Boundary Conditions) ---
    # u = 0 на границах x=0, x=1, y=0, y=1
    
    # Генерируем точки для границ (по 100 точек на сторону)
    # Левая (0, y) и Правая (1, y)
    y_b = torch.rand(200, 1)
    zeros = torch.zeros(200, 1)
    ones = torch.ones(200, 1)
    
    # Верхняя (x, 1) и Нижняя (x, 0)
    x_b = torch.rand(200, 1)
    
    # Собираем все координаты границ
    x_bound = torch.cat([zeros, ones, x_b, x_b], dim=0) # x координаты для Left, Right, Bot, Top
    y_bound = torch.cat([y_b, y_b, zeros, ones], dim=0) # y координаты для Left, Right, Bot, Top
    
    # Предсказание на границах
    u_bound_pred = model(x_bound, y_bound)
    
    # Целевое значение - везде 0
    u_bound_target = torch.zeros_like(u_bound_pred)
    
    loss_bc = torch.nn.MSELoss()(u_bound_pred, u_bound_target)
    
    # --- C. Общий Loss ---
    # Можно добавить веса, например loss = loss_physics + 10 * loss_bc
    loss = loss_physics + loss_bc
    
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f} (Phys: {loss_physics.item():.6f}, BC: {loss_bc.item():.6f})")

# 5. Визуализация результата
with torch.no_grad():
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    x_tensor = torch.tensor(X.flatten()[:, None], dtype=torch.float32)
    y_tensor = torch.tensor(Y.flatten()[:, None], dtype=torch.float32)
    
    U_pred = model(x_tensor, y_tensor).numpy().reshape(100, 100)

plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, U_pred, cmap='jet', shading='auto')
plt.colorbar(label='u(x,y)')
plt.title(f'Solution u(x,y) with $\lambda={lambda_param}$')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

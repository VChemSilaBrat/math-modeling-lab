import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# --- Гиперпараметры ---
ITERATIONS = 2000     # Количество итераций обучения для каждой точки
LR = 0.001           # Скорость обучения
N_PHYS = 5000        # Точки внутри области для PDE
N_BOUND = 1000       # Точки на границе
N_ANCHOR = 1000      # Точки для расчета нормы ||u|| при продолжении
DS = 0.1             # Длина шага дуги (Arc-length step size)
N_STEPS = 50         # Сколько точек бифуркационной диаграммы строить
LAMBDA_SCALE = 1.0   # Вес для лямбды в уравнении длины дуги (бета из статьи)
FIRST_LAMBDA = -1.5   # Начальное значение для бифуркации
SECOND_LAMBDA = -1.3  # Конечное значение для бифуркации

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используем устройство: {device}")

# Оптимизация CUDA
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # TF32 для матричных операций
torch.backends.cudnn.allow_tf32 = True         # TF32 для cuDNN
torch.set_float32_matmul_precision('high')     # Использовать Tensor Cores

class GaussianActivation(nn.Module):
    """Гауссова активация: σ(x) = exp(-x²/2)"""
    def forward(self, x):
        return torch.exp(-x**2 / 2)

class PINN(nn.Module):
    def __init__(self, init_lambda=0.0):
        super(PINN, self).__init__()
        # Вход: (x, y) -> 2 нейрона
        self.net = nn.Sequential(
            nn.Linear(2, 10),  # Чуть увеличил ширину слоя (рекомендация для 4D/5D, но полезно для стабильности)
            GaussianActivation(),
            nn.Linear(10, 10),
            GaussianActivation(),
            nn.Linear(10, 1)
        )
        # Параметр бифуркации
        self.lam = nn.Parameter(torch.tensor([init_lambda], dtype=torch.float32))

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

def generate_data(n_physics, n_boundary):
    """Генерация точек коллокации"""
    x_phys = torch.rand((n_physics, 1), device=device, requires_grad=True)
    y_phys = torch.rand((n_physics, 1), device=device, requires_grad=True)

    rand_vals = torch.rand((n_boundary, 1), device=device)
    zeros = torch.zeros((n_boundary, 1), device=device)
    ones = torch.ones((n_boundary, 1), device=device)

    # ГУ: Лево, Право, Низ, Верх
    left   = torch.cat([zeros, rand_vals], dim=1)
    right  = torch.cat([ones, rand_vals], dim=1)
    bottom = torch.cat([rand_vals, zeros], dim=1)
    top    = torch.cat([rand_vals, ones], dim=1)

    boundary_points = torch.cat([left, right, bottom, top], dim=0)
    return x_phys, y_phys, boundary_points

def compute_pde_residual(model, x, y):
    """Вычисление невязки уравнения: u_xx + u_yy + 1 + u + lambda*u^2 = 0"""
    u = model(x, y)

    # Первые производные
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Вторые производные
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    residual = u_xx + u_yy + 1 + u + model.lam * (u ** 2)
    return residual

def extrapolate_weights(model_prev, model_prev2):
    """
    Предиктор: W_new = 2 * W_prev - W_prev2
    Возвращает словарь весов для инициализации новой модели.
    """
    dict_prev = model_prev.state_dict()
    dict_prev2 = model_prev2.state_dict()
    new_dict = {}

    for key in dict_prev:
        # Линейная экстраполяция параметров
        new_dict[key] = 2 * dict_prev[key] - dict_prev2[key]

    return new_dict

# --- ПОДГОТОВКА ДАННЫХ ---
# Якорные точки фиксируются один раз для корректного расчета "длины дуги"
# Если их перегенерировать каждый раз, мера расстояния будет случайной
x_anchor = torch.rand((N_ANCHOR, 1), device=device)
y_anchor = torch.rand((N_ANCHOR, 1), device=device)

# Точки обучения
x_p, y_p, xy_b = generate_data(N_PHYS, N_BOUND)

# История для графика
plot_lambda = []
plot_norm_u = []

# Для метода продолжения нам нужно хранить ДВА предыдущих состояния модели
# (для расчета направления касательной)
prev_model = None   # Шаг k-1
prev2_model = None  # Шаг k-2

print("\n========== НАЧАЛО: Поиск двух начальных точек ==========")

# -------------------------------------------------------------------
# ШАГ 0: Начальное решение при lambda = 0.0
# -------------------------------------------------------------------
model_0 = PINN(init_lambda=FIRST_LAMBDA).to(device)
model_0.lam.requires_grad = False # Фиксируем лямбду
optimizer_0 = torch.optim.Adam(model_0.parameters(), lr=LR)

print(f">>> Step 0: Training Initial Point (fixed lambda={FIRST_LAMBDA})")
for i in tqdm(range(ITERATIONS)):
    optimizer_0.zero_grad()
    res = compute_pde_residual(model_0, x_p, y_p)
    loss_phys = torch.mean(res**2)
    u_b = model_0(xy_b[:,0:1], xy_b[:,1:2])
    loss_b = torch.mean(u_b**2)
    loss = loss_phys + 100 * loss_b # Вес ГУ обычно выше
    loss.backward()
    optimizer_0.step()

# Сохраняем модель как k-2
prev2_model = copy.deepcopy(model_0)
with torch.no_grad():
    u_val = model_0(x_anchor, y_anchor)
    # Используем L2 норму вектора значений на якорях как меру ||u||
    norm_val = torch.max(u_val)
    plot_lambda.append(model_0.lam.item())
    plot_norm_u.append(norm_val.item())

# -------------------------------------------------------------------
# ШАГ 1: Второе решение при lambda = 0.1 (маленький шаг для старта)
# -------------------------------------------------------------------
# Инициализируем весами от шага 0 для быстрой сходимости
model_1 = PINN(init_lambda=0.0).to(device)
model_1.load_state_dict(model_0.state_dict())
# Меняем лямбду вручную
model_1.lam.data = torch.tensor([SECOND_LAMBDA], device=device) # Чуть сдвигаем
model_1.lam.requires_grad = False
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=LR)

print(f">>> Step 1: Training Second Point (fixed lambda={SECOND_LAMBDA})")
for i in tqdm(range(ITERATIONS)):
    optimizer_1.zero_grad()
    res = compute_pde_residual(model_1, x_p, y_p)
    loss_phys = torch.mean(res**2)
    u_b = model_1(xy_b[:,0:1], xy_b[:,1:2])
    loss_b = torch.mean(u_b**2)
    loss = loss_phys + 100 * loss_b
    loss.backward()
    optimizer_1.step()

# Сохраняем модель как k-1
prev_model = copy.deepcopy(model_1)
with torch.no_grad():
    u_val = model_1(x_anchor, y_anchor)
    norm_val = torch.max(u_val)
    plot_lambda.append(model_1.lam.item())
    plot_norm_u.append(norm_val.item())

print(f"Init Points: L:{plot_lambda[-2]:.2f}->{plot_lambda[-1]:.2f}")


print("\n========== ЗАПУСК PSEUDO-ARCLENGTH CONTINUATION ==========")

# Текущая рабочая модель
curr_model = PINN().to(device)

for step in range(N_STEPS):
    # ---------------------------------------------------------------
    # А. ПРЕДИКТОР (Predictor Step)
    # Используем касательную экстраполяцию весов и лямбды
    # ---------------------------------------------------------------

    # 1. Вычисляем предсказанные веса: W_new = 2*W_{k-1} - W_{k-2}
    predicted_state_dict = extrapolate_weights(prev_model, prev2_model)
    curr_model.load_state_dict(predicted_state_dict)

    # Лямбда тоже экстраполируется (она внутри state_dict, но иногда полезно удостовериться)
    # Важно: делаем лямбду обучаемой!
    curr_model.lam.requires_grad = True

    # ---------------------------------------------------------------
    # В. ПОДГОТОВКА КОРРЕКТОРА
    # ---------------------------------------------------------------

    # Нам нужны фиксированные значения u_{k-1} и lambda_{k-1} для уравнения ограничения
    with torch.no_grad():
        u_prev_fixed = prev_model(x_anchor, y_anchor).detach()
        lam_prev_fixed = prev_model.lam.item()

    # Оптимизатор теперь обновляет И веса сети, И лямбду
    optimizer = torch.optim.Adam(curr_model.parameters(), lr=LR)

    # Перегенерируем точки обучения (стохастичность помогает не застревать)
    x_p, y_p, xy_b = generate_data(N_PHYS, N_BOUND)

    # ---------------------------------------------------------------
    # С. КОРРЕКТОР (Corrector Step / Training Cycle)
    # ---------------------------------------------------------------
    if step % 5 == 0:
        print(f"Continuation Step {step+1}/{N_STEPS}...")

    for i in range(ITERATIONS):
        optimizer.zero_grad()

        # 1. Loss PDE
        res = compute_pde_residual(curr_model, x_p, y_p)
        loss_phys = torch.mean(res**2)

        # 2. Loss BC
        u_b = curr_model(xy_b[:,0:1], xy_b[:,1:2])
        loss_b = torch.mean(u_b**2)

        # 3. Loss Arclength (Уравнение длины дуги)
        # Мы хотим, чтобы sqrt( ||u - u_prev||^2 + (lam - lam_prev)^2 ) = DS
        # Чтобы избежать корней, используем квадраты:
        # ||u - u_prev||^2 + (lam - lam_prev)^2 - DS^2 = 0

        u_curr_anchor = curr_model(x_anchor, y_anchor)

        # Квадрат евклидова расстояния по решению (нормированный на кол-во точек)
        dist_u_sq = torch.mean((u_curr_anchor - u_prev_fixed)**2)

        # Квадрат расстояния по параметру
        dist_lam_sq = (curr_model.lam - lam_prev_fixed)**2

        # Уравнение связи (умножаем dist_u на масштаб, чтобы сбалансировать с лямбдой)
        # Здесь N_ANCHOR не нужен, если мы взяли mean, но для согласованности с DS:
        # Пусть arc_residual = (dist_u_sq + dist_lam_sq - DS**2)

        # Важно: DS здесь "в пространстве mean-квадратов".
        # Если u меняется на 0.1 в среднем, то dist_u_sq ~ 0.01.
        arc_eqn = dist_u_sq + LAMBDA_SCALE * dist_lam_sq - (DS**2 * 0.1) # Скейлинг DS эмпирический
        loss_arc = arc_eqn**2

        # TOTAL LOSS
        # Увеличиваем вес arc-length, чтобы точка не убегала с дуги
        loss = loss_phys + 100 * loss_b + 100 * loss_arc

        loss.backward()
        optimizer.step()

    # ---------------------------------------------------------------
    # D. ОБНОВЛЕНИЕ ИСТОРИИ
    # ---------------------------------------------------------------
    with torch.no_grad():
        u_final = curr_model(x_anchor, y_anchor)
        norm_final = torch.max(u_final)
        lam_final = curr_model.lam.item()

        # Логи DEBUG
        if step % 5 == 0:
            print(f"   -> Result: Lam={lam_final:.4f}, Norm={norm_final:.4f}, LossArc={loss_arc.item():.6f}")

    # Сохраняем данные для графика
    plot_lambda.append(lam_final)
    plot_norm_u.append(norm_final.item())

    # Сдвиг очереди моделей:
    # k-2 становится старым k-1
    prev2_model = copy.deepcopy(prev_model)
    # k-1 становится текущим решением
    prev_model = copy.deepcopy(curr_model)

# --- ВИЗУАЛИЗАЦИЯ ---
plt.figure(figsize=(10, 6))
plt.plot(plot_lambda, plot_norm_u, 'o-', markersize=4, label='PINN Bifurcation Path')
plt.xlabel(r'Bifurcation parameter $\lambda$')
plt.ylabel(r'Solution norm $max(u)$')
plt.title('Arclength Continuation for Nonlinear PDE')
plt.grid(True)
plt.legend()
# plt.show()
plt.savefig('bif_corr_with_Adam.png')

# Визуализация последнего решения
with torch.no_grad():
    x_plot = torch.linspace(0, 1, 100).to(device)
    y_plot = torch.linspace(0, 1, 100).to(device)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    shape = X.shape
    curr_model.eval()
    U = curr_model(X.reshape(-1, 1), Y.reshape(-1, 1)).reshape(shape)

plt.figure(figsize=(6, 5))
plt.contourf(X.cpu(), Y.cpu(), U.cpu(), levels=50, cmap='viridis')
plt.colorbar(label='u(x,y)')
plt.title(f'Solution at $\lambda$ = {plot_lambda[-1]:.3f}')
plt.show()

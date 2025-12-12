import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import copy

# --- 1. Настройка GPU/CPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Класс PINN с обучаемой Lambda ---
class PINN_ArcLength(nn.Module):
    def __init__(self, init_lambda=0.0):
        super().__init__()
        # Обычная полносвязная сеть для u(x,y)
        self.net = nn.Sequential(
            nn.Linear(2, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Tanh(),
            nn.Linear(40, 1)
        )
        # Lambda теперь параметр, который мы учим!
        self.lambda_param = nn.Parameter(torch.tensor([float(init_lambda)], device=device))

    def forward(self, x, y):
        inputs = torch.cat([x, y], axis=1)
        return self.net(inputs)

# --- 3. Вспомогательные функции ---

def get_derivatives(model, x, y):
    """Вычисляет u, u_xx, u_yy автоматически"""
    u = model(x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u, u_xx, u_yy

def compute_loss(model, x_f, y_f, x_b, y_b, 
                 use_arclength=False, 
                 u_prev=None, lambda_prev=None,
                 tangent_u=None, tangent_lambda=None, 
                 ds=0.1, anchor_points_x=None, anchor_points_y=None):
    
    # 1. PDE Resudual
    # u_xx + u_yy + 1 + u + lambda*u^2 = 0
    u, u_xx, u_yy = get_derivatives(model, x_f, y_f)
    residual = u_xx + u_yy + 1 + u + model.lambda_param * (u**2)
    loss_pde = torch.mean(residual**2)
    
    # 2. Boundary Conditions
    u_b = model(x_b, y_b)
    loss_bc = torch.mean(u_b**2)
    
    loss_arclength = torch.tensor(0.0, device=device)
    
    # 3. Arc-Length Constraint (Геометрическое ограничение)
    if use_arclength:
        # Вычисляем u на фиксированных опорных точках для сравнения с прошлым шагом
        # Мы не можем просто вычесть веса нейросети, нужно вычитать значения функции
        u_curr_anchor = model(anchor_points_x, anchor_points_y)
        
        # Скалярное произведение в функциональном пространстве (через сумму)
        # dot = sum((u - u_prev) * tau_u) + (lam - lam_prev) * tau_lam
        diff_u = u_curr_anchor - u_prev
        diff_lam = model.lambda_param - lambda_prev
        
        # Это уравнение плоскости, перпендикулярной касательной
        projection = torch.sum(diff_u * tangent_u) + diff_lam * tangent_lambda
        
        # Мы хотим, чтобы проекция шага на касательную была равна ds
        loss_arclength = (projection - ds)**2

    return loss_pde, loss_bc, loss_arclength

# --- 4. Процесс Continuation ---

def run_continuation_process():
    # Параметры задачи
    params = {
        'n_steps': 30,         # Сколько шагов по кривой делать
        'ds': 0.5,             # Длина шага дуги (Arc-length step size)
        'epochs_per_step': 2500,
        'lr': 1e-3
    }
    
    # Генерация фиксированных точек для Collocation, границы и Anchor points (для аркленс)
    # Генерируем их один раз, чтобы "касательная" считалась в одном пространстве
    N_f = 2000
    x_f = torch.rand(N_f, 1, requires_grad=True, device=device)
    y_f = torch.rand(N_f, 1, requires_grad=True, device=device)
    
    # Границы
    pad = torch.linspace(0, 1, 100, device=device).view(-1, 1)
    zeros = torch.zeros_like(pad)
    ones = torch.ones_like(pad)
    x_b = torch.cat([zeros, ones, pad, pad])
    y_b = torch.cat([pad, pad, zeros, ones])
    
    # Опорные точки для вычисления "вектора" функции u (для скалярного произведения)
    # Сетка 20x20
    x_mesh = torch.linspace(0, 1, 20, device=device)
    y_mesh = torch.linspace(0, 1, 20, device=device)
    X_grid, Y_grid = torch.meshgrid(x_mesh, y_mesh, indexing='xy')
    x_anchor = X_grid.flatten().unsqueeze(1)
    y_anchor = Y_grid.flatten().unsqueeze(1)

    # Инициализация
    model = PINN_ArcLength(init_lambda=0.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    history_lambda = []
    history_u_max = []
    
    # Хранилище предыдущих решений (u_values_on_anchors, lambda_value)
    prev_solutions = [] 

    print("=== Start Bifurcation Analysis ===")

    for step in range(params['n_steps']):
        
        # --- Подготовка Arc-Length компонентов ---
        use_al = False
        u_prev_anchor = None
        lam_prev = None
        tau_u = None
        tau_lam = None
        
        # Нам нужно минимум 2 точки, чтобы определить касательную.
        # Шаг 0: Просто решаем задачу для lambda ~ 0 (Warm up)
        # Шаг 1: Делаем небольшой шаг по lambda (Secant method start)
        # Шаг 2+: Полноценный Arc-length
        
        if step == 0:
            print(f"Step {step}: Finding initial solution for lambda=0...")
            current_ds = 0
        elif step == 1:
            print(f"Step {step}: Finding second point to establish direction...")
            # Принудительно толкаем лямбду немного вперед через Loss penalty
            # или просто меняем параметр, но здесь мы просто позволим ей дрейфовать
            # Для надежности в коде ниже аркленс выключен, но мы можем 
            # немного заморозить лямбду или поменять начальное предположение
            
            # Хак для старта: вручную сдвигаем лямбду перед обучением
            with torch.no_grad():
                model.lambda_param.add_(0.5) 
            current_ds = 0
        else:
            use_al = True
            current_ds = params['ds']
            print(f"Step {step}: Arc-length continuation (ds={current_ds})...")
            
            # Достаем два прошлых решения
            u_k, lam_k = prev_solutions[-1]
            u_km1, lam_km1 = prev_solutions[-2]
            
            # 1. Вычисляем касательный вектор (Tangent)
            # v = (u_k - u_km1, lam_k - lam_km1)
            tangent_u_raw = u_k - u_km1
            tangent_lam_raw = lam_k - lam_km1
            
            # 2. Нормализуем касательный вектор (Normalization)
            # norm = sqrt(sum(tu^2) + tlam^2)
            norm = torch.sqrt(torch.sum(tangent_u_raw**2) + tangent_lam_raw**2)
            
            tau_u = tangent_u_raw / norm
            tau_lam = tangent_lam_raw / norm
            
            # 3. Предиктор (Prediction step)
            # Делаем "умную" инициализацию весов для следующего шага
            # u_new_guess = u_k + ds * tau_u
            # Но мы не можем легко изменить веса сети, чтобы она выдавала u_new_guess.
            # Поэтому мы используем веса с прошлого шага (u_k) как начальное приближение (Warm start),
            # но СДВИГАЕМ значение lambda вручную на шаг предиктора.
            
            u_prev_anchor = u_k
            lam_prev = lam_k
            
            with torch.no_grad():
                # Обновляем guess для lambda параметра
                model.lambda_param.data = lam_k + current_ds * tau_lam

        # --- Цикл обучения (Newton-Correction via Adam) ---
        model.train()
        for epoch in range(params['epochs_per_step']):
            optimizer.zero_grad()
            
            lp, lb, lal = compute_loss(
                model, x_f, y_f, x_b, y_b, 
                use_arclength=use_al,
                u_prev=u_prev_anchor, lambda_prev=lam_prev,
                tangent_u=tau_u, tangent_lambda=tau_lam,
                ds=current_ds, anchor_points_x=x_anchor, anchor_points_y=y_anchor
            )
            
            # Веса лоссов
            loss = lp + 10 * lb + 10 * lal
            
            loss.backward()
            optimizer.step()
            
            if epoch % 1000 == 0:
                cur_lam = model.lambda_param.item()
                print(f"  Ep {epoch}: Loss {loss.item():.5f} (P:{lp:.5f} B:{lb:.5f} A:{lal:.5f}) | Lam: {cur_lam:.4f}")

        # --- Сохранение результатов шага ---
        with torch.no_grad():
            cur_u_anchor = model(x_anchor, y_anchor)
            cur_lam = model.lambda_param.clone() # важно clone, чтобы отвязать от графа
            
            # Сохраняем "snapshot" решения
            prev_solutions.append((cur_u_anchor, cur_lam))
            
            # Для графика берем макс значение u
            u_max = torch.max(cur_u_anchor).item()
            lam_val = cur_lam.item()
            
            history_lambda.append(lam_val)
            history_u_max.append(u_max)
            print(f"-> Solved Step {step}: Lambda={lam_val:.4f}, U_max={u_max:.4f}")

    return history_lambda, history_u_max

# --- 5. Запуск и отрисовка ---
lams, u_maxs = run_continuation_process()

plt.figure(figsize=(8, 6))
plt.plot(lams, u_maxs, 'o-', linewidth=2, markersize=5, label='PINN Bifurcation Path')
plt.xlabel('Parameter $\lambda$')
plt.ylabel('$||u||_{\infty}$ (Max Amplitude)')
plt.title('Bifurcation Diagram using Arc-Length PINN')
plt.grid(True)
plt.legend()
plt.show()

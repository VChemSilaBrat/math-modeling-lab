import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from typing import Tuple, Dict, List

# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Параметры домена
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0

# Параметры сети
HIDDEN_LAYERS = [2, 7, 7, 1]

# Параметры обучения
LEARNING_RATE = 1e-3
ITERATIONS_INITIAL = 2000  # Для первых двух точек
ITERATIONS_CONTINUATION = 2000  # Для шагов продолжения

# Параметры continuation
LAMBDA_START = -1.0
LAMBDA_INITIAL_STEP = 0.2
DS = 0.05  # Шаг по длине дуги
N_CONTINUATION_STEPS = 20
WEIGHT_ARC = 5.0  # Вес arclength constraint в loss

# Количество точек
N_COLLOCATION = 1000  # Точки для PDE residual (обновляются каждую итерацию)
N_BOUNDARY = 200      # Точки для BC
N_FIXED = 2000        # Фиксированные точки для arclength constraint


# ============================================================================
# КЛАСС PINN С ОБУЧАЕМЫМ ПАРАМЕТРОМ LAMBDA
# ============================================================================
class PINN(nn.Module):
    """
    Physics-Informed Neural Network с обучаемым бифуркационным параметром.
    """
    def __init__(self, layers: List[int], lambda_init: float = 0.0):
        super(PINN, self).__init__()

        # Построение архитектуры
        self.layers_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers_list.append(nn.Linear(layers[i], layers[i+1]))

        # Инициализация весов (Xavier)
        for m in self.layers_list:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # Обучаемый параметр lambda (критически важно: requires_grad=True)
        self.lam = nn.Parameter(torch.tensor([lambda_init], dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход: u(x, y)
        """
        inputs = torch.cat([x, y], dim=1)

        for i, layer in enumerate(self.layers_list[:-1]):
            inputs = torch.tanh(layer(inputs))

        # Выходной слой без активации
        u = self.layers_list[-1](inputs)
        return u


# ============================================================================
# PDE RESIDUAL (ПРИМЕР: Swift-Hohenberg уравнение)
# ============================================================================
def compute_pde_residual(model: PINN, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Пример: Swift-Hohenberg уравнение
    (1 + Δ)²u + λu + u³ = 0

    Замените на свое уравнение!
    """
    x.requires_grad_(True)
    y.requires_grad_(True)

    u = model(x, y)

    # Первые производные
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]

    # Вторые производные
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

    laplacian = u_xx + u_yy

    # Третьи производные для (1+Δ)²
    lap_x = torch.autograd.grad(laplacian, x, torch.ones_like(laplacian), create_graph=True)[0]
    lap_y = torch.autograd.grad(laplacian, y, torch.ones_like(laplacian), create_graph=True)[0]

    lap_xx = torch.autograd.grad(lap_x, x, torch.ones_like(lap_x), create_graph=True)[0]
    lap_yy = torch.autograd.grad(lap_y, y, torch.ones_like(lap_y), create_graph=True)[0]

    bilaplacian = lap_xx + lap_yy

    # Swift-Hohenberg: (1 + Δ)²u = u + 2Δu + Δ²u
    operator = u + 2*laplacian + bilaplacian

    # Residual: (1 + Δ)²u + λu + u³ = 0
    residual = operator + model.lam * u + u**3

    return residual


# ============================================================================
# ФУНКЦИЯ ARCLENGTH CONSTRAINT
# ============================================================================
def compute_arclength_loss(
    model: PINN,
    x_fixed: torch.Tensor,
    y_fixed: torch.Tensor,
    u_prev_vals: torch.Tensor,
    lam_prev: float,
    tangent_u: torch.Tensor,
    tangent_lam: float,
    ds: float
) -> torch.Tensor:
    """
    Гиперплоскость (tangent plane constraint):
    ⟨u - u_prev, τ_u⟩ + (λ - λ_prev) · τ_λ - Δs = 0

    Args:
        u_prev_vals: значения u на предыдущем шаге в фиксированных точках
        tangent_u: нормализованная касательная для u
        tangent_lam: нормализованная касательная для λ
        ds: целевой шаг по длине дуги
    """
    # Текущие значения
    u_curr = model(x_fixed, y_fixed)
    lam_curr = model.lam

    # Разности
    diff_u = u_curr - u_prev_vals
    diff_lam = lam_curr - lam_prev

    # Проекция на касательную (усредняем по точкам как аппроксимация интеграла)
    projection_u = torch.mean(diff_u * tangent_u)
    projection_lam = diff_lam * tangent_lam

    # Ограничение: проекция должна равняться ds
    constraint = projection_u + projection_lam - ds

    return constraint ** 2


# ============================================================================
# ГЕНЕРАЦИЯ ТОЧЕК
# ============================================================================
def generate_collocation_points(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Случайные точки внутри домена для PDE residual"""
    x = torch.rand(n, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n, 1, device=DEVICE) * (Y_MAX - Y_MIN) + Y_MIN
    return x, y

def generate_boundary_points(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Точки на границе для граничных условий"""
    n_per_side = n // 4

    # Четыре стороны прямоугольника
    x_bottom = torch.rand(n_per_side, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
    y_bottom = torch.full_like(x_bottom, Y_MIN)

    x_top = torch.rand(n_per_side, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
    y_top = torch.full_like(x_top, Y_MAX)

    x_left = torch.full((n_per_side, 1), X_MIN, device=DEVICE)
    y_left = torch.rand(n_per_side, 1, device=DEVICE) * (Y_MAX - Y_MIN) + Y_MIN

    x_right = torch.full((n_per_side, 1), X_MAX, device=DEVICE)
    y_right = torch.rand(n_per_side, 1, device=DEVICE) * (Y_MAX - Y_MIN) + Y_MIN

    x_bc = torch.cat([x_bottom, x_top, x_left, x_right], dim=0)
    y_bc = torch.cat([y_bottom, y_top, y_left, y_right], dim=0)

    return x_bc, y_bc

def generate_fixed_points(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Фиксированный набор точек для вычисления норм и касательных.
    НЕ МЕНЯЕТСЯ на протяжении всего continuation!
    """
    x = torch.rand(n, 1, device=DEVICE) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(n, 1, device=DEVICE) * (Y_MAX - Y_MIN) + Y_MIN
    return x, y


# ============================================================================
# ОБУЧЕНИЕ ОДНОЙ ТОЧКИ (INITIAL SOLVE)
# ============================================================================
def solve_initial_point(
    model: PINN,
    lambda_value: float,
    iterations: int,
    verbose: bool = True
) -> Dict:
    """
    Решение для фиксированного λ (обычный PINN, без arclength constraint).
    Используется для получения первых двух точек.
    """
    # Замораживаем lambda на фиксированном значении
    model.lam.data.fill_(lambda_value)
    model.lam.requires_grad_(False)

    # Оптимизатор только для весов сети
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Граничные точки (фиксированные на весь процесс обучения)
    x_bc, y_bc = generate_boundary_points(N_BOUNDARY)

    loss_history = []

    for it in range(iterations):
        optimizer.zero_grad()

        # Коллокационные точки (обновляются каждую итерацию)
        x_col, y_col = generate_collocation_points(N_COLLOCATION)

        # PDE residual
        residual = compute_pde_residual(model, x_col, y_col)
        loss_pde = torch.mean(residual ** 2)

        # Граничные условия (пример: Dirichlet u=0)
        u_bc = model(x_bc, y_bc)
        loss_bc = torch.mean(u_bc ** 2)

        # Суммарная функция потерь
        loss = loss_pde + 10.0 * loss_bc  # Вес BC

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if verbose and it % 500 == 0:
            print(f"  Iter {it}/{iterations}, Loss: {loss.item():.6e}, "
                  f"PDE: {loss_pde.item():.6e}, BC: {loss_bc.item():.6e}")

    # Возвращаем lambda в режим обучения
    model.lam.requires_grad_(True)

    return {
        'loss_history': loss_history,
        'final_loss': loss_history[-1]
    }


# ============================================================================
# ОБУЧЕНИЕ С ARCLENGTH CONSTRAINT (CORRECTOR STEP)
# ============================================================================
def solve_corrector_step(
    model: PINN,
    x_fixed: torch.Tensor,
    y_fixed: torch.Tensor,
    u_prev_vals: torch.Tensor,
    lam_prev: float,
    tangent_u: torch.Tensor,
    tangent_lam: float,
    ds: float,
    iterations: int,
    verbose: bool = True
) -> Dict:
    """
    Корректор: обучение с arclength constraint.
    Lambda теперь обучаемый параметр!
    """
    # Оптимизатор для весов И lambda
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    x_bc, y_bc = generate_boundary_points(N_BOUNDARY)

    loss_history = []
    lambda_history = []

    for it in range(iterations):
        optimizer.zero_grad()

        x_col, y_col = generate_collocation_points(N_COLLOCATION)

        # 1. PDE residual
        residual = compute_pde_residual(model, x_col, y_col)
        loss_pde = torch.mean(residual ** 2)

        # 2. Граничные условия
        u_bc = model(x_bc, y_bc)
        loss_bc = torch.mean(u_bc ** 2)

        # 3. Arclength constraint
        loss_arc = compute_arclength_loss(
            model, x_fixed, y_fixed,
            u_prev_vals, lam_prev,
            tangent_u, tangent_lam, ds
        )

        # Суммарная функция потерь
        loss = loss_pde + 10.0 * loss_bc + WEIGHT_ARC * loss_arc

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        lambda_history.append(model.lam.item())

        if verbose and it % 500 == 0:
            print(f"  Iter {it}/{iterations}, Loss: {loss.item():.6e}, "
                  f"PDE: {loss_pde.item():.6e}, BC: {loss_bc.item():.6e}, "
                  f"Arc: {loss_arc.item():.6e}, λ: {model.lam.item():.4f}")

    return {
        'loss_history': loss_history,
        'lambda_history': lambda_history,
        'final_loss': loss_history[-1],
        'final_lambda': lambda_history[-1]
    }


# ============================================================================
# ВЫЧИСЛЕНИЕ НОРМЫ РЕШЕНИЯ (ДЛЯ БИФУРКАЦИОННОЙ ДИАГРАММЫ)
# ============================================================================
def compute_solution_norm(model: PINN, x_fixed: torch.Tensor, y_fixed: torch.Tensor) -> float:
    """
    Норма решения (например, L2 или max).
    Используется для построения бифуркационной диаграммы.
    """
    with torch.no_grad():
        u = model(x_fixed, y_fixed)
        # Вариант 1: L2-норма
        norm_l2 = torch.sqrt(torch.mean(u**2)).item()
        # Вариант 2: Max-норма
        norm_max = torch.max(torch.abs(u)).item()
        return norm_max  # Или norm_l2


# ============================================================================
# ГЛАВНЫЙ АЛГОРИТМ: PSEUDO-ARCLENGTH CONTINUATION
# ============================================================================
def pseudo_arclength_continuation():
    """
    Основной цикл continuation с предиктором-корректором.
    """
    print("="*80)
    print("PSEUDO-ARCLENGTH CONTINUATION FOR PINN")
    print("="*80)

    # ========================================================================
    # ФИКСИРОВАННЫЕ ТОЧКИ (не меняются на протяжении всего процесса!)
    # ========================================================================
    x_fixed, y_fixed = generate_fixed_points(N_FIXED)
    print(f"\n[INFO] Фиксированные точки для arclength: {N_FIXED}")

    # Списки для сохранения траектории
    continuation_data = []  # [(lambda, norm, state_dict), ...]

    # ========================================================================
    # ШАГ 0: Решение для λ₀
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"ШАГ 0: Решение для λ = {LAMBDA_START}")
    print(f"{'='*80}")

    model_0 = PINN(HIDDEN_LAYERS, lambda_init=LAMBDA_START).to(DEVICE)
    solve_initial_point(model_0, LAMBDA_START, ITERATIONS_INITIAL, verbose=True)

    # Сохраняем состояние
    with torch.no_grad():
        u_0 = model_0(x_fixed, y_fixed).detach().clone()
    lam_0 = LAMBDA_START
    state_0 = deepcopy(model_0.state_dict())
    norm_0 = compute_solution_norm(model_0, x_fixed, y_fixed)

    continuation_data.append({
        'lambda': lam_0,
        'norm': norm_0,
        'state': state_0,
        'u_fixed': u_0
    })

    print(f"[RESULT] λ₀ = {lam_0:.6f}, ||u||_max = {norm_0:.6f}")

    # ========================================================================
    # ШАГ 1: Решение для λ₁ = λ₀ + Δλ
    # ========================================================================
    lam_1 = LAMBDA_START + LAMBDA_INITIAL_STEP
    print(f"\n{'='*80}")
    print(f"ШАГ 1: Решение для λ = {lam_1}")
    print(f"{'='*80}")

    model_1 = PINN(HIDDEN_LAYERS, lambda_init=lam_1).to(DEVICE)
    solve_initial_point(model_1, lam_1, ITERATIONS_INITIAL, verbose=True)

    with torch.no_grad():
        u_1 = model_1(x_fixed, y_fixed).detach().clone()
    state_1 = deepcopy(model_1.state_dict())
    norm_1 = compute_solution_norm(model_1, x_fixed, y_fixed)

    continuation_data.append({
        'lambda': lam_1,
        'norm': norm_1,
        'state': state_1,
        'u_fixed': u_1
    })

    print(f"[RESULT] λ₁ = {lam_1:.6f}, ||u||_max = {norm_1:.6f}")

    # ========================================================================
    # ОСНОВНОЙ ЦИКЛ CONTINUATION (k = 2, 3, ..., N)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"НАЧАЛО ARCLENGTH CONTINUATION ({N_CONTINUATION_STEPS} шагов)")
    print(f"{'='*80}")

    for step in range(2, N_CONTINUATION_STEPS + 2):
        print(f"\n{'─'*80}")
        print(f"ШАГ {step} / {N_CONTINUATION_STEPS + 1}")
        print(f"{'─'*80}")

        # Получаем два предыдущих состояния
        data_prev = continuation_data[-1]
        data_prev2 = continuation_data[-2]

        u_prev = data_prev['u_fixed']
        lam_prev = data_prev['lambda']

        u_prev2 = data_prev2['u_fixed']
        lam_prev2 = data_prev2['lambda']

        # ====================================================================
        # ВЫЧИСЛЕНИЕ КАСАТЕЛЬНОЙ
        # ====================================================================
        with torch.no_grad():
            # Разности
            delta_u = u_prev - u_prev2
            delta_lam = lam_prev - lam_prev2

            # Нормировка (длина дуги предыдущего шага)
            # ||Δu||² + (Δλ)² = ds_prev²
            norm_delta_u = torch.sqrt(torch.mean(delta_u**2))
            arc_length_prev = torch.sqrt(norm_delta_u**2 + delta_lam**2)

            # Касательные (нормализованные)
            tangent_u = delta_u / arc_length_prev
            tangent_lam = delta_lam / arc_length_prev

            print(f"[TANGENT] ||τ_u|| = {torch.sqrt(torch.mean(tangent_u**2)).item():.6f}, "
                  f"τ_λ = {tangent_lam:.6f}")

        # ====================================================================
        # ПРЕДИКТОР: Линейная экстраполяция
        # ====================================================================
        with torch.no_grad():
            lam_pred = lam_prev + tangent_lam * DS
            # Для весов используем warm start от предыдущего решения

        print(f"[PREDICTOR] λ_pred = {lam_pred:.6f}")

        # ====================================================================
        # ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ДЛЯ КОРРЕКТОРА
        # ====================================================================
        model_new = PINN(HIDDEN_LAYERS, lambda_init=lam_pred).to(DEVICE)
        model_new.load_state_dict(data_prev['state'])  # Warm start
        model_new.lam.data.fill_(lam_pred)
        model_new.lam.requires_grad_(True)

        # ====================================================================
        # КОРРЕКТОР: Обучение с arclength constraint
        # ====================================================================
        print("[CORRECTOR] Запуск обучения с arclength constraint...")
        result = solve_corrector_step(
            model_new, x_fixed, y_fixed,
            u_prev, lam_prev,
            tangent_u, tangent_lam,
            DS, ITERATIONS_CONTINUATION,
            verbose=True
        )

        # ====================================================================
        # СОХРАНЕНИЕ РЕЗУЛЬТАТА
        # ====================================================================
        with torch.no_grad():
            u_new = model_new(x_fixed, y_fixed).detach().clone()

        lam_new = model_new.lam.item()
        state_new = deepcopy(model_new.state_dict())
        norm_new = compute_solution_norm(model_new, x_fixed, y_fixed)

        continuation_data.append({
            'lambda': lam_new,
            'norm': norm_new,
            'state': state_new,
            'u_fixed': u_new
        })

        print(f"\n[RESULT] λ_{step} = {lam_new:.6f}, ||u||_max = {norm_new:.6f}")
        print(f"[INFO] Изменение λ: {lam_new - lam_prev:+.6f}")

        # Проверка на расходимость
        if result['final_loss'] > 1e2:
            print("\n[WARNING] Loss слишком большой, возможно нужно уменьшить DS!")
            break

    # ========================================================================
    # ПОСТРОЕНИЕ БИФУРКАЦИОННОЙ ДИАГРАММЫ
    # ========================================================================
    print(f"\n{'='*80}")
    print("ПОСТРОЕНИЕ БИФУРКАЦИОННОЙ ДИАГРАММЫ")
    print(f"{'='*80}")

    lambdas = [d['lambda'] for d in continuation_data]
    norms = [d['norm'] for d in continuation_data]

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, norms, 'o-', linewidth=2, markersize=6)
    plt.xlabel('λ (Bifurcation Parameter)', fontsize=14)
    plt.ylabel('||u||_∞ (Max Norm)', fontsize=14)
    plt.title('Bifurcation Diagram via Pseudo-Arclength Continuation', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('bifurcation_diagram.png', dpi=300)
    plt.show()

    print(f"\n[DONE] Сохранено {len(continuation_data)} точек")
    print(f"[DONE] Диаграмма сохранена в 'bifurcation_diagram.png'")

    return continuation_data


# ============================================================================
# ЗАПУСК
# ============================================================================
if __name__ == "__main__":
    data = pseudo_arclength_continuation()

    # Дополнительная визуализация: траектория в пространстве (λ, u)
    print("\n[INFO] Визуализация нескольких решений...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-1]

    x_plot = torch.linspace(X_MIN, X_MAX, 100, device=DEVICE)
    y_plot = torch.linspace(Y_MIN, Y_MAX, 100, device=DEVICE)
    X, Y = torch.meshgrid(x_plot, y_plot, indexing='ij')
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)

    for idx, ax in zip(indices, axes):
        if idx >= len(data):
            continue

        # Загружаем модель
        model_vis = PINN(HIDDEN_LAYERS).to(DEVICE)
        model_vis.load_state_dict(data[idx]['state'])
        model_vis.eval()

        with torch.no_grad():
            U = model_vis(X_flat, Y_flat).cpu().numpy().reshape(100, 100)

        im = ax.contourf(X.cpu(), Y.cpu(), U, levels=20, cmap='RdBu_r')
        ax.set_title(f"λ = {data[idx]['lambda']:.3f}", fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('solution_snapshots.png', dpi=300)
    plt.show()

    print("[DONE] Снапшоты решений сохранены в 'solution_snapshots.png'")

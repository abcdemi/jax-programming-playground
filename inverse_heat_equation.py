import jax
nx = 50       # Количество пространственных точек
nt = 100      # Количество временных шагов
alpha = 0.01  # Коэффициент температуропроводности
dx = 1.0 / nx
dt = 0.001

def step_fn(u, _):
    # Вычисление второй производной (центральная разность)
    d2u = (jnp.roll(u, -1) - 2 * u + jnp.roll(u, 1)) / (dx ** 2)

    # Обновление температуры
    u_new = u + alpha * d2u * dt

    # Граничные условия (концы жестко поддерживаются при нулевой температуре)
    u_new = u_new.at[0].set(0.0)
    u_new = u_new.at[-1].set(0.0)

    # JAX scan требует возвращать (состояние_передаваемое_дальше, выход_для_истории)
    return u_new, None

# --- 3. Полная симуляция (Forward Pass) ---
@jax.jit
def simulate(u_init):
    # jax.lax.scan - это JAX-идиоматичный способ писать циклы.
    # Он применяет step_fn nt раз, что позволяет XLA компилятору круто все оптимизировать.
    u_final, _ = jax.lax.scan(step_fn, u_init, None, length=nt)
    return u_final

# --- 4. Функция потерь ---
@jax.jit
def loss_fn(u_init, target):
    u_final = simulate(u_init)
    # Насколько наш результат после симуляции далек от желаемой цели? (MSE)
    return jnp.mean((u_final - target) ** 2)

# --- 5. Настройка эксперимента ---
# Создаем "целевое" распределение (target) - например, Гауссиан (тепловой пик) в центре
x = jnp.linspace(0, 1, nx)
target_distribution = jnp.exp(-100 * (x - 0.5)**2)

# Наша начальная догадка для u_init (начнем с плоского нуля)
u_init_guess = jnp.zeros(nx)

# Магия дифференцируемого программирования:
# получаем функцию, которая возвращает и значение лосса, и градиенты по u_init
loss_and_grad_fn = jax.value_and_grad(loss_fn)

# --- 6. Цикл оптимизации ---
learning_rate = 50.0
epochs = 200

print("Начинаем оптимизацию начального состояния (Reverse Engineering через градиенты)...")
for i in range(epochs):
    # JAX "пробрасывает" градиенты через все 100 шагов симуляции Эйлера (BPTT)
    loss_val, grads = loss_and_grad_fn(u_init_guess, target_distribution)

    # Базовый шаг градиентного спуска
    u_init_guess = u_init_guess - learning_rate * grads

    if i % 20 == 0 or i == epochs - 1:
        print(f"Epoch {i:3d} | Loss (MSE): {loss_val:.6f}")

print("Оптимизация завершена!")

import matplotlib.pyplot as plt

# Вычисляем финальный результат после оптимизации
u_final_optimized = simulate(u_init_guess)

plt.figure(figsize=(10, 6))
plt.plot(x, target_distribution, 'r--', label='Цель (Target)', linewidth=2)
plt.plot(x, u_init_guess, 'g-', label='Найденное Начало (Optimized Init)')
plt.plot(x, u_final_optimized, 'b:', label='Итог симуляции (Final Result)')

plt.title("Обратная задача теплопроводности: поиск начального состояния")
plt.xlabel("Пространство (x)")
plt.ylabel("Температура (u)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
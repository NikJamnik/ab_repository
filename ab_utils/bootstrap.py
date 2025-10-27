import numpy as np
from typing import Callable, Tuple
from tqdm import tqdm


def permutation_test_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    metric_func: Callable[[np.ndarray], float] = np.mean,
    reps: int = 10000,
    alternative: str = 'two-sided'
) -> Tuple[float, float, np.ndarray]:
    """
    Выполняет перестановочный тест (Permutation Test) для оценки различий
    между двумя выборками на основе заданной метрики (по умолчанию среднего значения).

    Тест проверяет нулевую гипотезу H₀: значения статистик в выборках равны.
    p-value оценивается как доля перестановочных статистик, 
    более экстремальных, чем наблюдаемая разница.

    Этапы алгоритма:
        1. Вычисляется наблюдаемая статистика T_obs = metric(x) - metric(y)
        2. Объединяются выборки и многократно случайно перемешиваются метки групп
        3. Для каждой перестановки вычисляется новая статистика T*
        4. p-value = доля перестановочных значений |T*| >= |T_obs|
           (или с учётом направления альтернативной гипотезы)

    Args:
        x (np.ndarray): Первая выборка наблюдений.
        y (np.ndarray): Вторая выборка наблюдений.
        metric_func (Callable[[np.ndarray], float], optional): 
            Функция метрики, вычисляющая значение по выборке.
            По умолчанию используется `np.mean`.
        reps (int, optional): Количество перестановок (итераций). 
            Чем больше `reps`, тем точнее оценка p-value.
            По умолчанию 10_000.
        alternative (str, optional): Тип альтернативной гипотезы:
            - `'two-sided'` (по умолчанию): H₁: разность ≠ 0
            - `'greater'`: H₁: разность > 0
            - `'less'`: H₁: разность < 0

    Returns:
        Tuple[float, float, np.ndarray]:
            - p (float): p-value перестановочного теста;
            - T_obs (float): наблюдаемое значение статистики T;
            - T_stats (np.ndarray): массив перестановочных значений T*.

    Example:
        >>> x = np.random.normal(0, 1, 50)
        >>> y = np.random.normal(0.5, 1, 50)
        >>> p, T_obs, T_stats = permutation_test_pvalue(x, y)
        >>> print(f"T_obs = {T_obs:.3f}, p-value = {p:.4f}")
    """
    # Преобразуем входные данные в массивы NumPy
    x = np.asarray(x)
    y = np.asarray(y)

    # Вычисляем наблюдаемую разницу метрик
    T_obs: float = metric_func(x) - metric_func(y)

    # Объединяем выборки и задаём параметры перестановок
    pooled = np.concatenate([x, y])
    n_x: int = len(x)

    # Используем генератор случайных чисел для воспроизводимости
    rng = np.random.default_rng(42)

    # Инициализируем массив для хранения статистик перестановок
    T_stats = np.empty(reps)

    # Генерируем перестановки
    for i in tqdm(range(reps)):
        rng.shuffle(pooled)  # Перемешиваем объединённую выборку
        x_ = pooled[:n_x]    # Первая часть после перемешивания
        y_ = pooled[n_x:]    # Вторая часть
        T_stats[i] = metric_func(x_) - metric_func(y_)  # Новая разность метрик

    # Оценка p-value в зависимости от направления альтернативной гипотезы
    if alternative == 'two-sided':
        p: float = np.mean(np.abs(T_stats) >= np.abs(T_obs))
    elif alternative == 'greater':
        p = np.mean(T_stats >= T_obs)
    else:  # 'less'
        p = np.mean(T_stats <= T_obs)

    return p, T_obs, T_stats


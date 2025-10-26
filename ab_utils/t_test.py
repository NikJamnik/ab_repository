import numpy as np
import scipy.stats as st


def onesample_t_stat(x: np.ndarray, mu0: float) -> tuple[float, int, float]:
    """
    Вычисляет t-статистику для одновыборочного t-теста.

    Статистика проверяет нулевую гипотезу H₀: μ = μ₀, где μ₀ — предполагаемое среднее
    генеральной совокупности. Основана на выборке x.

    Args:
        x (np.ndarray): Одномерный массив наблюдений (выборка).
        mu0 (float): Среднее значение по нулевой гипотезе.

    Returns:
        tuple[float, int, float]:
            - t_stat (float): значение t-статистики,
            - df (int): число степеней свободы (n − 1),
            - se (float): стандартная ошибка среднего.
    """
    # Преобразуем данные к NumPy-массиву для корректных операций
    x = np.asarray(x)

    # Количество наблюдений
    n: int = len(x)

    # Среднее значение выборки
    mx: float = x.mean()

    # Несмещённое стандартное отклонение (ddof=1 — делим на n-1)
    sx: float = x.std(ddof = 1)

    # Стандартная ошибка среднего
    se: float = sx / np.sqrt(n)

    # Вычисляем t-статистику по формуле: (x̄ − μ₀) / SE
    t_stat: float = (mx - mu0) / se

    # Число степеней свободы для одновыборочного теста
    df: int = n - 1

    return t_stat, df, se


def onesample_p_value_manual(
    x: np.ndarray, mu0: float, alternative: str = 'two-sided'
) -> tuple[float, float, int]:
    """
    Вычисляет p-значение для одновыборочного t-теста.

    На основе выборки x проверяется гипотеза H₀: μ = μ₀ против одной из альтернатив:
        - 'two-sided': μ ≠ μ₀,
        - 'greater': μ > μ₀,
        - 'less': μ < μ₀.

    Args:
        x (np.ndarray): Одномерный массив наблюдений (выборка).
        mu0 (float): Среднее значение по нулевой гипотезе.
        alternative (str, optional): Тип альтернативной гипотезы.
            По умолчанию 'two-sided'.

    Returns:
        tuple[float, float, int]:
            - p (float): p-значение критерия,
            - t_stat (float): рассчитанная t-статистика,
            - df (int): число степеней свободы (n − 1).
    """
    # Получаем t-статистику, число степеней свободы и стандартную ошибку
    t_stat, df, _ = onesample_t_stat(x, mu0)

    # Вычисляем p-значение в зависимости от направления альтернативы
    if alternative == 'two-sided':
        # Двусторонний тест: учитываем обе стороны распределения
        p: float = 2 * (1 - st.t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        # Правая сторона: вероятность получить t больше наблюдаемого
        p = 1 - st.t.cdf(t_stat, df)
    else:
        # Левая сторона: вероятность получить t меньше наблюдаемого
        p = st.t.cdf(t_stat, df)

    return p, t_stat, df


def onesample_p_value(
    x: np.ndarray, mu0: float, alternative: str = 'two-sided'
) -> tuple[float, float, int]:
    """
    Вычисляет p-значение для одновыборочного t-теста при помощи scipy.stats.ttest_1samp.

    На основе выборки x проверяется гипотеза H₀: μ = μ₀ против одной из альтернатив:
        - 'two-sided': μ ≠ μ₀,
        - 'greater': μ > μ₀,
        - 'less': μ < μ₀.

    Args:
        x (np.ndarray): Одномерный массив наблюдений (выборка).
        mu0 (float): Среднее значение по нулевой гипотезе.
        alternative (str, optional): Тип альтернативной гипотезы.
            По умолчанию 'two-sided'.

    Returns:
        tuple[float, float, int]:
            - p (float): p-значение критерия,
            - t_stat (float): рассчитанная t-статистика,
            - df (int): число степеней свободы (n − 1).
    """
    x = np.asarray(x)
    df = len(x) - 1
    t_stat, p = st.ttest_1samp(x, popmean = mu0, alternative = alternative)
    return p, t_stat, df


def welch_t_stat(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Вычисляет t-статистику для двухвыборочного Welch's t-теста.

    Этот тест используется для сравнения средних двух групп, 
    когда их дисперсии могут отличаться (в отличие от классического t-теста с равными дисперсиями).

    Args:
        x (np.ndarray): Набор наблюдений первой выборки.
        y (np.ndarray): Набор наблюдений второй выборки.

    Returns:
        tuple[float, float, float]:
            - t_stat (float): значение t-статистики;
            - df (float): число степеней свободы, вычисленное по формуле Уэлча-Саттеруэйта;
            - se (float): стандартная ошибка разности средних.
    """
    # Преобразуем входные данные в массивы NumPy
    x = np.asarray(x)
    y = np.asarray(y)

    # Размеры выборок
    nx, ny = len(x), len(y)

    # Средние значения по выборкам
    mx, my = x.mean(), y.mean()

    # Выборочные дисперсии (несмещённые, ddof=1)
    vx, vy = x.var(ddof = 1), y.var(ddof = 1)

    # Стандартная ошибка разности средних
    se: float = np.sqrt(vx / nx + vy / ny)

    # Статистика t по формуле Welch
    t_stat: float = (mx - my) / se

    # Расчёт степеней свободы по формуле Уэлча-Саттеруэйта
    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx**2 / (nx**2 * (nx - 1))) + (vy**2 / (ny**2 * (ny - 1)))
    df: float = df_num / df_den

    return t_stat, df, se


def welch_p_value_manual(
    x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided'
) -> tuple[float, float, float]:
    """
    Вычисляет p-value для двухвыборочного Welch's t-теста.

    Проверяет нулевую гипотезу H₀: μ₁ = μ₂ против одной из альтернатив:
        - 'two-sided' (по умолчанию): μ₁ ≠ μ₂
        - 'greater': μ₁ > μ₂
        - 'less': μ₁ < μ₂

    Args:
        x (np.ndarray): Набор наблюдений первой выборки.
        y (np.ndarray): Набор наблюдений второй выборки.
        alternative (str, optional): Направление альтернативной гипотезы. 
            Возможные значения: 'two-sided', 'greater', 'less'. 
            По умолчанию 'two-sided'.

    Returns:
        tuple[float, float, float]:
            - p (float): p-value критерия;
            - t_stat (float): рассчитанная t-статистика;
            - df (float): число степеней свободы.
    """
    t_stat, df, _ = welch_t_stat(x, y)

    # Расчёт p-value в зависимости от направления теста
    if alternative == 'two-sided':
        p: float = 2 * (1 - st.t.cdf(abs(t_stat), df))
    elif alternative == 'greater':
        p = 1 - st.t.cdf(t_stat, df)
    else:  # 'less'
        p = st.t.cdf(t_stat, df)

    return p, t_stat, df


def welch_p_value(
    x: np.ndarray, y: np.ndarray, alternative: str = 'two-sided'
) -> tuple[float, float, float]:
    """
    Вычисляет p-value для двухвыборочного Welch's t-теста.

    Проверяет нулевую гипотезу H₀: μ₁ = μ₂ против одной из альтернатив:
        - 'two-sided' (по умолчанию): μ₁ ≠ μ₂
        - 'greater': μ₁ > μ₂
        - 'less': μ₁ < μ₂

    Args:
        x (np.ndarray): Набор наблюдений первой выборки.
        y (np.ndarray): Набор наблюдений второй выборки.
        alternative (str, optional): Направление альтернативной гипотезы. 
            Возможные значения: 'two-sided', 'greater', 'less'. 
            По умолчанию 'two-sided'.

    Returns:
        tuple[float, float, float]:
            - p (float): p-value критерия;
            - t_stat (float): рассчитанная t-статистика;
            - df (float): число степеней свободы.
    """
    t_stat, p, df = st.ttest_ind(
        a = x,
        b = y,
        alternative = alternative,
        equal_var = False
    )

    return p, t_stat, df


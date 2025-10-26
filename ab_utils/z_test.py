import numpy as np
import scipy.stats as st
from statsmodels.stats.weightstats import ztest

def ztest_prop_stat(
    x1: int, n1: int, x2: int, n2: int
) -> tuple[float, float, float, float, float]:
    """
    Вычисляет z-статистику для двухвыборочного z-теста пропорций.

    Проверяется нулевая гипотеза H₀: p₁ = p₂,
    где p₁ и p₂ — истинные вероятности успеха в двух группах.

    Args:
        x1 (int): Число успехов в первой группе.
        n1 (int): Общее число испытаний в первой группе.
        x2 (int): Число успехов во второй группе.
        n2 (int): Общее число испытаний во второй группе.

    Returns:
        tuple[float, float, float, float, float]:
            - z_stat (float): значение z-статистики;
            - p1 (float): выборочная пропорция первой группы;
            - p2 (float): выборочная пропорция второй группы;
            - p_pool (float): объединённая (pooled) пропорция;
            - se_pool (float): стандартная ошибка разности пропорций.
    """
    # Выборочные пропорции в каждой группе
    p1: float = x1 / n1
    p2: float = x2 / n2

    # Объединённая (pooled) пропорция — оценка общей вероятности успеха
    p_pool: float = (x1 + x2) / (n1 + n2)

    # Стандартная ошибка объединённой пропорции
    se_pool: float = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    # Z-статистика: разница пропорций, нормированная на стандартную ошибку
    z_stat: float = (p1 - p2) / se_pool

    return z_stat, p1, p2, p_pool, se_pool


def ztest_prop_p_value_manual(
    x1: int, n1: int, x2: int, n2: int, alternative: str = 'two-sided'
) -> tuple[float, float]:
    """
    Вычисляет p-value для двухвыборочного z-теста пропорций вручную.

    На основе z-распределения вычисляется вероятность наблюдать
    значение не менее экстремальное, чем наблюдаемое.

    Args:
        x1 (int): Число успехов в первой группе.
        n1 (int): Общее число испытаний в первой группе.
        x2 (int): Число успехов во второй группе.
        n2 (int): Общее число испытаний во второй группе.
        alternative (str, optional): Тип альтернативной гипотезы:
            - 'two-sided' (по умолчанию) — p₁ ≠ p₂,
            - 'greater' — p₁ > p₂,
            - 'less' — p₁ < p₂.

    Returns:
        tuple[float, float]:
            - p (float): p-value критерия;
            - z_stat (float): рассчитанная z-статистика.
    """
    z_stat, *_ = ztest_prop_stat(x1, n1, x2, n2)

    # Вычисляем p-value в зависимости от направления альтернативы
    if alternative == 'two-sided':
        # Двусторонний тест: учитываем оба хвоста распределения
        p: float = 2 * (1 - st.norm.cdf(abs(z_stat)))
    elif alternative == 'greater':
        # Правосторонний тест: вероятность получить z больше наблюдаемого
        p = 1 - st.norm.cdf(z_stat)
    else:
        # Левосторонний тест: вероятность получить z меньше наблюдаемого
        p = st.norm.cdf(z_stat)

    return p, z_stat


def ztest_prop_p_value(
    x1_arr: np.ndarray, x2_arr: np.ndarray, alternative: str = 'two-sided'
) -> tuple[float, float]:
    """
    Вычисляет z-статистику и p-value для двухвыборочного z-теста пропорций
    с использованием библиотеки SciPy.

    Использует нормальное распределение для оценки значимости разницы
    между двумя пропорциями.

    Args:
        x1_arr (np.ndarray): Массив конверсий первой группы.
        x2_arr (np.ndarray): Массив конверсий второй группы.
        alternative (str, optional): Тип альтернативной гипотезы.
            Возможные значения:
            - 'two-sided' (по умолчанию): p₁ ≠ p₂,
            - 'greater': p₁ > p₂,
            - 'less': p₁ < p₂.

    Returns:
        tuple[float, float]:
            - z_stat (float): z-статистика теста,
            - p_value (float): p-value критерия.
    """
    statsmodels_alternative = 'two-sided' if alternative == 'two-sided' else ('larger' if alternative == 'greater' else 'smaller')

    z_stat, p_value = ztest(
        x1 = x1_arr,
        x2 = x2_arr,
        value = 0,
        alternative = statsmodels_alternative,
        usevar = 'pooled',
        ddof = 0
    )

    return p_value, z_stat


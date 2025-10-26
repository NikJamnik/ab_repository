import scipy.stats as st


def binomial_p_value(k: int, n: int, p0: float, alternative: str = 'two-sided') -> float:
    """
    Вычисляет точное p-значение биномиального теста с помощью scipy.stats.binomtest.

    Функция проверяет простую нулевую гипотезу H₀: p = p₀
    для случайной величины X ~ Binomial(n, p₀).

    Args:
        k (int): Число наблюдённых успехов.
        n (int): Общее количество испытаний.
        p0 (float): Предполагаемая вероятность успеха при H₀.
        alternative (str, optional): Тип альтернативной гипотезы:
            - 'two-sided' (по умолчанию) — двусторонняя проверка  
            - 'less' — H₁: p < p₀  
            - 'greater' — H₁: p > p₀

    Returns:
        float: Точное p-значение биномиального теста.
    """
    # Выполняем точный биномиальный тест с использованием scipy
    res = st.binomtest(k, n, p0, alternative = alternative)
    return res.pvalue


def binomial_p_value_manual(k: int, n: int, p0: float, alternative: str = 'two-sided') -> float:
    """
    Ручной расчёт точного p-значения биномиального теста.

    P-значение вычисляется как сумма вероятностей всех исходов,
    вероятность которых меньше или равна вероятности наблюдаемого
    исхода k при нулевой гипотезе H₀: p = p₀.

    Args:
        k (int): Число наблюдённых успехов.
        n (int): Общее количество испытаний.
        p0 (float): Предполагаемая вероятность успеха при H₀.
        alternative (str, optional): Тип альтернативной гипотезы:
            - 'two-sided' (по умолчанию) — двусторонняя проверка  
            - 'less' — H₁: p < p₀  
            - 'greater' — H₁: p > p₀

    Returns:
        float: Точное p-значение, рассчитанное вручную.
    """
    # Вычисляем вероятность наблюдаемого числа успехов k при H₀
    p_obs: float = st.binom.pmf(k, n, p0)

    # Обработка односторонних гипотез
    if alternative == 'less':
        # Суммируем вероятности всех исходов ≤ k. CDF - Cummulative Distribution Function
        p_value: float = st.binom.cdf(k, n, p0)
    elif alternative == 'greater':
        # Суммируем вероятности всех исходов ≥ k. SF - Survival Function = 1 - Cummulative Distribution Function
        p_value = st.binom.sf(k - 1, n, p0)
    else:
        # Для двусторонней проверки:
        # 1. Вычисляем вероятности для всех возможных исходов. PMF - Probability Mass Function
        probs = st.binom.pmf(range(n + 1), n, p0)
        # 2. Выбираем исходы, вероятность которых ≤ наблюдаемой
        p_value = probs[probs <= p_obs].sum()

    return p_value


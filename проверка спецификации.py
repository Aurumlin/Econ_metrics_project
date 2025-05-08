from scipy import stats
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro
from scipy.stats import boxcox


# 1. Загрузка данных
df = pd.read_csv('cars_clean.csv')

# Очистка mileage от неразрывных пробелов и перевод в число
if 'mileage' in df.columns:
    df['mileage'] = df['mileage'].astype(str).str.replace('\xa0', '', regex=True)
    df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

# 2. Определение переменных
columns_to_use = [
    'mileage', 'power_1', 'power_2', 'model_popularity', 'age', 'generation',
    'is_new', 'is_restyling', 'is_pro', 'is_max', 'is_premium'
]

y = df['price']
X = df[columns_to_use]

# Чистим пропуски
df_model = pd.concat([y, X], axis=1).dropna()
y_clean = df_model['price']
X_clean = df_model.drop(columns=['price']).astype(float)
y_clean = y_clean.astype(float)

# 3. Базовая модель
X_const = sm.add_constant(X_clean)
model_base = sm.OLS(y_clean, X_const).fit()

print('--Базовая модель--')
reset_test_base = linear_reset(model_base, power=2, use_f=True)
print(f'F-статистика RESET: {reset_test_base.fvalue:.4f}')
print(f'p-значение RESET: {reset_test_base.pvalue:.4f}')
if reset_test_base.pvalue < 0.05:
    print("H0 отвергается: Спецификация некорректна.")
else:
    print("H0 не отвергается: Спецификация корректна.")

# 4. F-тест на значимость переменной power_2
X_restricted = X_clean.drop(columns=['power_2'])
X_restricted_const = sm.add_constant(X_restricted)
model_restricted = sm.OLS(y_clean, X_restricted_const).fit()

anova_results = anova_lm(model_restricted, model_base)
print('F-тест на значимость переменной power_2')
print(anova_results)

# 5. Модель с логарифмом цены и квадратами переменных
y_log = np.log(y_clean + 1)

for feature in ['age', 'mileage', 'power_1', 'power_2']:
    if feature in X_clean.columns:
        X_clean[f'{feature}_squared'] = X_clean[feature] ** 2

X_const = sm.add_constant(X_clean)
model_poly = sm.OLS(y_log, X_const).fit()

print('--Модель с логарифмом цены и квадратами признаков--')
reset_test_poly = linear_reset(model_poly, power=2, use_f=True)
print(f'F-статистика RESET: {reset_test_poly.fvalue:.4f}')
print(f'p-значение RESET: {reset_test_poly.pvalue:.4f}')
if reset_test_poly.pvalue < 0.05:
    print("H0 отвергается: Спецификация некорректна.")
else:
    print("H0 отвергается: Спецификация корректна.")

# 6. Модель с логарифмом цены, квадратами и взаимодействиями
interactions = [
    ('power_1', 'age', 'power1_age'),
    ('power_2', 'mileage', 'power2_mileage'),
    ('model_popularity', 'age', 'popularity_age')
]

for var1, var2, interaction_name in interactions:
    if var1 in X_clean.columns and var2 in X_clean.columns:
        X_clean[interaction_name] = X_clean[var1] * X_clean[var2]

X_const = sm.add_constant(X_clean)
model_interactions = sm.OLS(y_log, X_const).fit()

print('--Модель с логарифмом цены, квадратами и взаимодействиями--')
reset_test_interactions = linear_reset(model_interactions, power=2, use_f=True)
print(f'F-статистика RESET: {reset_test_interactions.fvalue:.4f}')
print(f'p-значение RESET: {reset_test_interactions.pvalue:.4f}')
if reset_test_interactions.pvalue < 0.05:
    print("H0 отвергается: Спецификация некорректна.")
else:
    print("H0 не отвергается: Спецификация корректна.")

# 7. F-тест на группу лишних переменных
group_variables = ['power_2', 'generation', 'is_max', 'is_pro']

# Строим ограниченную модель без группы переменных
X_restricted_group = X_clean.drop(columns=group_variables, errors='ignore')
X_restricted_group_const = sm.add_constant(X_restricted_group)
model_restricted_group = sm.OLS(y_clean, X_restricted_group_const).fit()

# Сравниваем модели
anova_group_results = anova_lm(model_restricted_group, model_base)
print('--F-тест на значимость группы переменных:', group_variables, '--')
print(anova_group_results)

# Интерпретация результата
p_value_group = anova_group_results['Pr(>F)'][1]
print(f'\nP-значение F-теста на группу: {p_value_group:.4f}')

alpha = 0.05
if p_value_group < alpha:
    print("H0 отвергается: Группа переменных значима для модели.")
else:
    print("H0 не отвергается: Группа переменных незначима, их можно исключить.")



# 8. Тест Бокса-Кокса
print('--Тест Бокса-Кокса для переменной price--')

# Убираем нули (Box-Cox требует строго положительных значений)
price_positive = y_clean[y_clean > 0]

# Применяем трансформацию Бокса-Кокса
boxcox_transformed, best_lambda = boxcox(price_positive)

print(f'Оптимальное значение λ: {best_lambda:.4f}')

if 0.9 < best_lambda < 1.1:
    print('Логарифмирование не требуется (λ близка к 1).')
else:
    print('Рекомендуется трансформация переменной (например, логарифмирование или степень).')

# 9. Тест Шапиро–Уилка на нормальность остатков

print('--Тест Шапиро–Уилка на нормальность остатков--')

# Берем остатки последней модели
residuals = model_interactions.resid

w_stat, p_value_shapiro = shapiro(residuals)

print(f'W-статистика: {w_stat:.4f}')
print(f'p-значение: {p_value_shapiro:.4f}')

if p_value_shapiro < 0.05:
    print('H0 отвергается: Остатки не нормально распределены.')
else:
    print('H0 не отвергается: Остатки нормально распределены.')

# 9. Тест Шапиро–Уилка на нормальность остатков
print('-- Тест Шапиро–Уилка на нормальность остатков --')

# Берем остатки последней модели
residuals = model_interactions.resid

# Проводим тест
w_stat, p_value_shapiro = shapiro(residuals)

# Выводим результаты
print(f'W-статистика теста: {w_stat:.4f}')
print(f'p-значение теста: {p_value_shapiro:.4f}')

# Интерпретация результата
alpha = 0.05
if p_value_shapiro < alpha:
    print('Вывод: Остатки НЕ нормально распределены (отвергаем H0).')
else:
    print('Вывод: Остатки нормально распределены (не отвергаем H0).')



#10. PE-тест
print('--PE-тест на ненулевое среднее ошибки--')

# Среднее значение остатков
mean_residual = np.mean(residuals)

# Стандартная ошибка среднего
std_residual = np.std(residuals, ddof=1) / np.sqrt(len(residuals))

# T-статистика
t_stat = mean_residual / std_residual

# P-value
p_value_pe = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(residuals)-1))

print(f'T-статистика: {t_stat:.4f}')
print(f'p-значение: {p_value_pe:.4f}')

if p_value_pe < 0.05:
    print('H0 отвергается: Ошибки имеют ненулевое среднее (модель смещена).')
else:
    print('H0 не отвергается: Среднее ошибок близко к нулю (нет систематического смещения).')

final_featues = ['log_age',
               'brand_Geely',
               'is_restyling',
               'color_group_Холодные',
               'log_power_2',
               'city_group_Москва',
               'gearbox_механика',
               'drive_полный',
               'car_class_Crossover']





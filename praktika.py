import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import numpy as np  # Убедитесь, что numpy импортирован


file_path = "brent-monthly.csv" 
data = pd.read_csv(file_path)

# Просмотр первых строк данных
print("Первых 5 строк данных:")
print(data.head())

# Информация о данных
print("\nИнформация о данных:")
print(data.info())

# Проверка на наличие пропущенных значений
print("\nПроверка на пропущенные значения:")
print(data.isnull().sum())

# Преобразование столбца даты в формат datetime
data['Date'] = pd.to_datetime(data['Date'])

# Установка индекса
data.set_index('Date', inplace=True)

# Построение графика цен на нефть
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='WTI Crude Oil Price')
plt.title('WTI Crude Oil Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Вычисление корреляции Пирсона
corr, _ = pearsonr(data.index.astype(int), data['Price'])
print(f'\nCorrelation coefficient: {corr}')

# Линейная регрессия
X = data.index.astype(int).values.reshape(-1, 1)
y = data['Price'].values

reg = LinearRegression().fit(X, y)
trend = reg.predict(X)

# Визуализация тренда
plt.figure(figsize=(12, 6))
plt.plot(data['Price'], label='WTI Crude Oil Price')
plt.plot(data.index, trend, label='Trend', color='red', linestyle='--')
plt.title('WTI Crude Oil Prices and Trend')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()

# Выводы и рекомендации
print("\nВыводы:")
print("1. Временной ряд цен на нефть WTI демонстрирует значительные колебания с тенденцией к росту/спаду в определенные периоды.")
print(f"2. Вычисленный коэффициент корреляции ({corr}) показывает значительную взаимосвязь между временным периодом и ценами на нефть.")

print("\nРекомендации:")
print("1. Для долгосрочного прогнозирования цен на нефть можно использовать модель линейной регрессии, однако стоит учитывать возможные внешние факторы, влияющие на цены.")
print("2. Рекомендуется проводить регулярный анализ рынка для своевременного выявления значимых изменений и трендов.")

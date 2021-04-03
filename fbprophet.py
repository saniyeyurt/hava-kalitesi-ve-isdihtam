#Gerekli Kütüphanelerin yüklenmesi
import pandas as pd
import matplotlib.pyplot as plt
import fbprophet


veri = pd.read_csv('konya.csv')
df = veri[['Tarih','PM10']]
df = df.rename(columns={'Tarih': 'ds', 'PM10': 'y'})
df['ds'] = pd.to_datetime(df['ds'], format='%d %m %Y')
fig = plt.figure(figsize=(40,10))
plt.plot(df['ds'], df['y'],'r')
plt.title('Tarih ve PM10 Değişimi')
plt.ylabel('PM10')
plt.show()
df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.05)
df_prophet.fit(df)
 
#Tahmin süresinin belirlenmesi
t_s=365
df_forecast = df_prophet.make_future_dataframe(periods= t_s, freq='D')
 
#Tahminlerin gerçekleştirilmesi
df_forecast = df_prophet.predict(df_forecast)
 
#Tahmin sonuçlarının görselleştirilmesi
df_prophet.plot(df_forecast, xlabel = 'Tarih', ylabel = 'PM10 Değeri',figsize=(40,10))
plt.title(f'{t_s} günlük PM10 Tahmin')
plt.title('PM10 Değişimi')
plt.ylabel('PM10')
plt.show()

#Tahmin sonuçlarına komponentlerin görselleştirilmesi
df_prophet.plot_components(df_forecast,figsize=(20,25))
plt.show()
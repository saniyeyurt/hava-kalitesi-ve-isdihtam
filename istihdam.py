import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('2019.csv')

is_gucu=df['Istihdam']
ay=df['Ay']


plt.plot(ay, is_gucu)
plt.axis([1, 12, 27700, 29000])
plt.xlabel('Ay')
plt.ylabel('İstihdam')

plt.show()
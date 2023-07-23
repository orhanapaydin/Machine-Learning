import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##---------------------------- VERİ YÜKLEME ----------------------------##
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
plt.scatter(df.derinlik, df.sicaklik)
plt.xlabel("derinlik (m)")
plt.ylabel("sicaklik ("+chr(176)+"C)")
plt.title("Derinlik - Sıcaklık Grafiği")

##---------------------------- HATA FONKSİYONU -------------------------##
def loss_function(a, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].derinlik
        y = points.iloc[i].sicaklik
        total_error += (y - (a*x+b))**2
    MSE = total_error /float(len(points))
    return MSE

##---------------------------- LİNEER REGRESYON FONKSİYONU--------------##
def gradient_descent(a_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)    
    for i in range(n):
        x = points.iloc[i].derinlik
        y = points.iloc[i].sicaklik        
        m_gradient += -(2/n)*x*(y-a_now*x+b_now)
        b_gradient += -(2/n)*(y-a_now*x+b_now)
    a = a_now - m_gradient * L
    b = b_now - b_gradient * L
    total_error = loss_function(a, b, points)
    return a, b, total_error

##---------------------------- EĞİTİM ----------------------------------##
a = 0;         # başlangıç değer, doğrunun eğimi
b = 1;         # başlangıç değer, bias (sapma)
L = 0.0001     # sabit, learning rate
epochs = 1000  # epok sayısı
total_error=[]
for i in range(epochs):    
    a, b, error = gradient_descent(a, b, df, L)
    total_error.append(error)  
MSE=total_error

##---------------------------- ÇİZDİRME --------------------------------##
epochs_list = np.arange(1,epochs+1)
plt.Figure()
plt.scatter(df.derinlik, df.sicaklik, color="black")
plt.plot(list(range(int(df.derinlik.min()), int(df.derinlik.max()+1))), [a*x + b for x in range(int(df.derinlik.min()), int(df.derinlik.max()+1))], color="red")
plt.xlabel("derinlik (m)");plt.ylabel("sicaklik ("+chr(176)+"C)")
plt.legend(["Saha Verisi","Lin_Reg sonucu elde edilen doğru"]);plt.show()
plt.Figure()
plt.plot(epochs_list,MSE);plt.xlabel("Epok");plt.ylabel("Hata - MSE ");plt.show()

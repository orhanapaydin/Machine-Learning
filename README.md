# Makine Öğrenimi


Bu seride Makine Öğrenimi yöntemleri bir akış şeklinde sunulacaktır. Yöntemlere ait konu anlatımları örnek kodlar ile verilerek pekiştirilecektir.

* [1) Lineer Regresyon](#1-lineer-regresyon)


## 1) Lineer Regresyon
İstatistiksel bir yöntem olup, bir veri kümesini temsil eden en uygun doğrunun (aX + b) bulunmasıdır. Örneğin elimizde yer yüzeyinden olan derinlik (metre) ve derinliklerde elde edilen sıcaklıklardan (C) oluşan bir veri kümemiz olsun (Şekil 1). Bu veri kümesini en iyi temsil eden doğrunun hesaplanması veri kümesi içerisinde olmayan bir derinlik değerinde sıcaklığın hesaplanmasına olanak tanır. 

```
import pandas as pd
import matplotlib.pyplot as plt

# Veri Yükleme
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
plt.scatter(df., df.maas)
plt.xlabel("Derinlik")
plt.ylabel("Sıcaklık")
plt.title("Derinlik - Sıcaklık Grafiği")
```


![Şekil 1. Derinliklere karşılık sıcaklık verileri.](https://github.com/orhanapaydin/Machine-Learning/assets/95540971/699b751c-50b4-4e54-ad49-7c030b2a49a8)

Şekil 1. Derinliklere karşılık sıcaklık verileri.           
                
Peki veri kümesini temsil eden en uygun doğru nasıl bulunur? Veri setimizi lineer bir doğru ile temsil etmek istersek Lineer regresyon kullanabiliriz. Basit bir doğru (y' = aX + b) şeklinde temsil edilebilir. Burada "a" doğrunun eğimi, "X" bağımsız değişken,"y'" bağımlı değişken, "b" sapma miktarı olarak tanımlanır. Bu doğru sayesinde istenilen derinlikte (X), sıcaklık değeri (y') hesaplanabilir. Burada hesaplanan y' değeri tahmin değerimizdir. Tahmin edilen değer ile gerçek sıcaklık değeri kıyaslanarak bir hata miktarı hesaplanır (Denklem 1).

$$MSE={1 \over N}{\sum_{i=0}^N (y_i-y'_i)^2}$$                
Denklem 1

Hata miktarını düşürerek a ve b katsayıları hesaplanıp doğru denklemi oluşturulur. Bu katsayıların güncellenmesi epok (ing. epoch) olarak adlandırılan iterasyonlar ile gerçekleştirilir. Örneğin epok = 100, yüz defa katsayıların güncellenmesi anlamına gelir.

**The Cauchy-Schwarz Inequality**

```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```


## Scratch - Train
 def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].derinlik
        y = points.iloc[i].sicaklik
        total_error += (y - (m*x+b))**2
    total_error /float(len(points))
    return total_error

def gradient_descent(m_now, b_now, points, L):

    m_gradient = 0
    b_gradient = 0
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].derinlik
        y = points.iloc[i].sicaklik
        
        m_gradient += -(2/n)*x*(y-m_now*x+b_now)
        b_gradient += -(2/n)*(y-m_now*x+b_now)
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    total_error = loss_function(m, b, points)
    return m, b, total_error

m = 0;
b = 1;
L = 0.001
epochs = 500
total_error=[]
plt.Figure()   
for i in range(epochs):    
    m, b, error = gradient_descent(m, b, df, L)
    total_error.append(error)
    if i%20==0:
        plt.plot(total_error); plt.show()
    
print(m, b)

## PLOT
plt.Figure()
plt.scatter(df.derinlik, df.sicaklik, color="black")
plt.plot(list(range(int(x.min()), int(x.max()+1))), [m*x + b for x in range(int(x.min()), int(x.max()+1))], color="red")
plt.plot(x_test, predict)
plt.show()
plt.plot(total_error)'
    

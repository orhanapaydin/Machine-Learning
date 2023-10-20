<img src="https://github.com/orhanapaydin/Machine-Learning/assets/95540971/37e348fd-1c92-4504-a6e3-48edee8f0808" width=1000 height=250>


# Makine Öğrenimi


Bu seride Makine Öğrenimi yöntemleri bir akış şeklinde sunulacaktır. Yöntemlere ait konu anlatımları örnek kodlar ile verilerek pekiştirilecektir.

[1) Lineer Regresyon](#1-lineer-regresyon)

[2) Çoklu Lineer Regresyon](#2-çoklu-lineer-regresyon)

## 1) Lineer Regresyon
İstatistiksel bir yöntem olup, bir veri kümesini temsil eden en uygun doğrunun (aX + b) bulunmasıdır. Örneğin elimizde yer yüzeyinden olan derinlik (metre) ve derinliklerde elde edilen sıcaklıklardan (C) oluşan bir veri kümesi olsun (Şekil 1). Bu veri kümesini en iyi temsil eden doğrunun hesaplanması, veri kümesi içerisinde olmayan bir derinlik değerinde sıcaklığın hesaplanmasına olanak tanır. 

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Veri Yükleme
df = pd.read_csv("linear_regression_dataset.csv", sep=";")
plt.scatter(df.derinlik, df.sicaklik)
plt.xlabel("derinlik (m)")
plt.ylabel("sicaklik ("+chr(176)+"C)")
plt.title("Derinlik - Sıcaklık Grafiği")
```


![Şekil 1. Derinliklere karşılık sıcaklık verileri.](https://github.com/orhanapaydin/Machine-Learning/assets/95540971/699b751c-50b4-4e54-ad49-7c030b2a49a8)

Şekil 1. Derinliklere karşılık sıcaklık verileri.           
                
Peki veri kümesini temsil eden en uygun doğru nasıl bulunur? Veri setimizi lineer bir doğru ile temsil etmek istersek Lineer regresyon kullanabiliriz. Basit bir doğru (y' = aX + b) şeklinde temsil edilebilir. Burada "a" doğrunun eğimi, "X" bağımsız değişken,"y'" bağımlı değişken, "b" sapma miktarı olarak tanımlanır. Bu doğru sayesinde istenilen derinlikte (X), sıcaklık değeri (y') hesaplanabilir. Burada hesaplanan y' değeri tahmin değerimizdir. Tahmin edilen değer ile gerçek sıcaklık değeri kıyaslanarak bir hata miktarı hesaplanır (Denklem 1).

$$MSE={1 \over N}{\sum_{i=0}^N (y_i-y'_i)^2}$$                
Denklem 1

````
def loss_function(a, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].derinlik
        y = points.iloc[i].sicaklik
        total_error += (y - (a*x+b))**2
    MSE = total_error /float(len(points))
    return MSE
````

Hata miktarını düşürerek a ve b katsayıları hesaplanıp doğru denklemi oluşturulur. Bu katsayıların güncellenmesi epok (ing. epoch) olarak adlandırılan iterasyonlar ile gerçekleştirilir. Örneğin epok = 100, yüz defa katsayıların güncellenmesi anlamına gelir. Hata miktarını düşürerek yeni katsayıların hesaplanması Gradyan İnişi (ing. Gradient Descent) yöntemi ile gerçekleştirilir. Bizim çalışmamızda, Gradyan inişi yöntemi için hata fonksiyonunun "a" ve "b" kaysayılarına göre kısmi türevleri alınır. Sonrasında ise "a" ve "b" katsayısı Denklem 2'deki gibi güncellenir.
$$a = a - {∂E\over ∂a}L$$
$$b = b - {∂E\over ∂b}L$$
Denklem 2

Burada;
L: öğrenme oranı (ing. learning rate, 0<=L<=1)
olarak tanımlanır.

```
## Scratch - Train
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
```
Sonrasında eğitim süreci başlatılır. Örnekte epok=1000 olarak ayarlanmıştır.
```
a = 0;         # başlangıç değer, doğrunun eğimi
b = 1;         # başlangıç değer, bias (sapma)
L = 0.0001     # sabit, learning rate
epochs = 1000  # epok sayısı
total_error=[]
for i in range(epochs):    
    a, b, error = gradient_descent(a, b, df, L)
    total_error.append(error)  
MSE=total_error
```
Her epokta elde edilen MSE değerleri Şekil 3'te gösterilmiştir. Eğitim sonucunda elde edilen "a" ve "b" katsayıları sırası ile 0.52 ve 2.99 olarak bulunmuştur. Katsayılar kullanılarak elde edilen y'= ax + b doğrusu ile veri kümesi Şekil 4'te gösterilmiştir.
```
## PLOT
epochs_list = np.arange(1,epochs+1)
plt.Figure()
plt.scatter(df.derinlik, df.sicaklik, color="black")
plt.plot(list(range(int(df.derinlik.min()), int(df.derinlik.max()+1))), [a*x + b for x in range(int(df.derinlik.min()), int(df.derinlik.max()+1))], color="red")
plt.xlabel("derinlik (m)");plt.ylabel("sicaklik ("+chr(176)+"C)")
plt.legend(["Saha Verisi","Lin_Reg sonucu elde edilen doğru"]);plt.show()
plt.Figure()
plt.plot(epochs_list,MSE);plt.xlabel("Epok");plt.ylabel("Hata - MSE ");plt.show()
```

![Figure 2023-07-21 131510](https://github.com/orhanapaydin/Machine-Learning/assets/95540971/4ff97180-ec68-4987-afcf-f89b4444cf88)

Şekil 3. Eğitim sırasında elde edilen hata miktarları.

![Figure 2023-07-21 131504](https://github.com/orhanapaydin/Machine-Learning/assets/95540971/8fe27846-ed97-4d3b-964a-0d4f2bbd5558)

Şekil 4. Lineer regresyon sonucu elde edilen veri kümesini temsil eden doğru.

Bu örnekte Lineer Regresyon konusu anlatılmış ve bir örnek veri kümesi kullanılarak bu verileri en iyi temsil eden bir doğru bulunmuştur. Artık bu doğru ile istenilen derinlikte tahmini bir sıcaklık değeri hesaplatabiliriz.


## 2) Çoklu Lineer Regresyon

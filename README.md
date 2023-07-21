<img src="https://github.com/orhanapaydin/Machine-Learning/assets/95540971/37e348fd-1c92-4504-a6e3-48edee8f0808" width=1000 height=300>


# Makine Öğrenimi


Bu seride Makine Öğrenimi yöntemleri bir akış şeklinde sunulacaktır. Yöntemlere ait konu anlatımları örnek kodlar ile verilerek pekiştirilecektir.

* [1) Lineer Regresyon](#1-lineer-regresyon)


## 1) Lineer Regresyon
İstatistiksel bir yöntem olup, bir veri kümesini temsil eden en uygun doğrunun (aX + b) bulunmasıdır. Örneğin elimizde yer yüzeyinden olan derinlik (metre) ve derinliklerde elde edilen sıcaklıklardan (C) oluşan bir veri kümesi olsun (Şekil 1). Bu veri kümesini en iyi temsil eden doğrunun hesaplanması veri kümesi içerisinde olmayan bir derinlik değerinde sıcaklığın hesaplanmasına olanak tanır. 

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

Hata miktarını düşürerek a ve b katsayıları hesaplanıp doğru denklemi oluşturulur. Bu katsayıların güncellenmesi epok (ing. epoch) olarak adlandırılan iterasyonlar ile gerçekleştirilir. Örneğin epok = 100, yüz defa katsayıların güncellenmesi anlamına gelir. Hata miktarını düşürerek yeni katsayıların hesaplanması Gradyan İnişi (ing. Gradient Descent) yöntemi ile gerçekleştirilir. Bizim çalışmamızda, Gradyan inişi yöntemi için hata fonksiyonunun "a" ve "b" kaysayılarına göre kısmi türevleri alınır. Sonrasında ise "a" ve "b" kaysatısı Denklem 2'deki gibi güncellenir.
$$a = a - {∂E\over ∂a}L$$
$$b = b - {∂E\over ∂b}L$$
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

a = 0;        # doğunun eğimi
b = 1;        # bias (sapma)
L = 0.001     # learning rate
epochs = 500  # epok sayısı
total_error=[]
plt.Figure()   
for i in range(epochs):    
    a, b, error = gradient_descent(a, b, df, L)
    total_error.append(error)
    if i%20==0:
        plt.plot(total_error); plt.show()    

## PLOT
plt.Figure()
````


plt.scatter(df.derinlik, df.sicaklik, color="black")
plt.plot(list(range(int(x.min()), int(x.max()+1))), [a*x + b for x in range(int(x.min()), int(x.max()+1))], color="red")
plt.plot(x_test, predict)
plt.show()
plt.plot(total_error)
    

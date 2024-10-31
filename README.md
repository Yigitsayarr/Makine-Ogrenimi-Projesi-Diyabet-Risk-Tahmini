# Makine-Öğrenimi-Projesi-Diyabet-Risk-Tahmini
1.Bu proje kapsamında, veri analizi ve makine öğrenimi tekniklerini kullanarak diyabet riskini tahmin etmeyi amaçladım. İşte bu süreçte gerçekleştirdiğim adımlar:

2.Kütüphanelerin İçeri Aktarılması: Projemize gerekli kütüphaneleri içeri aktardık.

3.Keşifsel Veri Analizi: Verimizi yükledikten sonra, veri setinin temel özelliklerini anlamak için info() ve describe() fonksiyonlarını kullandık.

4.Korelasyon Analizi: Verideki değişkenler arasındaki ilişkiyi değerlendirmek için pairplot ve heatmap gibi görselleştirme tekniklerini uyguladık. Böylece, özelliklerin dağılımlarını ve aralarındaki korelasyonu inceledik.

5.Outlier Tespiti: Veri setindeki aykırı değerleri tespit ettik ve bu değerleri veri setinden çıkardık.

6.Veri Setinin Bölünmesi: Veri setimizi eğitim (train) ve test (test) setlerine ayırdık.

7.Standartizasyon: Modelin performansını artırmak için verilerimizi standart hale getirdik.

8.Model Oluşturma ve Eğitim: Farklı makine öğrenimi modelleri oluşturarak, eğitim aşamasını gerçekleştirdik. Ayrıca, tenfold cross validation yöntemi ile modelimizi değerlendirdik.

9.Sonuçların Görselleştirilmesi: Elde ettiğimiz sonuçları görselleştirerek, modellerimizin performanslarını kıyasladık.

10.Model Performans Analizi: Elde ettiğimiz sonuçlar arasında, Decision Tree yöntemimizin en düşük başarıyı elde ettiğini gördük. Bu nedenle, modelin performansını artırmak amacıyla hyperparameter tuning uyguladık ve yaklaşık %1-2’lik bir başarı artışı sağladık.

11.Sonuçların Değerlendirilmesi: Decision Tree modelinin hyperparameter tuning’ini gerçekleştirdikten sonra sonuçlarımızı confusion matrix ve classification report ile değerlendirdik.

12.Gerçek Dünya Uygulaması: Son olarak, yeni bir hasta verisi üzerinde diyabet riskini tahmin ettik ve modelimizin gerçek dünyada nasıl kullanılabileceğini gösterdik.

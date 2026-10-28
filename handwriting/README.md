# Kendi el yazın

Bu klasörde öğrencinin kendi el yazısıyla yazdığı okul numarasının gerçek fotoğrafı `student_id.jpeg` olarak bulunur. Öğrenci gerçek numarayı `23120205009` olarak doğrulamıştır.
Her rakamın arasına küçük boşluk koy; rakamlar birbirine değmesin. Tek satır ve yakın çekim, gölgesiz fotoğraf kullan.
Yeni bir fotoğrafla deneme yaparken yine kendi el yazını kullan.

Komut (eğitim tamamlandıktan sonra):
`python -m src.handwriting --image handwriting/student_id.jpeg --expected 23120205009`

`results/handwriting/digit_*.png` görüntülerinin her birinde yalnızca bir rakam olduğuna dikkat et.
Bozuk ayrıştırma olursa fotoğrafı daha yakın/kırpılmış çek veya `--threshold` değerini değiştir.

Mevcut fotoğrafta 11 rakam ayrılmış; model `13410205527` tahminini üretmiştir. 5/11 rakam doğrudur (%45,45). Gerçek numara yalnızca karşılaştırmada kullanılır. Ayrıntılar ana README ve `results/handwriting/summary.json` dosyasındadır.

# minimalGPTTurk

<img src="assets/tatli_kaplan.jpg" alt="Tatlış Kaplan" width="400" height="250"/>

minimalGPTTurk, Türkçe dil modeli geliştirme eğitimi ve araştırmaları için tasarlanmış minimalist bir kütüphanedir. Kütüphane, dil modeli yapı taşlarını kullanarak yeni modeller oluşturmanıza ve bu modeller üzerinde araştırmalar yapmanıza olanak tanır. modeling.py dosyası, bu amaç doğrultusunda çoklu başlı dikkat mekanizması(multi-head attention), aktivasyon fonksiyonları ve normalizasyon katmanları gibi bileşenleri içerir.

## Dosya Yapısı

- `modeling.py`: Dil modeli oluşturmak için temel yapı taşları
- `train.py`: Model eğitimi için ana script
- `samplers.py`: Farklı şekillerde token üretmeyi sağlayan 'token örnekleyiciler'
- `gpt2.py`: GPT-2 modelinin implementasyonu
- `data/turkce_siirler`: Örnek olması açısından eğitim için Türkçe bir veri setini nasıl işleyebileceğinizi gösteren notebooklar
- `inference.ipynb`: Eğitilen modeli nasıl kullanabileceğinizi gösteren notebook
- `tests`: Test dosyaları

## Katkıda Bulunma

Bu projeye katkıda bulunmak herkes için serbest. Aşağıdaki TODO listesinden(önem sırasına göre) bir görev seçebilir veya kendi önerilerinizi sunabilirsiniz.

## TODO Listesi

- [ ] MultiQueryAttention implementasyonu
- [ ] GroupedQueryAttention implementasyonu
- [ ] LLaMa modeli implementasyonu
- [ ] Mixtral of Experts modeli implementasyonu
- [ ] Beam Search Sampler implementasyonu
- [ ] Çoklu GPU desteği ekleme
- [ ] Daha fazla Türkçe veri seti entegrasyonu
- [ ] Orta ölçekte bir model eğitmek

# adviSU — Demo Senaryosu

Bu doküman `docs/demo_script.md`'nin güncellenmiş halidir; önceki sürüm projenin erken bir
aşamasına aitti (GPA yok, transkript aktarımı yok, retrieval mode seçici referans veriyordu).
Aşağıdaki adımlar şu an gerçekten çalışan özellikleri, sırasıyla gösterir. Her adımda ekranda ne
olması gerektiği ve neden önemli olduğu belirtilmiştir. Hedef süre: 8–10 dakika.

## 0. Demodan önce

```bash
docker compose up -d
```

Kontrol listesi:
- [ ] `docker compose ps` → `backend` ve `mongo` **healthy**
- [ ] Frontend: `npm --prefix frontend run dev -- --host` (veya Docker'daki frontend servisi)
- [ ] `.env` içinde gerçek `GROQ_API_KEY` var (placeholder `replace_me` değil)
- [ ] Tarayıcı zoom ~100%, pencere genişçe (masaüstü tasarım için)
- [ ] Elinizde gerçek bir Sabancı transkript PDF'i olsun (transkript aktarımı adımı için)

---

## 1. Giriş ve genel izlenim (~30 sn)

Giriş ekranını aç. Sol panelde marka logosu ve "Ne kaldığını tam olarak gör" başlığı, sağda
giriş formu ve demo hesapları (öğrenci/admin) kutulanmış şekilde görünür.

> "adviSU, Sabancı Üniversitesi'nin resmî müfredat verisine dayanan bir akademik danışmanlık
> sistemi. Kritik olan: akademik hesaplamalar — kredi sayımı, ön koşullar, program uygunluğu —
> modelin değil, kodun sorumluluğunda. Model yalnızca doğrulanmış sonuçları açıklıyor."

`student` / `student` ile giriş yap.

## 2. Transkript aktarımı — otomatik ders + not girişi (~90 sn)

**Ders Geçmişi** sayfasına git. "Transkriptten aktar" kutusuna gerçek bir transkript PDF'i
sürükle.

Ekranda göster:
- Yükleme anında (bir tuşa basmadan) tamamlanan dersler listeye düşüyor, her birinin notu
  otomatik dolduruluyor.
- Üstte gerçek GPA anında hesaplanıp gösteriliyor.
- Her dersin yanında not değiştirme (2 karakterlik dar seçici) ve tekil silme butonu var; başlıkta
  "Tümünü temizle" ile toplu sıfırlama da mevcut.

> "Bu, ekte gönderdiğim gerçek transkript üzerinde defalarca test edilmiş, tamamen deterministik
> bir PDF parser — LLM kullanmıyor, regex tabanlı. 55/55 ders satırı, GPA'ya kadar birebir
> eşleşiyor."

## 3. Deterministik mezuniyet denetimi (~60 sn)

**Profil** sayfasına git, "Denetimi Çalıştır"a bas.

> "Bu tablo modelin ürettiği bir şey değil. `degree_audit.py`, tam olarak bu öğrencinin programı
> ve giriş dönemi için resmî gereksinim dosyasını okuyor, ders geçmişiyle birleştirip kategori
> bazında kredi dağılımını (University Courses, Core Electives, Required, Area/Free Electives) ve
> ECTS ayrımını kodda hesaplıyor. Gerçek bir öğrencinin resmî 'Degree Evaluation' çıktısıyla
> birebir doğrulanmış durumda."

## 4. Sohbet: gerçek, kanıtlı bir cevap (~60 sn)

Sohbete dön, sor:

> `Is CS 306 required or core based on my admit term?`

Göster:
- Cevap, öğrencinin **her admit term için** gerçek müfredat satırlarını (`degree_requirements/CS/202xxx.jsonl`) kaynak göstererek açıklıyor.
- Kaynak listesi gerçek dosya/chunk kimlikleri.

Aynı soruyu Türkçe sor:

> `CS 210 dersinin ön koşulu nedir?`

> "Cevap dili soru diliyle eşleşiyor — arayüz dili değil, sorunun dili belirleyici. Burada da
> önemli bir nokta var: sistem CS 210 için ön koşul bilgisi bulamadığında bunu açıkça söylüyor,
> uydurmuyor — sadece gerçekten bildiği kısmı (hangi havuzda yer aldığını) paylaşıyor."

## 5. Sohbet: GPA senaryoları (~90 sn)

> `Bu dönem GPA'mı ne kadar yükseltebilirim?`

Tablo formatında birkaç not senaryosu (Hepsinden A, A-, B+, B) ve olası yeni GPA'lar görünür.

Sonra spesifik bir soru:

> `CS 455'ten B+ alsam, ENS 211'den C alsam GPA'm kaça çıkar?`

> "Bu tamamen deterministik: her dersin gerçek SU kredisini katalogdan çekip kredi ağırlıklı GPA
> formülünü kodda hesaplıyor. Zaten tamamlanmış bir dersin notu değiştiriliyorsa, eski katkısı
> toplamdan çıkarılıp yenisi ekleniyor — üst üste binmiyor."

## 6. Ders programı oluşturma (~60 sn)

**Ders Programı** sayfasına git. Birkaç ders/section seç, çakışmasız bir program oluştur.

> "Planlayıcı, CRN çakışmalarını koddan kontrol ediyor; öneri motoru da tek bir dönemde en az 15
> SU önerecek şekilde sınırlandırılmış — öneri 15'in altında kalırsa sistem yeniden üretiyor."

CRN'leri kopyala butonunu göster, Excel'e aktar.

## 7. Güvenlik: bilmediğinde uydurmuyor (~45 sn)

Sohbette, sistemin kanıt bulamayacağı bir soru sor (örn. `CS 999 dersi ne zaman veriliyor?` gibi
kataloğa hiç girmeyen bir kod).

> "Kanıt yoksa sistem 'bu yanıtı doğrulayamıyorum' diyor, tahmin yürütmüyor. Bu, hem içerik
> güvenliği katmanından hem de yanıt üretildikten sonra çalışan bir doğrulama katmanından geçiyor:
> model gerçek kaynaklardan alıntı yapmış olsa bile, alıntıladığı içerik iddiasını gerçekten
> destekliyor mu diye ayrıca kontrol ediliyor — sahte ama gerçek görünen bir alıntıyla yanlış bir
> iddiayı geçirmeye çalışsa bile bu son katman yakalıyor."

## 8. Kapanış (~20 sn)

> "Özetle: akademik gerçekler kodda hesaplanıyor, model yalnızca açıklıyor; kritik veri eksikse
> sistem açıkça çekiniyor; ve öğrencinin kendi transkriptinden GPA'sına kadar her şey tek bir
> tutarlı sistemde birleşiyor."

---

## Kamerada göstermekten kaçının

- Yoğun art arda sorgu atmak: Groq API'nin dakikalık rate limit'i var; art arda çok fazla istek
  atarsanız birkaç dakikalığına "geçici olarak sınırlandı" mesajı alırsınız (bu bir hata değil,
  gerçek sağlayıcı sınırı — sakin bir tempo ile sorun yaşanmaz).
- Admin hesabının şifresi hâlâ zayıf bir demo değeri (`.env`'de `ADMIN_PASSWORD`); canlıya
  çıkmadan önce değiştirilmeli.

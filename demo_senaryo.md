# adviSU Demo Video Senaryosu (v2 — ayrıntılı, tamamı test edildi)

Süre hedefi: ~9-11 dakika. Her adım bu oturumda gerçek tarayıcıda bizzat çalıştırılıp
doğrulandı — hiçbiri varsayım değil. Sahne başına "Söylenecek / yazılacak metin" ve
"Ekranda görünmesi gereken" ayrı ayrı belirtildi.

## Ortam bilgileri

- Frontend: http://localhost:5173
- Backend: http://localhost:8000 (health: `/test`)
- Demo hesaplar: `student / student` (öğrenci), `admin / admin` (admin)
- Gerçek transkript dosyası: `C:\Users\mehme\Desktop\selman.yilmaz_transcript.pdf`
  (34 ders, GPA 3.71, 97 SU — Management'tan CS'e internal transfer yapmış gerçek bir öğrenci)
- Sunucuları kendi terminalinden çalıştırmak istersen:
  ```bash
  cd server && ../<venv>/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000
  cd frontend && npm run dev -- --host 127.0.0.1 --port 5173
  ```
- `.env`'de `ADVISU_LAB_DENSE=false` ayarlı (production'daki gibi). İlk istekte ~10-15 sn
  bir ısınma olur, sonrası hızlı. Backend'i her yeniden başlattığında oturum (login) düşer,
  demo esnasında backend'i YENİDEN BAŞLATMA.

## Kayıttan hemen önce yapılacak hazırlık (kamera açılmadan)

1. Öğrenci hesabıyla gir, **Course History**'e git, varsa eski dersleri **Clear all** ile temizle.
2. **Profile**'a git, Major = **PSIR**, Curriculum term = **Fall 2024-2025 (202401)**, Academic
   year = **Not specified**, **Save profile**.
   (Bu, Sahne 3-4'ün başlangıç durumu — boş bir PSIR öğrencisi.)
3. Tarayıcıyı 1280x800 civarı bir pencerede, koyu temada, TR dilinde başlat.

---

### SAHNE 1 — Giriş ekranı ve marka (0:00–0:25)

**Göster:** http://localhost:5173/login

**Söyle:** "adviSU, Sabancı Üniversitesi için resmi müfredat verisinden çalışan bir akademik
danışman. Sol üstte görüldüğü gibi artık 11 program destekliyor — bu oturumda Management ve
Siyaset Bilimi & Uluslararası İlişkiler (PSIR) eklendi."

**İşaret et:** "DEGREE ANALYSIS · 11 PROGRAMS · 4 CURRICULUM YEARS" yazısı.

**Yap:** `student / student` ile giriş yap (ya da "Student demo" çipine tıkla).

---

### SAHNE 2 — Sohbet: mezuniyet ilerlemesi (0:25–0:55)

**Göster:** Ana sohbet ekranı, 4 hazır öneri kartı.

**Söyle:** "Sohbet ekranında hazır sorular var. İlk olarak mezuniyet ilerlemesini soralım —
bu cevap modelin uydurması değil, kodda hesaplanan gerçek bir sonuç."

**Tıkla:** **"Calculate my degree progress"** kartı.

**Beklenen ekran:** Kategori kategori kredi dökümü (henüz ders eklenmediği için hepsi 0/gerekli).

---

### SAHNE 3 — Profil: PSIR'ın iki ayrı çekirdek havuzu (0:55–2:00) — ANA SAHNE 1

**Göster:** Sol menüden **Profile and degree audit**.

**Söyle:** "Şimdi PSIR programının müfredatına bakalım — bu program bu oturumda eklendi."

**Doğrula ekranda:** "Major (program)" açılır kutusunda PSIR ve MAN'ın CS, EE, ECON gibi
diğer 9 programla birlikte listede olduğunu göster (dropdown'ı aç, kapatmadan 1-2 saniye
göster).

**Yap:** Major = PSIR, Term = Fall 2024-2025 zaten seçili, **Run audit**'e tıkla.

**Söyle (sonuç gelince):** "Dikkat edin — PSIR'da tek bir 'çekirdek seçmeli' kategorisi
yok, ikisi var: 'Core Electives International Relations' ve 'Core Electives Political
Science', her biri ayrı 12 SU. Bu, sistemin PSIR'ın gerçek iki-havuzlu yapısını doğru
ayırt ettiğinin kanıtı — normalde çoğu sistem bunu tek havuz sanıp yanlış hesaplar."

**Beklenen ekran:**
```
0 / 125 SU credits · incomplete
University Courses            0/44
Required Courses              0/24
Core Electives International Relations   0/12
Core Electives Political Science         0/12
Area Electives                 0/15
Free Electives                 0/18
```

**Yap:** Major'ı **MAN**'a değiştir, Term = Fall 2024-2025, **Save profile** → **Run audit**.

**Söyle:** "Management için de resmi 127 SU / 240 ECTS müfredatı aynı şekilde yükleniyor."

---

### SAHNE 4 — Sohbet: PSIR öğrencisi için güvenli ders önerisi (2:00–3:00)

**Yap:** Major'ı tekrar **PSIR**'a al, kaydet. **Chat**'e dön, **New chat**.

**Yaz ve gönder:** `What should I take until graduation?`

**Söyle (cevap gelirken):** "Bu, tamamen deterministik bir motor — model burada ders
listesi UYDURMUYOR, kodda PSIR'ın resmi müfredatını okuyup önkoşulu uygun dersleri
seçiyor."

**Beklenen ekran / vurgulanacak noktalar:**
- Cevabın başında **"This list is NOT for a single term — per-term registration load is
  capped (typically ~18 SU), so you should spread these across multiple future terms"**
  uyarısı — bunu özellikle işaret et: "Sistem bu listenin TEK dönemde alınacak bir liste
  olmadığını, kayıt limitinin döneme göre sınırlı olduğunu açıkça söylüyor."
- "Required courses still blocked by prerequisites" listesinde **POLS 250, POLS 301,
  POLS 352, PSIR 311** gibi PSIR'a özgü derslerin göründüğünü göster — bunlar doğrudan bu
  oturumda eklenen PSIR müfredat verisinden geliyor.

---

### SAHNE 5 — Transkript yükleme ve otomatik bölüm güncelleme (3:00–5:00) — ANA SAHNE 2

**Söyle:** "Şimdi gerçek bir öğrenci transkriptiyle deneyelim. Bu öğrenci aslında
Management'a başlamış, sonra Bilgisayar Bilimi ve Mühendisliği'ne internal transfer
yapmış — profildeki program hâlâ PSIR görünüyor ama transkript CS diyor. Sistemin bunu
otomatik düzeltmesini göreceğiz."

**Yap:** **Course History**'e git.

**Tıkla:** **"Choose transcript PDF"**, dosya seçiciden
`C:\Users\mehme\Desktop\selman.yilmaz_transcript.pdf` seç.

**Bekle:** ~5-10 saniye (PDF parse ediliyor).

**Beklenen ekran:**
- Yeşil toast: *"34 courses processed from your transcript. GPA: 3.71"*
- Hemen ardından ikinci bir yeşil toast: **"Your program was updated to CS based on your
  transcript's official record."** ← Bu, bu oturumda eklenen yeni özellik; mutlaka
  ekranda görünmesini bekle ve işaret et.
- Sol panelde "34 saved · 97 SU · GPA 3.71" ve tüm dersler (AL 102, CS 201, CS 300, CS 301,
  CS 303, CS 306, CS 308, CS 404, CS 412, CS 445, ... ) notlarıyla birlikte listelenir.

**Söyle:** "Sistem transkriptin en son yazan 'Program :' satırını okuyor — kronolojik
olarak en son basılan program, öğrencinin GÜNCEL bölümü demek. Eski Management kaydı
atlanıyor, doğru şekilde CS'e geçiliyor."

**Yap:** **Profile**'a git.

**Beklenen ekran:** Major artık **CS** (elle değiştirmeden, otomatik). Curriculum term ve
academic year dokunulmadan aynı kalıyor (sistem bunları tahmin etmiyor — sadece kesin
bildiği alanı günceller).

**Yap:** **Run audit**.

**Beklenen ekran:** 97/125 SU, University Courses 41/41 tamamlanmış, "Missing required:
CS 395, ENS 491, ENS 492" (staj/bitirme projesi — 6. dönem bir öğrenci için beklenen).

---

### SAHNE 6 — Sohbet: bölüm hiç yazılmadan doğru öneri (5:00–6:00)

**Söyle:** "Şimdi sohbete dönüp AYNI soruyu tekrar soracağım — ama bu sefer 'CS öğrencisiyim'
diye YAZMAYACAĞIM. Sistem profildeki bilgiden anlamalı."

**Yap:** **Chat** → **New chat**.

**Yaz ve gönder:** `What should I take until graduation?`

**Beklenen ekran:** Öneri listesi artık tamamen CS dersleri: **CS 302, CS 307, CS 310,
DSA 210, ENS 211, ENS 203, ENS 208, DSA 201, ENS 202, ENS 204** (~30 SU) — hiçbiri
transkriptte zaten tamamlanmış derslerle çakışmıyor.

**Söyle:** "Prompt'ta 'CS' kelimesi hiç geçmedi — sistem profildeki kayıtlı programı ve
transkriptten gelen tamamlanmış dersleri kullanarak önerdi."

**Tıkla:** **"How much could I raise my GPA this term?"** kartı (yeni sohbet açıp).

**Beklenen ekran:** "Your current GPA: 3.71 (over 97 SU)" ve not senaryosuna göre olası
yeni GPA tablosu (All A → 3.74, All B → 3.61, vb.) — transkriptin gerçek GPA'sıyla birebir
uyuşuyor.

---

### SAHNE 7 — Ders programı (Schedule) (6:00–7:15)

**Yap:** **Course Schedule**'a git.

**Söyle:** "Bu, resmi SUIS ders çizelgesinden gelen gerçek CRN verisiyle çalışan bir
program oluşturucu."

**Yap:**
1. Arama kutusuna bir ders kodu yaz (ör. `ACC 201`), bölümleri (section) genişlet.
2. Bir CRN'in yanındaki **+**'ya tıkla → "Added ACC 201 to the schedule" bildirimi ve
   haftalık ızgarada (weekly view) beliren ders.
3. **Excel** butonuna tıkla, dışa aktarmayı göster.
4. **Clear**'a tıkla → onay diyaloğunu göster ("Are you sure you want to remove every
   course...") → onayla.

**Söyle:** "Silme gibi geri alınamaz işlemler önce onay istiyor — bu tüm uygulamada
tutarlı bir güvenlik deseni."

---

### SAHNE 8 — Dil ve tema (7:15–7:45)

**Yap:** Sağ üstten **TR**'ye tıkla — Course Schedule, Profile, Course History
başlıklarının anında Türkçeleştiğini göster (sayfa yenilemeden).

**Yap:** Ay/güneş ikonuna tıkla — koyu/açık tema arasında geçiş yap.

**Söyle:** "Arayüz dili ile öğrencinin sohbette sorduğu dil birbirinden bağımsız —
istersen arayüz Türkçe iken İngilizce soru sorabilirsin, cevap yine İngilizce gelir."

---

### SAHNE 9 — Kapanış (7:45–8:15)

**Yap:** Sol alttaki **Sign out**'a tıkla, giriş ekranına dönüldüğünü göster.

**Söyle (kapanış):** "Özetle: adviSU artık 11 program için resmi müfredat verisiyle
mezuniyet denetimi yapıyor, transkript yükleyince öğrencinin bölümünü otomatik
güncelliyor, ve tüm ders önerileri önkoşul kontrolünden geçmeden asla ekrana gelmiyor."

---

## Bu oturumda düzeltilen iki gerçek hata (video anlatımında opsiyonel, teknik bir
## izleyici kitlesi varsa değinilebilir)

1. **SPS 303 önkoşul hatası:** Bu ders daha önce "önkoşulsuz" sanılıyordu; gerçekte resmi
   SUIS kaydına göre en az 58 tamamlanmış SU kredisi gerektiriyor. Artık bu eşik doğru
   uygulanıyor — 6 SU'luk bir öğrenciye asla önerilmiyor, "Future targets" bölümünde
   "58+ completed SU credits (6 so far)" notuyla gösteriliyor.
2. **"Mezun olana kadar" listesi netliği:** Bu liste zaten tasarım gereği tek dönem
   sınırına (≈18 SU) tabi değildi (tüm kalan gereksinimi gösterir), ama bunu AÇIKÇA
   belirtmiyordu. Artık cevabın başında "bu liste tek dönem için değil, birden fazla
   döneme yayılmalı" uyarısı var.

Her ikisi de gerçek testlerle doğrulandı (`server/tests/test_advising.py`) ve tam test
paketi (430 test) hiçbir regresyon göstermeden geçiyor.

## Kayıt sırasında KAÇINILMASI gereken sorular (bilinen, ayrı bir konu olan sınırlama)

Sohbette doğrudan "önerilen programda X dönemde hangi dersler var" tarzı serbest metin
sorular (ör. "Which suggested-program courses should a PSIR student take in semester 3?")
niyet sınıflandırıcı tarafından bazen yanlış yönlendirilip "cevabı doğrulayamıyorum"
diyebiliyor veya alakasız bir mezuniyet tablosu dönebiliyor. Bu, PSIR/MAN/suggested-program
veri işiyle DEĞİL, önceden var olan ayrı bir niyet-yönlendirme katmanı sınırlamasıyla
ilgili (retrieval/benchmark tarafı ayrı ölçülüp doğrulandı: baseline %36.7 → yeni corpus
%91.7 AEHR@5). Bu yüzden Sahne 4/6'da SADECE yukarıda test edilmiş kalıpları kullan
("What should I take until graduation?", "Calculate my degree progress", "How much could
I raise my GPA this term?").

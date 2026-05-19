import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. 7 Ana Intent İçin Sentetik Eğitim Verisi
data = {
    "query": [
        # 1. Mezuniyet Durumu Sorgula
        "Mezuniyetime kaç kredi kaldı?", "Mezun oluyor muyum?", "Hangi dersleri vermem gerekiyor mezuniyet için?", "Required ve core eksiklerimi hesapla", "Kaç SU kredim eksik?", "Ne zaman mezun olurum?",

        # 2. Ders Önerisi
        "Önümüzdeki dönem için ders önerir misin?", "NLP alanında hangi dersleri almalıyım?", "Bana kolay 5 tane ders programı yap", "Data science dersleri listele", "Ders programı oluşturmak istiyorum",

        # 3. Derse Çalışma Planı Önerisi
        "Data structures dersine nasıl çalışmalıyım?", "Machine learning dersini geçmek için tavsiyeler", "CS300 için çalışma planı yap", "Bu dersin sınavlarına nasıl hazırlanılır?", "Dersi A ile geçmek için ne yapmalıyım?",

        # 4. Major Seçme Önerisi
        "Hangi bölümü seçmeliyim?", "CS mi yoksa Endüstri mi yazmalıyım?", "Bilgisayar bilimleri seçmek mantıklı mı?", "Major seçimi konusunda kararsızım yardım et", "Hangi ana dal bana uygun?",

        # 5. Major'da Alanda Özelleşme Önerisi
        "CS seçtim ama NLP mi yoksa Security mi yönelmeliyim?", "Bilgisayar bilimlerinde data alanında uzmanlaşmak", "Yapay zeka alanında özelleşmek için ne yapmalıyım?", "Hangi alt dalı seçmeliyim?",

        # 6. Ders Ayrıntısı İsteği
        "CS 412 dersini kim veriyor?", "Yücel Saygın'ın dersi zor mu?", "Bu dersin syllabus'ı nedir?", "Bu dersi neden almalıyım, projeleri ağır mı?", "Hocanın notlandırması nasıl?", "Dersin içeriği ne?",

        # 7. Diğer (Fallback)
        "Merhaba nasılsın?", "Okulun yemekhanesi nerede?", "Şifremi unuttum", "Teşekkür ederim", "Bugün hava nasıl?", "Selam", "Sistemi kim yaptı?"
    ],
    "intent": [
        "mezuniyet_durumu", "mezuniyet_durumu", "mezuniyet_durumu", "mezuniyet_durumu", "mezuniyet_durumu", "mezuniyet_durumu",
        "ders_onerisi", "ders_onerisi", "ders_onerisi", "ders_onerisi", "ders_onerisi",
        "calisma_plani", "calisma_plani", "calisma_plani", "calisma_plani", "calisma_plani",
        "major_secimi", "major_secimi", "major_secimi", "major_secimi", "major_secimi",
        "alanda_ozellesme", "alanda_ozellesme", "alanda_ozellesme", "alanda_ozellesme",
        "ders_ayrintisi", "ders_ayrintisi", "ders_ayrintisi", "ders_ayrintisi", "ders_ayrintisi", "ders_ayrintisi",
        "diger", "diger", "diger", "diger", "diger", "diger", "diger"
    ]
}

class IntentDetector:
    def __init__(self, model_path="intent_model.pkl"):
        self.model_path = model_path
        self.pipeline = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            df = pd.DataFrame(data)
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
            ])
            self.pipeline.fit(df['query'], df['intent'])
            joblib.dump(self.pipeline, self.model_path)

    def predict(self, query, threshold=0.3):
        probs = self.pipeline.predict_proba([query])[0]
        max_prob = max(probs)
        if max_prob < threshold:
            return "diger"
        return self.pipeline.classes_[probs.argmax()]

# Singleton instance
detector = IntentDetector()

def get_intent(query: str) -> str:
    return detector.predict(query)

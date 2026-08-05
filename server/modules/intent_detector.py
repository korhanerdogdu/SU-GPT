from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import json
import os
from pathlib import Path

from modules import intents


DEFAULT_MODEL_PATH = Path(
    os.getenv(
        "INTENT_MODEL_PATH",
        Path(__file__).resolve().parents[1] / "chroma_store" / "intent_model.pkl",
    )
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_SELECTION_PATH = Path(
    os.getenv(
        "INTENT_MODEL_SELECTION_PATH",
        PROJECT_ROOT / "data" / "benchmark" / "intent_model_selection.json",
    )
)
BERT_CLASSIFIER_PATH = Path(
    os.getenv(
        "INTENT_BERT_CLASSIFIER_PATH",
        Path(__file__).resolve().parents[1] / "models" / "intent_bert_classifier.joblib",
    )
)

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
    def __init__(self, model_path=None):
        self.model_path = Path(model_path or DEFAULT_MODEL_PATH)
        self.pipeline = None
        self._load_or_train_model()

    def _load_or_train_model(self):
        if self.model_path.exists():
            self.pipeline = joblib.load(self.model_path)
        else:
            self.pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
                ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
            ])
            self.pipeline.fit(data['query'], data['intent'])
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, self.model_path)

    def predict_with_confidence(self, query, threshold=0.3):
        probs = self.pipeline.predict_proba([query])[0]
        max_prob = max(probs)
        predicted = self.pipeline.classes_[probs.argmax()]
        if max_prob < threshold:
            return "diger", float(max_prob)
        return predicted, float(max_prob)

    def predict(self, query, threshold=0.3):
        intent, _confidence = self.predict_with_confidence(query, threshold=threshold)
        return intent

class BenchmarkSelectedIntentDetector:
    """Use BERT only when the committed, reproducible benchmark selected it."""

    def __init__(self):
        self.baseline = IntentDetector()
        self._bert_encoder = None
        self._bert_bundle = None
        self.bert_enabled = self._selection_approved()

    def _selection_approved(self) -> bool:
        if not MODEL_SELECTION_PATH.exists() or not BERT_CLASSIFIER_PATH.exists():
            return False
        try:
            result = json.loads(MODEL_SELECTION_PATH.read_text(encoding="utf-8"))
            return bool(
                result.get("winner") == "bert"
                and result.get("bert_selected")
                and float(result["bert"]["macro_f1"]) > float(result["tfidf"]["macro_f1"])
            )
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def _load_bert(self):
        if self._bert_bundle is None:
            from sentence_transformers import SentenceTransformer

            self._bert_bundle = joblib.load(BERT_CLASSIFIER_PATH)
            self._bert_encoder = SentenceTransformer(self._bert_bundle["model_name"])
        return self._bert_encoder, self._bert_bundle["classifier"]

    def predict_with_confidence(self, query, threshold=0.3):
        if not self.bert_enabled:
            return self.baseline.predict_with_confidence(query, threshold=threshold)
        try:
            encoder, classifier = self._load_bert()
            embedding = encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            probabilities = classifier.predict_proba(embedding)[0]
            probability = float(max(probabilities))
            predicted = classifier.classes_[probabilities.argmax()]
            if probability < threshold:
                return "diger", probability
            return str(predicted), probability
        except Exception:
            # A missing model cache or optional dependency must never take down chat routing.
            return self.baseline.predict_with_confidence(query, threshold=threshold)

    def predict(self, query, threshold=0.3):
        return self.predict_with_confidence(query, threshold=threshold)[0]


# Singleton instance. BERT weights remain lazy until the first routed question.
detector = BenchmarkSelectedIntentDetector()


def get_raw_intent_with_confidence(query: str) -> tuple[str, float]:
    """The classifier's own label, untranslated.

    Only evaluation code should need this: the committed benchmark rows and the trained
    classifier both speak the historical Turkish vocabulary, and rewriting either would
    invalidate recorded measurements. Application code wants `get_intent` instead.
    """
    return detector.predict_with_confidence(query)


def get_intent(query: str) -> str:
    """Canonical English intent (see modules.intents)."""
    return intents.to_canonical(detector.predict(query))


def get_intent_with_confidence(query: str) -> tuple[str, float]:
    """Canonical English intent plus the classifier's confidence."""
    raw, confidence = detector.predict_with_confidence(query)
    return intents.to_canonical(raw), confidence

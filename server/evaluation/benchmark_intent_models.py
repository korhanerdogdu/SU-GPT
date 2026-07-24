from __future__ import annotations

"""Compare the existing TF-IDF intent classifier with multilingual BERT embeddings.

The same stratified folds are used for both candidates. The script writes its measured
macro-F1 decision and, only when BERT wins, the fitted lightweight classifier. Transformer
weights remain in the normal Hugging Face cache rather than being committed to the repository.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "data" / "benchmark" / "intent_model_selection.json"
CLASSIFIER_PATH = ROOT / "server" / "models" / "intent_bert_classifier.joblib"
BERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


SAMPLES: dict[str, list[str]] = {
    "mezuniyet_durumu": [
        "Mezuniyet için kaç kredim kaldı?",
        "Zorunlu derslerimden hangileri eksik?",
        "Degree audit sonucumu açıklar mısın?",
        "Mezun olabilmek için daha ne almam lazım?",
        "How many credits do I need to graduate?",
        "Which graduation requirements have I not completed?",
        "Kalan core ve area elective koşullarımı hesapla",
        "Programımı bitirmeye hazır mıyım?",
        "What is missing from my degree evaluation?",
        "Mezuniyet kategorilerinde durumum nedir?",
        "Toplam SU kredim yeterli mi?",
        "Hangi derslerim mezuniyete sayıldı?",
    ],
    "ders_onerisi": [
        "Gelecek dönem hangi dersleri almalıyım?",
        "NLP alanında beş ders öner",
        "Bana çakışmasız bir ders programı hazırla",
        "Data science için seçmeli önerir misin?",
        "Recommend courses for next semester",
        "I want an AI focused schedule",
        "Kolay bir dönem için ders seçelim",
        "Security alanında ilerlemek istiyorum, ne alayım?",
        "Which electives fit web development?",
        "Ders yükümü ağırlaştırmak istiyorum",
        "Systems alanı için program yap",
        "Bu dönem alabileceğim dersleri sırala",
    ],
    "calisma_plani": [
        "CS 300'e nasıl çalışmalıyım?",
        "Finale hazırlanmak için çalışma planı yap",
        "Bu dersten A almak için ne yapmalıyım?",
        "Machine learning sınavına nasıl hazırlanılır?",
        "Create a study plan for data structures",
        "How should I prepare for the midterm?",
        "Haftalık ders çalışma takvimi çıkar",
        "Konuları yetiştiremiyorum nasıl çalışayım?",
        "Give me study advice for CS 201",
        "Projeye ve sınava zamanı nasıl böleyim?",
        "Dersi geçmek için çalışma stratejisi ver",
        "Quizlere hazırlanma programı istiyorum",
    ],
    "major_secimi": [
        "Hangi bölümü seçmeliyim?",
        "CS mi IE mi bana daha uygun?",
        "Major seçimi konusunda kararsızım",
        "Bilgisayar bilimi okumak mantıklı mı?",
        "Which major should I choose?",
        "Help me compare engineering majors",
        "Ana dalımı seçmeme yardım et",
        "Ekonomi mi bilgisayar mı seçsem?",
        "What degree program fits my interests?",
        "Bölüm tercihinde nelere bakmalıyım?",
        "Psikoloji ve ekonomi arasında kaldım",
        "Endüstri mühendisliğini major yapmalı mıyım?",
    ],
    "alanda_ozellesme": [
        "CS içinde hangi alanda uzmanlaşmalıyım?",
        "NLP mi security mi seçmeliyim?",
        "Yapay zeka alanında nasıl özelleşirim?",
        "Bilgisayar bilimlerinde alt alan seçimi",
        "Which CS specialization fits me?",
        "I want to specialize in data engineering",
        "Kariyerim için hangi CS yönelimi uygun?",
        "Systems alanına yönelmek istiyorum",
        "Should I focus on AI or web?",
        "Veri bilimi uzmanlığına nasıl ilerlerim?",
        "Siber güvenlikte derinleşmek istiyorum",
        "Major içinde bir odak alanı seçmek istiyorum",
    ],
    "ders_ayrintisi": [
        "CS 412 dersini kim veriyor?",
        "Bu dersin önkoşulu nedir?",
        "CS 300 içeriğini açıklar mısın?",
        "Dersin syllabus ve notlandırması nasıl?",
        "Who teaches CS 201?",
        "What are the prerequisites for this course?",
        "Bu ders hangi gün ve saatte?",
        "CS 455 zor bir ders mi?",
        "Tell me the workload of CS 307",
        "Bu ders kaç SU ve ECTS?",
        "Dersin projeleri ağır mı?",
        "CS 310 hakkında ayrıntı ver",
    ],
    "diger": [
        "Merhaba nasılsın?",
        "Bugün hava nasıl?",
        "Şifremi unuttum",
        "Yemekhane nerede?",
        "Thank you",
        "Who built this application?",
        "Kampüse nasıl giderim?",
        "Bana bir şaka anlat",
        "What time is it?",
        "Selam",
        "Kütüphane saat kaçta kapanıyor?",
        "Wi-Fi şifresi nedir?",
    ],
}


def _data() -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for label, examples in SAMPLES.items():
        texts.extend(examples)
        labels.extend([label] * len(examples))
    return texts, labels


def _scores(labels: list[str], predicted: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": round(float(f1_score(labels, predicted, average="macro")), 6),
        "accuracy": round(float(accuracy_score(labels, predicted)), 6),
    }


def main() -> None:
    texts, labels = _data()
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

    baseline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            (
                "classifier",
                LogisticRegression(
                    random_state=42,
                    class_weight="balanced",
                    max_iter=2000,
                ),
            ),
        ]
    )
    baseline_pred = cross_val_predict(baseline, texts, labels, cv=folds)
    baseline_scores = _scores(labels, baseline_pred)

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(BERT_MODEL)
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    bert_classifier = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=2000,
    )
    bert_pred = cross_val_predict(bert_classifier, embeddings, labels, cv=folds)
    bert_scores = _scores(labels, bert_pred)

    winner = (
        "bert"
        if bert_scores["macro_f1"] > baseline_scores["macro_f1"]
        else "tfidf"
    )
    result = {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(texts),
        "labels": sorted(SAMPLES),
        "cross_validation": "StratifiedKFold(n_splits=4, shuffle=True, random_state=42)",
        "selection_metric": "macro_f1",
        "minimum_improvement": 0.0,
        "tfidf": baseline_scores,
        "bert": {**bert_scores, "model": BERT_MODEL},
        "winner": winner,
        "bert_selected": winner == "bert",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if winner == "bert":
        bert_classifier.fit(embeddings, labels)
        CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "classifier": bert_classifier,
                "model_name": BERT_MODEL,
                "labels": sorted(SAMPLES),
            },
            CLASSIFIER_PATH,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

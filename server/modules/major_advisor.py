from __future__ import annotations

"""
Deterministic major-selection advisor (a short "mini-test", no LLM, no RAG).

"Which major should I choose?" must NOT trigger a graduation audit or dump retrieved requirement
chunks. Instead it runs a bounded, one-shot questionnaire: the student is asked a few preference
questions once, answers in a single message, and the module scores those answers against a fixed
rubric and returns ONE definitive best-fit program (plus the runner-up). Everything is computed in
code so the recommendation is stable and testable; the LLM is only used afterwards, for free-form
follow-up questions about the recommended program.

Flow (driven from main.py via the session working context):
  1. First "hangi bölüm" turn  -> build_quiz(); set working_context.major_quiz_pending = True
  2. The student's answer turn  -> evaluate(); clear the flag, store recommended_major
  3. Later questions           -> normal RAG (major_secimi intent)
"""

import re
from dataclasses import dataclass, field


# code -> (Turkish name, English name)
PROGRAM_NAMES: dict[str, tuple[str, str]] = {
    "CS": ("Bilgisayar Bilimi ve Mühendisliği", "Computer Science and Engineering"),
    "DSA": ("Veri Bilimi ve Analitiği", "Data Science and Analytics"),
    "IE": ("Endüstri / Üretim Sistemleri Mühendisliği", "Industrial / Manufacturing Systems Engineering"),
    "EE": ("Elektronik Mühendisliği", "Electronics Engineering"),
    "ME": ("Mekatronik Mühendisliği", "Mechatronics Engineering"),
    "BIO": ("Moleküler Biyoloji, Genetik ve Biyomühendislik", "Molecular Biology, Genetics and Bioengineering"),
    "MAT": ("Malzeme Bilimi ve Nano Mühendislik", "Materials Science and Nano Engineering"),
    "ECON": ("Ekonomi", "Economics"),
    "PSY": ("Psikoloji", "Psychology"),
}

# One-line theme used to justify the recommendation.
PROGRAM_THEME: dict[str, tuple[str, str]] = {
    "CS": ("yazılım, algoritma ve yapay zekâ", "software, algorithms and AI"),
    "DSA": ("veri, istatistik ve analitik", "data, statistics and analytics"),
    "IE": ("optimizasyon, süreç ve operasyon", "optimization, processes and operations"),
    "EE": ("elektronik, sinyal ve donanım", "electronics, signals and hardware"),
    "ME": ("mekanik, robotik ve tasarım", "mechanics, robotics and design"),
    "BIO": ("biyoloji, genetik ve laboratuvar", "biology, genetics and lab work"),
    "MAT": ("malzeme, nano teknoloji ve kimya/fizik", "materials, nanotechnology and chemistry/physics"),
    "ECON": ("ekonomi, finans ve karar analizi", "economics, finance and decision analysis"),
    "PSY": ("insan davranışı ve zihin", "human behaviour and the mind"),
}

_LETTERS = "abcde"


@dataclass(frozen=True)
class _Option:
    tr: str
    en: str
    weights: dict[str, float]


@dataclass(frozen=True)
class _Question:
    tr: str
    en: str
    options: tuple[_Option, ...]


QUESTIONS: tuple[_Question, ...] = (
    _Question(
        "Aşağıdakilerden hangisi sana en heyecan verici geliyor?",
        "Which of these excites you the most?",
        (
            _Option("Yazılım ve algoritma yazmak, yapay zekâ", "Writing software/algorithms, AI",
                    {"CS": 2, "DSA": 1}),
            _Option("Veriyi ve istatistiği analiz etmek", "Analysing data and statistics",
                    {"DSA": 2, "IE": 1, "ECON": 1}),
            _Option("Fiziksel makineler / robotlar tasarlamak", "Designing physical machines / robots",
                    {"ME": 2, "EE": 1}),
            _Option("Elektronik, devreler ve sinyaller", "Electronics, circuits and signals",
                    {"EE": 2, "ME": 1}),
        ),
    ),
    _Question(
        "Hangi dersten / konudan daha çok keyif alırsın?",
        "Which subject do you enjoy more?",
        (
            _Option("Matematik ve soyut düşünme", "Mathematics and abstraction",
                    {"MAT": 1, "CS": 1, "IE": 1}),
            _Option("Biyoloji ve canlı sistemler", "Biology and living systems",
                    {"BIO": 2}),
            _Option("Ekonomi, piyasalar ve kararlar", "Economics, markets and decisions",
                    {"ECON": 2, "IE": 1}),
            _Option("İnsan davranışı ve zihin", "Human behaviour and the mind",
                    {"PSY": 2}),
        ),
    ),
    _Question(
        "En çok hangi tür problemi çözmek istersin?",
        "What kind of problem would you most like to solve?",
        (
            _Option("Uygulama / yapay zekâ sistemleri geliştirmek", "Building apps / AI systems",
                    {"CS": 2, "DSA": 1}),
            _Option("Süreçleri ve lojistiği optimize etmek", "Optimising processes and logistics",
                    {"IE": 2, "ECON": 1}),
            _Option("Malzeme, kimya, nano teknoloji", "Materials, chemistry, nanotechnology",
                    {"MAT": 2, "BIO": 1}),
            _Option("Laboratuvarda deney yapmak", "Doing experiments in a lab",
                    {"BIO": 1, "MAT": 1, "EE": 1}),
        ),
    ),
    _Question(
        "İleride hangi işi yapmayı hayal ediyorsun?",
        "Which career would you love?",
        (
            _Option("Yazılım mühendisi / veri bilimci", "Software engineer / data scientist",
                    {"CS": 2, "DSA": 1}),
            _Option("Endüstri mühendisi / danışman", "Industrial engineer / consultant",
                    {"IE": 2, "ECON": 1}),
            _Option("Donanım / robotik mühendisi", "Hardware / robotics engineer",
                    {"EE": 1, "ME": 2}),
            _Option("Araştırmacı (biyoloji/fizik/psikoloji/ekonomi)", "Researcher (bio/physics/psych/econ)",
                    {"BIO": 1, "MAT": 1, "PSY": 1, "ECON": 1}),
        ),
    ),
    _Question(
        "Ne kadar programlama yapmak istersin?",
        "How much programming do you want to do?",
        (
            _Option("Çok — kod yazmayı seviyorum", "A lot — I love coding",
                    {"CS": 2, "DSA": 1}),
            _Option("Biraz, uygulamalı düzeyde", "Some, applied",
                    {"DSA": 1, "IE": 1, "EE": 1}),
            _Option("Az / neredeyse hiç", "Little / almost none",
                    {"BIO": 1, "PSY": 1, "ECON": 1, "ME": 1}),
        ),
    ),
)

# Keyword fallback so a free-text answer ("robotları severim, mekanik ilgimi çekiyor") still scores.
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "CS": ("yazılım", "software", "programlama", "kod", "coding", "algoritma", "algorithm", "uygulama", "app", "ai", "yapay zek"),
    "DSA": ("veri", "data", "istatistik", "statistic", "analitik", "analytics", "analiz"),
    "IE": ("endüstri", "industrial", "optimiz", "süreç", "process", "operasyon", "lojistik", "logistic", "üretim", "verimlilik"),
    "EE": ("elektronik", "electronic", "devre", "circuit", "sinyal", "signal", "donanım", "hardware", "gömülü", "embedded"),
    "ME": ("makine", "mechanic", "mekanik", "robot", "mekatronik", "mechatronic", "tasarım", "design"),
    "BIO": ("biyoloji", "biology", "genetik", "genetic", "canlı", "laboratuvar", "lab", "biyomüh"),
    "MAT": ("malzeme", "material", "nano", "kimya", "chemistry", "fizik", "physics"),
    "ECON": ("ekonomi", "econom", "finans", "finance", "piyasa", "market", "para", "yatırım"),
    "PSY": ("psikoloji", "psycholog", "davranış", "behaviour", "behavior", "zihin", "mind", "insan"),
}


def build_quiz(language: str = "tr") -> str:
    """The one-shot questionnaire message (all questions in one turn)."""
    tr = language == "tr"
    lines = []
    if tr:
        lines.append(
            "Sana en uygun bölümü bulmak için kısa bir mini test yapalım. Aşağıdaki 5 soruyu "
            "yanıtla; her soru için bir harf seç ve **hepsini tek mesajda** yaz (örn: `1a 2c 3a 4b 5a`)."
        )
    else:
        lines.append(
            "Let's do a short mini-test to find your best-fit major. Answer the 5 questions below; "
            "pick one letter per question and send **all of them in one message** (e.g. `1a 2c 3a 4b 5a`)."
        )
    for i, q in enumerate(QUESTIONS, 1):
        lines.append("")
        lines.append(f"**{i}. {q.tr if tr else q.en}**")
        for j, opt in enumerate(q.options):
            lines.append(f"   {_LETTERS[j]}) {opt.tr if tr else opt.en}")
    return "\n".join(lines)


def looks_like_answers(text: str) -> bool:
    """Heuristic: is this message a reply to the quiz rather than a fresh question?"""
    t = (text or "").lower()
    lettered = len(re.findall(r"(?<!\w)([1-5])\s*[\).:\-]?\s*([a-e])(?!\w)", t))
    loose_letters = len(re.findall(r"(?<![a-z])([a-e])(?![a-z])", t))
    keyword_hits = sum(1 for kws in _KEYWORDS.values() if any(k in t for k in kws))
    return lettered >= 2 or (loose_letters >= 3 and len(t) <= 60) or keyword_hits >= 2


def is_major_question(text: str) -> bool:
    """A generic 'which major should I pick' question that should open the quiz."""
    t = (text or "").lower()
    triggers = (
        "hangi bölüm", "hangi bolum", "bölümü seç", "bolumu sec", "bölüm seç", "bolum sec",
        "major seç", "major sec", "which major", "which program", "hangi programı", "ana dal",
    )
    return any(k in t for k in triggers)


@dataclass
class MajorRecommendation:
    best: str
    second: str | None
    scores: dict[str, float]
    body: str
    summary: str
    answered: int = 0
    used_keywords: bool = False


def _parse_letter_answers(text: str) -> dict[int, int]:
    """{question_index(0-based): option_index} from patterns like '1a', '2 c', '3) b'.

    Falls back to a bare sequence of standalone letters ('a c b a b') assigned to Q1..Q5 in order.
    """
    out: dict[int, int] = {}
    for qn, letter in re.findall(r"(?<!\w)([1-5])\s*[\).:\-]?\s*([a-eA-E])(?!\w)", text or ""):
        qi = int(qn) - 1
        oi = _LETTERS.index(letter.lower())
        if 0 <= qi < len(QUESTIONS) and oi < len(QUESTIONS[qi].options) and qi not in out:
            out[qi] = oi
    if out:
        return out
    bare = re.findall(r"(?<![a-zA-Z])([a-eA-E])(?![a-zA-Z])", text or "")
    for qi, letter in enumerate(bare[: len(QUESTIONS)]):
        oi = _LETTERS.index(letter.lower())
        if oi < len(QUESTIONS[qi].options):
            out[qi] = oi
    return out


def score_answers(text: str) -> tuple[dict[str, float], int, bool]:
    """Return (scores, answered_questions, used_keyword_fallback)."""
    scores: dict[str, float] = {p: 0.0 for p in PROGRAM_NAMES}
    letters = _parse_letter_answers(text)
    for qi, oi in letters.items():
        for prog, pts in QUESTIONS[qi].options[oi].weights.items():
            scores[prog] += pts
    used_keywords = False
    if len(letters) < 3:  # weak lettered signal -> add keyword evidence from the free text
        t = (text or "").lower()
        for prog, kws in _KEYWORDS.items():
            hits = sum(1 for k in kws if k in t)
            if hits:
                scores[prog] += hits
                used_keywords = True
    return scores, len(letters), used_keywords


def evaluate(text: str, *, current_major: str | None = None, language: str = "tr") -> MajorRecommendation:
    scores, answered, used_keywords = score_answers(text)
    current = (current_major or "").strip().upper() or None
    # Deterministic tie-break: higher score, then the student's own major, then a stable order.
    order = sorted(
        scores.items(),
        key=lambda kv: (-kv[1], 0 if kv[0] == current else 1, kv[0]),
    )
    best = order[0][0]
    second = order[1][0] if len(order) > 1 and order[1][1] > 0 else None
    if order[0][1] <= 0:  # no signal at all -> steer back to the questionnaire
        body = (
            "Cevaplarını tam anlayamadım. Lütfen her soru için bir harf seç ve tek mesajda yaz, "
            "örn: `1a 2c 3a 4b 5a`."
            if language == "tr" else
            "I couldn't read your answers. Please pick one letter per question in a single message, "
            "e.g. `1a 2c 3a 4b 5a`."
        )
        return MajorRecommendation(best, second, scores, body, body, answered, used_keywords)

    body, summary = _render(best, second, current, language)
    return MajorRecommendation(best, second, scores, body, summary, answered, used_keywords)


def _name(code: str, language: str) -> str:
    tr, en = PROGRAM_NAMES.get(code, (code, code))
    label = tr if language == "tr" else en
    return f"{label} ({code})"


def _render(best: str, second: str | None, current: str | None, language: str) -> tuple[str, str]:
    tr = language == "tr"
    theme = PROGRAM_THEME.get(best, ("", ""))[0 if tr else 1]
    lines: list[str] = []
    if tr:
        if current and best == current:
            lines.append(f"Verdiğin cevaplara göre sana en uygun bölüm **zaten kendi bölümün: {_name(best,'tr')}**.")
        else:
            lines.append(f"Verdiğin cevaplara göre sana en uygun bölüm: **{_name(best,'tr')}**.")
            if current:
                lines.append(f"(Şu an {_name(current,'tr')} bölümündesin.)")
        lines.append("")
        lines.append(f"Çünkü cevapların en çok {theme} yönünü işaret ediyor.")
        if second:
            lines.append(f"İkinci en yakın olduğun bölüm: **{_name(second,'tr')}**.")
        lines.append("")
        lines.append("İstersen bu bölümün dersleri, kariyeri veya müfredatı hakkında soru sorabilirsin.")
    else:
        if current and best == current:
            lines.append(f"Based on your answers, your best-fit major is **already your own: {_name(best,'en')}**.")
        else:
            lines.append(f"Based on your answers, your best-fit major is: **{_name(best,'en')}**.")
            if current:
                lines.append(f"(You are currently in {_name(current,'en')}.)")
        lines.append("")
        lines.append(f"Your answers point most strongly toward {theme}.")
        if second:
            lines.append(f"Your second closest fit is: **{_name(second,'en')}**.")
        lines.append("")
        lines.append("You can now ask me about this program's courses, career paths or curriculum.")

    summary = (
        f"En uygun bölüm: {_name(best,'tr')}." + (f" İkinci: {_name(second,'tr')}." if second else "")
        if tr else
        f"Best-fit major: {_name(best,'en')}." + (f" Runner-up: {_name(second,'en')}." if second else "")
    )
    return "\n".join(lines), summary

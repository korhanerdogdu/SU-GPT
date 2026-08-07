from functools import lru_cache

from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from modules.llm_providers import build_chat_model
from modules.language import detect_language
from modules import intents


@lru_cache(maxsize=1)
def _build_llm():
    # One process-wide adapter preserves connection/circuit state across requests. Tests and
    # controlled benchmarks may clear this cache explicitly when changing environment settings.
    model = build_chat_model()
    # OpenRouter may route aliases or remove models.  Authenticate against its catalogue once
    # before the process sends any inference request; an unavailable exact ID fails closed.
    model.verify_exact_model_available()
    return model


_LANGUAGE_DIRECTIVE = {
    "en": (
        "CRITICAL OUTPUT RULE — LANGUAGE: The user wrote in ENGLISH. Write your ENTIRE answer in "
        "English. Do not answer in Turkish. Headings, labels and the Sources list must also be in "
        "English. This rule overrides every other instruction below, including any Turkish "
        "phrasing in the templates."
    ),
    "tr": (
        "KRİTİK ÇIKTI KURALI — DİL: Kullanıcı TÜRKÇE yazdı. Cevabının TAMAMINI Türkçe yaz. "
        "Bu kural aşağıdaki diğer tüm talimatların üzerindedir."
    ),
}


LLM_ONLY_PROMPT = """Sen Sabancı Üniversitesi öğrencilerine yardımcı olan bir akademik danışman asistanısın.

Bu cevabı SADECE kendi genel bilginle üretiyorsun. Sana hiçbir belge, müfredat kaynağı veya \
öğrenci kaydı verilmedi. Bu mod, bir karşılaştırma temeli (baseline) olarak çalışır.

- Elinde resmi kaynak olmadığını unutma.
- Sabancı'ya özgü bir sayı, kural veya ders bilgisinden emin değilsen bunu açıkça belirt.
- Cevap dili için en üstteki KRİTİK ÇIKTI KURALI geçerlidir; bu kuralı hiçbir koşulda ihlal etme.

Soru: {question}

Cevap:"""


def answer_without_context(question: str) -> str:
    """LLM-only baseline (Section 3): no retrieval, no student data, no sources.

    Exists purely so the evaluation track can quantify what retrieval adds. It is expected to
    hallucinate Sabanci-specific facts — that measurement is the reason the mode exists — so it
    must never be the default and the UI must warn when it is selected.
    """
    answer, _ = answer_without_context_with_telemetry(question)
    return answer


def answer_without_context_with_telemetry(question: str) -> tuple[str, dict | None]:
    """Return the isolated answer plus safe provider usage for quota accounting."""
    llm = _build_llm()
    directive = _LANGUAGE_DIRECTIVE[detect_language(question)]
    response = llm.invoke(f"{directive}\n\n{LLM_ONLY_PROMPT.format(question=question)}")
    return getattr(response, "content", str(response)), llm.get_last_telemetry()


_PROMPT_STRATEGIES = {
    # "Zero-shot" in the standard prompting taxonomy: task instruction only, no worked examples,
    # no explicit reasoning scaffold.
    "basic": "Use the supplied context conservatively and answer the question directly.",
    # "Guided prompting": an explicit numbered procedure to follow, distinct from "algorithmic"
    # below in that it is a checklist a human advisor would follow, not a data-transform pass.
    "lookup": (
        "Before composing the answer, privately locate the exact context records that match "
        "the requested course, program, term, or requirement. Cross-check identifiers and "
        "statuses. Do not expose private chain-of-thought; show only supporting facts."
    ),
    "algorithmic": (
        "Treat the task as a deterministic data operation: normalize identifiers, filter to "
        "the student's program and curriculum, apply each rule exactly once, verify totals, "
        "and then explain the result. Do not expose private chain-of-thought."
    ),
    "structured_lookup": (
        "First perform a private exact lookup of the relevant records, then apply the required "
        "rules in order and run a consistency check. Prefer a compact Markdown table whenever "
        "the answer has repeated fields such as courses, credits, requirements, schedules, or "
        "status. Do not expose private chain-of-thought; show only inputs, results, and concise "
        "checkable arithmetic."
    ),
    # "Few-shot": worked input -> output exemplars, no explicit reasoning shown in the examples
    # themselves (contrast with few_shot_cot below, where the exemplars also show the reasoning).
    "few_shot": (
        "Follow the pattern in these two worked examples before answering the real question.\n"
        "Example 1 -- Input: student has completed CS 201, CS 204, MATH 101, MATH 102; asks for "
        "a course recommendation; candidate pool contains CS 300, CS 301, CS 303, ENS 211, "
        "MATH 201. Output: recommend from the candidate pool only, in the fixed numbered format "
        "the rest of this prompt specifies, never re-listing CS 201/204/101/102.\n"
        "Example 2 -- Input: student already answered a course-recommendation question and now "
        "writes only 'NLP'. Output: treat this as the interest-area answer to the prior "
        "question, not a new unrelated request; narrow the same recommendation to NLP-aligned "
        "candidates from the pool.\n"
        "Now answer the actual question the same way. Do not expose private chain-of-thought."
    ),
    # "Chain-of-Thought" in its standard form: an explicit instruction to reason through
    # intermediate steps before the final answer, without exemplars and without framing it as a
    # "lookup" or "algorithm" the way the two strategies above do.
    "cot": (
        "Reason through this step by step before answering: first identify what the question is "
        "actually asking, then identify which facts from the context are relevant, then apply "
        "the applicable rules in order, then check the result is internally consistent, and only "
        "then write the final answer. Keep these steps private; the user only sees the final "
        "answer, never the reasoning trace."
    ),
    # "Zero-shot CoT": the specific, minimal "let's think step by step" trigger phrase, without
    # the more elaborate multi-step scaffold "cot" above spells out.
    "zero_shot_cot": (
        "Let's think step by step before answering. Do not expose private chain-of-thought in "
        "the final answer -- think it through internally, then state only the conclusion."
    ),
    # "Few-shot CoT": exemplars that show the reasoning steps themselves, not just the final
    # answer -- this is what distinguishes it from plain "few_shot" above.
    "few_shot_cot": (
        "Follow the reasoning pattern in this worked example before answering.\n"
        "Example -- Question: 'What should I take next term?' Reasoning (kept private in the "
        "real answer): (1) the student's completed courses are CS 201, CS 204, MATH 101, MATH "
        "102 -> foundations are done; (2) the candidate pool marks CS 300, CS 301, ENS 211 as "
        "eligible and not yet taken; (3) no interest area was stated yet, so recommend a general "
        "next-stage set and ask for one at the end; (4) exclude anything already completed. "
        "Final answer shown to the student: the numbered recommendation list plus the interest "
        "question, with no visible trace of steps 1-4.\n"
        "Apply the same private reasoning pattern to the real question, then show only the final "
        "answer in the format the rest of this prompt specifies."
    ),
    # "Tree-of-Thoughts", bounded to a single model call for cost: instead of committing to the
    # first draft, generate a few internally, check each against the rules, and keep the best.
    "tree_of_thought": (
        "Before answering, privately draft two or three candidate answers that satisfy the "
        "context and rules differently (e.g. different course orderings, different framings of "
        "the same eligible set). For each draft, privately check it against every rule in this "
        "prompt (no already-taken course, correct category, correct language, correct format). "
        "Discard any draft that fails a check. From the drafts that pass, output only the single "
        "best one. Never show the discarded drafts or the evaluation itself to the user."
    ),
    # "Graph-of-Thoughts", also bounded to a single call: reason about sub-aspects of the
    # question as separate "nodes", then explicitly merge them into one coherent answer, rather
    # than reasoning linearly top-to-bottom the way "cot" does.
    "graph_of_thoughts": (
        "Before answering, privately reason about these aspects as separate, independent notes: "
        "(a) what the context's official records actually say, (b) what rule or category applies, "
        "(c) what the student has already completed and must not be re-shown, (d) any interest "
        "area, difficulty preference, or prior override already stated in this conversation. "
        "Then merge these notes into one single, internally consistent final answer -- resolve "
        "any conflict between notes explicitly (e.g. a stated override always wins over a "
        "default). Never show the separate notes or the merge step to the user."
    ),
}


def get_llm_chain(
    retriever,
    intent: str = intents.OTHER,
    language: str = "tr",
    prompt_strategy: str = "basic",
):
    llm = _build_llm()
    language_directive = _LANGUAGE_DIRECTIVE.get(language, _LANGUAGE_DIRECTIVE["tr"])
    strategy_directive = _PROMPT_STRATEGIES.get(prompt_strategy, _PROMPT_STRATEGIES["basic"])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
{language_directive}

ACTIVE PROMPT STRATEGY
{strategy_directive}

Sen Sabancı Üniversitesi programları için özelleştirilmiş, sıfır hata toleransıyla çalışan bir Yapay Zeka Akademik Danışmanısın. RAG üzerinden sana sağlanan resmi degree requirement / degree evaluation kaynaklarını ve öğrencinin MongoDB ders geçmişini kullanarak analiz yaparsın.

KULLANICI NİYETİ (Query Router sonucu): {detected_intent}
Bu intent'i cevap modunu belirlemek için aktif sinyal olarak kullan. Intent "mezuniyet_durumu" değilse mezuniyet audit'i üretme; intent "ders_onerisi" ise MongoDB geçmişini alınmış dersleri elemek için kullan; intent "calisma_plani", "major_secimi", "alanda_ozellesme" veya "ders_ayrintisi" ise yalnızca o sorunun gerektirdiği akademik cevabı ver.

ÖNEMLİ MUHAKEME KURALI
Adımları dikkatle uygula, ama özel zincirleme düşünceyi kullanıcıya gösterme. Sadece nihai hesapları, ders dağılımını, görünür kısa kontrol matematiğini ve sonucu yaz.

MOD SEÇİMİ
- Cevap modunu yalnızca User Question metnine göre seç. RAG Context veya MongoDB profile içinde mezuniyet/kredi bilgisi geçmesi, tek başına mezuniyet audit cevabı vermek için sebep değildir.
- Kullanıcı açıkça mezuniyet, kredi, kalan ders, degree evaluation, audit, kategori dağılımı veya "hangi derslerim sayıldı" gibi bir şey sorarsa SADECE mezuniyet audit cevabı ver.
- Kullanıcı "hangi dersleri alayım", "ders öner", "gelecek dönem", "program öner", "NLP", "Web", "Data", "kolay/zor ders", "schedule" gibi ders seçimi/öneri niyeti gösterirse MEZUNİYET AUDIT YAPMA. Bu durumda MongoDB geçmişini sadece alınmış dersleri elemek ve kişiselleştirmek için kullan.
- Intent "review" ise özellik kapalıdır; eğitmen puanı veya özel sohbet içeriği üretme.
- Intent "exam" ise sadece sınav/PDF kaynaklarında geçen soru, konu ve formatları kullan; kaynakta yoksa açıkça yok de.
- Kullanıcı sadece kısa bir ilgi alanı yazarsa, örn. "NLP", "Web", "Data", bunu ders öneri modu için ilgi alanı cevabı kabul et; mezuniyet durumu anlatma.
- Audit cevabında ASLA "Ders Önerileri", "Çalışma Tavsiyeleri" veya yeni ders listesi ekleme. Kullanıcı açıkça ders programı/öneri isterse ancak o zaman öneri moduna geç.
- Kullanıcı sadece mezuniyet/kredi durumunu sorduyse çalışma tavsiyesi verme.

BÖLÜM 1: KESİNLİKLE UYULACAK MEZUNİYET AUDIT ALGORİTMASI
Mezuniyet durumunu hesaplarken SADECE SU CREDIT kullan ve şu sıralı algoritmayı işlet:

Tekil Atama Kuralı: Bir ders KESİNLİKLE sadece bir kategoride sayılabilir. "Kategori Belirsiz" gibi başlıklar ASLA kullanılamaz.

RESMİ RAG KAYNAĞI ÖNCELİĞİ
- Kategori dağılımını prompt hafızasından üretme. Derslerin hangi kategoriye sayılacağını yalnızca RAG Context'teki resmi program kaynaklarından çıkar.
- Context içinde "official degree evaluation projection", "official degree evaluation category allocation" veya "official degree evaluation course assignment" varsa, bu kaynaklar en yüksek otoritedir; dersleri oradaki kategori atamalarına göre yerleştir.
- Context içinde sadece "degree requirement profile" ve "degree requirement category pool" varsa, kategori havuzlarını ve minimumları bu RAG kaynaklarından oku; bir ders birden fazla havuza uygunsa kaynakta yazan seçim/taşma kurallarını uygula.
- Öğrencinin programıyla aynı program/degree_code kaynaklarını kullan. CS/BSCS için CS/BSCS kaynaklarını, ileride IE yüklendiğinde IE kaynaklarını kullan. Farklı major/minor kaynaklarını birbirine karıştırma.
- Minor, başka major, faculty, engineering ve basic science kaynaklarını ana University/Core/Required/Area/Free dağılımı için kullanma; bunları yalnızca kullanıcı özellikle isterse üst şart olarak ayrıca değerlendir.
- RAG Context bir dersin kategori atamasını desteklemiyorsa kategori uydurma; hangi kaynak eksikse kısa ve açık söyle.

Uygulama disiplini:
- MongoDB öğrenci geçmişindeki tüm dersleri dikkate al; 0 kredilik dersleri de "Alınanlar" listesinde göster.
- Aynı dersi iki kere sayma. Eşdeğer/tekrar/withdrawn bilgisi Context içinde açıkça verilmişse onu esas al.
- Kategori toplamlarını derslerin SU Credit değerlerinden hesapla. ECTS kullanma.
- MongoDB Context içinde "Authoritative completed SU credit total" verilmişse, GENEL MEZUNİYET TOPLAMI için bu değeri kullan ve kategori toplamlarıyla çelişki varsa kısa bir notla belirt.
- Audit cevabında resmi kaynakta tanımlanan ana mezuniyet kategorilerini yaz. Faculty/Engineering/Basic Science'i kullanıcı özellikle istemedikçe ekleme.

ÇIKTI ŞABLONU: Her kategori için tam olarak şu formatı kullan:
[Kategori Adı] Durumu: Alınanlar: [Dersler] | Toplam: X/Y SU | Durum: [Tamamlandı / Z Kredi Eksik]

En sona Genel Mezuniyet Toplamını ekle:
GENEL MEZUNİYET DURUMU: Tamamlanan: X/125 SU Kredisi | Kalan: Y SU Kredisi.

BÖLÜM 2: AKILLI DERS PROGRAMI ÖNERME KURALLARI

Öneri modu davranışı:
- Ders önerisi/program sorularında cevaba "Öncelikle mezuniyet durumunu kontrol edelim" gibi bir girişle başlama.
- Ders önerisi/program sorularında genel mezuniyet durumunu, 125/125 bilgisini veya kategori audit tablosunu yazma; kullanıcı ayrıca açıkça isterse ayrı cevapta ver.
- Öğrenci ders programı ister ama ilgi alanı belirtmezse sadece şunu sor: "Hangi alana ilgilisin? Örn: NLP, Web, Data, Systems, AI, Security." Bu durumda ders listesi ve mezuniyet audit'i verme.
- RAG Context içinde "Course recommendation strategy" varsa kullanıcı ilgi alanını zaten vermiştir. Bu durumda ASLA tekrar "hangi alana ilgilisin" veya "hangi alt alanda derinleşmek istersin" diye sorma; doğrudan program öner.
- RAG Context içinde "CS recommendation candidate" kaynakları varsa, bunları ders programı aday havuzu olarak kullan. "already_taken_do_not_recommend" işaretli dersleri listeye alma; "eligible_candidate" işaretli derslerden tam 5 öneri üretmeye çalış.

Körlük Koruması (Kritik): Önerilecek ders havuzunu filtrelerken, öğrencinin halihazırda aldığı/tamamladığı dersleri (MongoDB geçmişi) ASLA tekrar önerme.

Yan Disiplin Yönlendirmesi: Öğrenci ilgilendiği alandaki tüm dersleri zaten almışsa, "Bu alandaki dersleri tamamlamışsın, sana şu destekleyici dersleri öneririm" diyerek alakalı farklı disiplinlerden ders ver.

Aksi belirtilmedikçe tam 5 ders öner. Öğrenci "zorlaştır/ağırlaştır" derse CS 308, 412 gibi üst düzey projeli dersleri; "kolaylaştır" derse giriş seviyesi/hafif dersleri öner.

RAG schedule verisini kullanarak dersi veren hocanın adını mutlaka belirt (Örn: CS 412 - Yücel Saygın). Eğer Context içinde hoca adı yoksa "retrieved schedule context hoca bilgisini vermedi" de; hoca uydurma.

Çalışma tavsiyelerini yalnızca kullanıcı açıkça "nasıl çalışmalıyım", "çalışma tavsiyesi ver", "ders programı öner" veya "bu dersleri alırsam nasıl hazırlanayım" derse ekle. Salt mezuniyet/kredi/audit cevaplarında çalışma tavsiyesi ekleme.

Ders programı cevap formatı:
- Kısa bir notla, alınmış ana dersleri tekrar önermediğini söyle.
- Sonra Markdown liste halinde tam 5 ders ver:
  1. **KOD - Ders Adı** — Hoca: X | Zaman: gün/saat | Neden: kısa gerekçe
- Sonda en fazla 2 cümlelik kısa strateji notu ekle. Mezuniyet audit'i ekleme.

BÖLÜM 3: CEVAP DİSİPLİNİ VE KAYNAKLAMA
- Sabancı'ya özgü müfredat, kredi, dönem, hoca, prerequisite ve ders uygunluğu bilgilerini sadece RAG Context'ten çıkar.
- Cevap dili için en üstteki KRİTİK ÇIKTI KURALI geçerlidir; bu kuralı hiçbir koşulda ihlal etme.
- Gereksiz uzun paragraf yazma; audit ve önerilerde net, şablonlu ve kontrol edilebilir ol.
- Kullanıcıya MongoDB, RAG, retrieval, deterministic engine, prompt, context veya kaynak etiketi gibi teknik uygulama ayrıntılarını ASLA anlatma.
- "Kullanacağım", "hesaplayacağım", "kontrol edeceğim" gibi gelecek zamanlı süreç anlatımıyla başlama; doğrudan nihai sonucu ver.
- LaTeX veya tablo kullanırken biçim teknolojisini açıklama; yalnızca okunabilir formülü, tabloyu ve akademik sonucu göster.
- Kaynak listesini cevap metnine yazma. Kaynaklar arayüzde yalnızca yöneticiye ayrı teknik metadata olarak gösterilir.

RAG Context:
{context}

User Question:
{question}

{language_directive}

Answer:
""".replace("{detected_intent}", intents.to_legacy(intent)).replace(
            "{language_directive}", language_directive
        ).replace("{strategy_directive}", strategy_directive),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

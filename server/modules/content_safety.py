from __future__ import annotations

"""Deterministic content-safety gate for incoming student questions.

adviSU is an academic advisor for Sabancı University students. Abuse, hate speech, threats and
sexual harassment are out of scope, and a generic "I can only answer academic questions" reply
is the wrong response to all of them: telling a student who wrote about self-harm the same
sentence you tell a student who swore at the bot is a failure, not a policy.

So this module classifies the *kind* of unsafe input and returns a category-specific response:

    self_harm            a short, human reply pointing at real help. Never a brush-off.
    hate                 named as unacceptable, no engagement, scope restated.
    violence             threats are refused explicitly, scope restated.
    sexual_harassment    refused, scope restated.
    profanity            de-escalating: the question is welcome, the language is not.

Design decisions worth keeping:

1. **Runs before retrieval and before the LLM.** Abusive text is never forwarded to Groq. That
   protects the daily token budget and stops the model from engaging with the content at all.
2. **Deterministic regex, not a model.** A moderation call that can fail open, cost tokens or
   time out is worse than a small explicit rule set for this job. The rules are testable.
3. **False positives are a real risk in a CS school.** "How do I kill a process?", "attack
   vector", "buffer overflow exploit" and "CS 432 Computer and Network Security" are ordinary
   academic questions here. Violence patterns therefore match *directed threats* ("I will kill
   you"), never bare topic words, and `_TECHNICAL_EXEMPTIONS` protects the common systems idioms.
4. **Self-harm is checked first**, so "I want to kill myself" is never routed to `violence`.
"""

import re
from dataclasses import dataclass

# --- Categories, in evaluation order ----------------------------------------------------
SELF_HARM = "self_harm"
HATE = "hate"
VIOLENCE = "violence"
SEXUAL_HARASSMENT = "sexual_harassment"
PROFANITY = "profanity"

SAFE = ""


@dataclass(frozen=True)
class SafetyVerdict:
    """Outcome of the gate. `blocked` is the only thing callers must branch on."""

    category: str = SAFE
    matched: str = ""

    @property
    def blocked(self) -> bool:
        return bool(self.category)


# Systems/security vocabulary that is ordinary academic language at a CS school. If the message
# matches one of these, the violence rules are not allowed to fire on that span.
_TECHNICAL_EXEMPTIONS = re.compile(
    r"\b(kill\s+(?:the\s+)?(?:process|thread|task|job|session|server|container|port|pid)|"
    r"kill\s*-\s*9|killall|process\s+kill|"
    r"attack\s+(?:vector|surface|tree|model)|denial[\s-]of[\s-]service|ddos|"
    r"brute[\s-]force|penetration\s+test|pen\s*test|exploit\s+(?:code|chain|kit)|"
    r"buffer\s+overflow|sql\s+injection|threat\s+model|adversarial\s+attack|"
    r"süreci\s+öldür|process(?:i|'i)\s+öldür|saldırı\s+(?:vektörü|yüzeyi|modeli))\b",
    re.IGNORECASE,
)

# --- Self-harm: highest priority, never treated as violence -----------------------------
_SELF_HARM_RE = re.compile(
    r"\b(?:"
    r"kill\s+myself|killing\s+myself|end\s+(?:my|it)\s+(?:life|all)|take\s+my\s+own\s+life|"
    r"commit\s+suicide|suicidal|self[\s-]harm|hurt(?:ing)?\s+myself|cut(?:ting)?\s+myself|"
    r"want\s+to\s+die|don'?t\s+want\s+to\s+live|no\s+reason\s+to\s+live|"
    r"intihar|kendimi\s+öldür\w*|kendimi\s+oldur\w*|canıma\s+kıy\w*|canima\s+kiy\w*|"
    r"yaşamak\s+istemiyorum|yasamak\s+istemiyorum|ölmek\s+istiyorum|olmek\s+istiyorum|"
    r"kendime\s+zarar\s+ver\w*|hayatıma\s+son\s+ver\w*|hayatima\s+son\s+ver\w*"
    r")\b",
    re.IGNORECASE,
)

# --- Hate speech: slurs, plus eliminationist constructions aimed at a group --------------
_HATE_SLUR_RE = re.compile(
    r"\b(?:"
    r"n[i1]gg[e3]r|n[i1]gg[a4]|k[i1]ke|sp[i1]c\b|ch[i1]nk|g[o0]ok|w[e3]tback|towelhead|"
    r"f[a4]gg?[o0]t|tr[a4]nny|retard(?:ed)?|"
    r"çingene|cingene|kıro|kiro|gavur|zenci|"
    r"ermeni\s+döl\w*|ermeni\s+dol\w*|kürt\s+döl\w*|kurt\s+dol\w*"
    r")\b",
    re.IGNORECASE,
)

# "<group> should be killed / deported / are subhuman" — hate independent of any slur word.
_HATE_TARGET = (
    r"(?:kürtler|kurtler|ermeniler|araplar|suriyeliler|mülteciler|multeciler|yahudiler|"
    r"aleviler|müslümanlar|muslumanlar|hristiyanlar|siyahlar|çingeneler|cingeneler|kadınlar|"
    r"kadinlar|eşcinseller|escinseller|jews|muslims|arabs|kurds|refugees|blacks|immigrants|gays)"
)
_HATE_PREDICATE = (
    r"(?:ölmeli|olmeli|ölsün|olsun|gebersin|defolsun|defolup\s+gitsin|yok\s+edilmeli|"
    r"temizlenmeli|sürülmeli|surulmeli|insan\s+değil|insan\s+degil|aşağılık|asagilik|"
    r"should\s+(?:die|be\s+killed|be\s+deported|be\s+gassed)|are\s+(?:subhuman|vermin|animals)|"
    r"deserve\s+to\s+die)"
)
_HATE_CONSTRUCTION_RE = re.compile(
    rf"\b{_HATE_TARGET}\b[^.?!]{{0,40}}\b{_HATE_PREDICATE}\b|"
    rf"\b{_HATE_PREDICATE}\b[^.?!]{{0,40}}\b{_HATE_TARGET}\b",
    re.IGNORECASE,
)

# --- Violence: directed threats only ----------------------------------------------------
_VIOLENCE_RE = re.compile(
    r"\b(?:"
    r"(?:i(?:'|\s+a|\s+wi)?ll\s+|i\s+want\s+to\s+|going\s+to\s+)"
    r"(?:kill|murder|stab|shoot|beat|strangle|behead)\s+(?:you|him|her|them|everyone|him\s?self)|"
    r"kill\s+(?:you|him|her|them|everyone)\b|"
    r"shoot\s+up\s+(?:the\s+)?(?:school|campus|class|university)|"
    r"bomb\s+(?:the\s+)?(?:school|campus|class|university|building)|"
    r"school\s+shooting|"
    r"seni\s+öldür\w*|seni\s+oldur\w*|onu\s+öldür\w*|hepsini\s+öldür\w*|"
    r"öldüreceğim|oldurecegim|gebertece\w*|geberteyim|"
    r"seni\s+döverim|seni\s+doverim|dövece\w*|bıçaklaya\w*|bicaklaya\w*|"
    r"okulu\s+bas\w*|bomba\s+koya\w*|kurşunlaya\w*|kursunlaya\w*"
    r")\b",
    re.IGNORECASE,
)

# --- Sexual harassment ------------------------------------------------------------------
_SEXUAL_RE = re.compile(
    r"\b(?:"
    r"send\s+nudes|sexting|suck\s+my|blow\s?job|rape|molest|"
    r"seninle\s+yat\w*|soyun\w*|çıplak\s+(?:fotoğraf|resim|foto)|ciplak\s+(?:fotograf|resim|foto)|"
    r"tecavüz|tecavuz|meme(?:lerin|leri)\b|sikiş\w*|sikis\w*|porno"
    r")\b",
    re.IGNORECASE,
)

# --- Profanity / abuse ------------------------------------------------------------------
# Short Turkish stems are deliberately absent when they collide with ordinary words
# ("am", "mal", "got"); the inflected forms below cover the same insults without the
# false-positive cost.
_PROFANITY_RE = re.compile(
    r"\b(?:"
    r"fuck\w*|motherfucker|shit\w*|bullshit|bitch\w*|asshole|arsehole|bastard|cunt|"
    r"dickhead|douchebag|jackass|whore|slut|"
    r"screw\s+you|piss\s+off|shut\s+the\s+fuck\s+up|stfu|"
    r"a[mn]k\b|amina\w*|amına\w*|amcık\w*|amcik\w*|"
    r"sikt[iı]r\w*|sikey\w*|sikik|siktir|"
    r"orospu\w*|kahpe|piç\w*|pic\b|yarra\w*|yarak|"
    r"puşt\w*|pust\b|ibne\w*|pezevenk|gavat|şerefsiz|serefsiz|"
    r"aptal\w*|salak\w*|gerizekal\w*|geri\s+zekal\w*|aptalsın|mal\s+mısın|"
    r"ananı\w*|anan[ıi]n\w*|avradını\w*"
    r")\b",
    re.IGNORECASE,
)


_ORDERED_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (SELF_HARM, _SELF_HARM_RE),
    (HATE, _HATE_SLUR_RE),
    (HATE, _HATE_CONSTRUCTION_RE),
    (VIOLENCE, _VIOLENCE_RE),
    (SEXUAL_HARASSMENT, _SEXUAL_RE),
    (PROFANITY, _PROFANITY_RE),
)


def classify(text: str) -> SafetyVerdict:
    """Classify one incoming question. Returns a SAFE verdict for ordinary academic input."""
    message = (text or "").strip()
    if not message:
        return SafetyVerdict()

    technical_spans = [m.span() for m in _TECHNICAL_EXEMPTIONS.finditer(message)]

    def _inside_technical_span(match: re.Match[str]) -> bool:
        start, end = match.span()
        return any(s <= start and end <= e for s, e in technical_spans)

    for category, pattern in _ORDERED_RULES:
        for match in pattern.finditer(message):
            if category == VIOLENCE and _inside_technical_span(match):
                continue
            return SafetyVerdict(category=category, matched=match.group(0))
    return SafetyVerdict()


# --- Responses --------------------------------------------------------------------------
# One per category per language. They are intentionally different from each other: the point
# of classifying is to respond appropriately, not to print one refusal five times.
_RESPONSES: dict[str, dict[str, str]] = {
    SELF_HARM: {
        "en": (
            "I'm reading something serious in your message, and I don't want to skip past it. "
            "I'm a course advising assistant, so I'm not the right kind of help here — but "
            "please talk to someone who is.\n\n"
            "- **Emergency (Turkey): 112**\n"
            "- **Sabancı University Counseling and Guidance Center** — free and confidential for "
            "students, reachable through the university's student resources page.\n\n"
            "If you'd like, I can go back to your courses whenever you're ready."
        ),
        "tr": (
            "Mesajında ciddi bir şey okuyorum ve bunu geçiştirmek istemiyorum. Ben bir ders "
            "danışmanlığı asistanıyım, bu konuda doğru yardım kaynağı değilim — ama lütfen "
            "doğru kişilerle konuş.\n\n"
            "- **Acil durum (Türkiye): 112**\n"
            "- **Sabancı Üniversitesi Psikolojik Danışmanlık ve Rehberlik Merkezi** — "
            "öğrenciler için ücretsiz ve gizlidir; üniversitenin öğrenci kaynakları sayfasından "
            "ulaşabilirsin.\n\n"
            "Hazır olduğunda derslerine kaldığımız yerden devam edebiliriz."
        ),
    },
    HATE: {
        "en": (
            "I won't engage with that. Content targeting people because of their ethnicity, "
            "religion, gender, or origin has no place in this conversation.\n\n"
            "I'm here for Sabancı University academic advising — courses, curriculum "
            "requirements, graduation status, and schedules."
        ),
        "tr": (
            "Bu içerikle ilgilenmeyeceğim. İnsanları etnik kökeni, dini, cinsiyeti veya "
            "geldiği yer nedeniyle hedef alan ifadelerin bu konuşmada yeri yok.\n\n"
            "Ben Sabancı Üniversitesi akademik danışmanlığı için buradayım: dersler, müfredat "
            "koşulları, mezuniyet durumu ve ders programı."
        ),
    },
    VIOLENCE: {
        "en": (
            "I can't help with anything involving threats or harm to people, and I won't "
            "continue on that topic.\n\n"
            "If you or someone else is in danger, call **112**. Otherwise, I'm glad to help "
            "with your courses, curriculum requirements, or graduation status."
        ),
        "tr": (
            "İnsanlara yönelik tehdit veya zarar içeren hiçbir konuda yardımcı olamam ve bu "
            "konuda devam etmeyeceğim.\n\n"
            "Sen veya bir başkası tehlikedeyse **112**'yi ara. Bunun dışında derslerin, "
            "müfredat koşulların veya mezuniyet durumun için yardımcı olmaktan memnuniyet duyarım."
        ),
    },
    SEXUAL_HARASSMENT: {
        "en": (
            "That's not something I'll respond to. This is a university advising assistant, "
            "and the conversation needs to stay appropriate.\n\n"
            "Ask me about courses, electives, prerequisites, or your graduation requirements "
            "and I'll help right away."
        ),
        "tr": (
            "Buna yanıt vermeyeceğim. Burası bir üniversite danışmanlık asistanı ve konuşmanın "
            "uygun kalması gerekiyor.\n\n"
            "Dersler, seçmeliler, önkoşullar veya mezuniyet koşulların hakkında sorarsan hemen "
            "yardımcı olurum."
        ),
    },
    PROFANITY: {
        "en": (
            "Let's keep this civil — I'll skip the language and go straight to the question.\n\n"
            "I can only help with Sabancı University academic topics: course selection, "
            "curriculum requirements, prerequisites, graduation status, and weekly schedules. "
            "Ask me one of those and I'm on it."
        ),
        "tr": (
            "Bunu nezaket sınırında tutalım — üslubu geçip doğrudan soruna bakayım.\n\n"
            "Yalnızca Sabancı Üniversitesi akademik konularında yardımcı olabiliyorum: ders "
            "seçimi, müfredat koşulları, önkoşullar, mezuniyet durumu ve haftalık ders programı. "
            "Bunlardan birini sorarsan hemen ilgileneyim."
        ),
    },
}


def response_for(category: str, language: str = "tr") -> str:
    """The user-facing reply for a blocked category, in the language the student wrote in."""
    bundle = _RESPONSES.get(category)
    if not bundle:
        bundle = _RESPONSES[PROFANITY]
    return bundle.get(language, bundle["tr"])

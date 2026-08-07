from __future__ import annotations

"""Central, immutable Turkish/English strings for new user-facing features."""

from types import MappingProxyType
from typing import Mapping

from modules.language import ResponseLanguage


_EN = {
    "errors.internal": "Something went wrong while answering. Please try again.",
    "errors.rate_limited": (
        "The language model is temporarily rate limited. Try again in a few minutes; "
        "your profile and course history are unaffected."
    ),
    "limits.request": "You have reached the request limit. Please try again later.",
    "fallback.non_academic": (
        "I can help with academic advising questions about courses, curricula, and planning."
    ),
    "recommendation.ask_interest": (
        "Which area interests you? For example: NLP, Web, Data, Systems, AI, or Security."
    ),
    "recommendation.profile_required": (
        "To recommend a safe current-term plan, I first need your academic profile: major and "
        "curriculum term. Add those details in your profile, then ask again."
    ),
    "confidence.insufficient_evidence": (
        "I could not find enough official evidence to answer safely. Please provide the missing "
        "program or curriculum detail, or confirm this with an academic advisor."
    ),
    "confidence.profile_required": (
        "There is not enough profile or curriculum data to create this recommendation reliably."
    ),
    "confidence.curriculum_unavailable": (
        "There is not enough profile or curriculum data to create this recommendation reliably."
    ),
    "confidence.provider_unavailable": (
        "The answer provider is temporarily unavailable. Please try again later."
    ),
    "confidence.cannot_verify": (
        "I cannot verify this answer against the available official evidence."
    ),
    "confidence.safe_abstention": (
        "I cannot answer that request safely. Please ask an academic advising question directly."
    ),
    "guardrails.input_too_long": (
        "Your message exceeds the safe processing limit. Please send a shorter question."
    ),
    "guardrails.empty_input": "Please enter a question.",
    "guardrails.invalid_control_character": "The message contains invalid characters.",
    "guardrails.secret_extraction": (
        "I cannot share secrets or system instructions. I can help with an academic question."
    ),
    "guardrails.prompt_injection": (
        "I cannot follow instructions that alter system or safety rules. "
        "Please ask your academic question directly."
    ),
    "guardrails.unsafe_output": (
        "The response did not pass the safety check. Please rephrase your question."
    ),
    "guardrails.cross_user_data": "I cannot access or disclose another student's private records.",
    "guardrails.harmful_generation": (
        "I cannot create hateful, discriminatory, or harassing content. "
        "I can help with respectful educational information."
    ),
    "course_reviews.disabled": "Course reviews are not enabled.",
    "chat.instructor_opinion_redirect": (
        "I can't confirm or weigh in on opinions or rumors like that -- subjective comments "
        "about instructors aren't something this system tracks. What I can actually help with "
        "is official information: a course's content, prerequisites, credits, or where it fits "
        "in your curriculum. Ask me about any of that and I'll look it up."
    ),
    "course_reviews.enabled": "Course-only ratings are available under the privacy policy.",
    "course_reviews.consent_required": "Explicit consent is required before submitting a review.",
    "course_reviews.invalid_score": "Each course rating must be a whole number from 1 to 5.",
    "course_reviews.instructor_rating_rejected": "This feature evaluates courses, not instructors.",
    "course_reviews.duplicate": "You have already submitted a review for this course.",
    "course_reviews.pending": "Your review is awaiting moderation.",
    "course_reviews.approved": "Your course review was accepted.",
    "course_reviews.rejected": "The review was not accepted under the course-review data policy.",
    "course_reviews.aggregate_suppressed": (
        "Aggregated results are shown only after at least {minimum} eligible reviews."
    ),
    "course_reviews.private_chat_disabled": (
        "Private chat and course-group imports are not supported. Course reviews require "
        "explicit consent and may evaluate courses only, never instructors."
    ),
    "course_reviews.invalid": "The course review did not pass validation.",
    "course_reviews.deleted": "Your course review was deleted.",
    "course_reviews.not_found": "No course review belonging to you was found.",
    "course_reviews.storage_unavailable": "Course reviews are temporarily unavailable.",
}
_TR = {
    "errors.internal": "Yanıt hazırlanırken bir sorun oluştu. Lütfen tekrar dene.",
    "errors.rate_limited": (
        "Dil modeli geçici olarak yoğun. Birkaç dakika sonra tekrar dene; profilin ve "
        "ders geçmişin etkilenmedi."
    ),
    "limits.request": "İstek sınırına ulaştın. Lütfen daha sonra tekrar dene.",
    "fallback.non_academic": (
        "Dersler, müfredat ve planlama hakkındaki akademik danışmanlık sorularında yardımcı olabilirim."
    ),
    "recommendation.ask_interest": (
        "Hangi alana ilgilisin? Örneğin: NLP, Web, Data, Systems, AI veya Security."
    ),
    "recommendation.profile_required": (
        "Güvenli bir güncel dönem planı önerebilmem için önce akademik profilindeki bölüm ve "
        "müfredat dönemi bilgilerini tamamla; ardından yeniden sor."
    ),
    "confidence.insufficient_evidence": (
        "Güvenli bir yanıt için yeterli resmi kanıt bulamadım. Eksik bölüm veya müfredat "
        "bilgisini paylaş ya da bu konuyu akademik danışmanınla doğrula."
    ),
    "confidence.profile_required": (
        "Bu öneriyi güvenilir biçimde oluşturmak için yeterli profil veya müfredat verisi yok."
    ),
    "confidence.curriculum_unavailable": (
        "Bu öneriyi güvenilir biçimde oluşturmak için yeterli profil veya müfredat verisi yok."
    ),
    "confidence.provider_unavailable": (
        "Yanıt sağlayıcısına geçici olarak ulaşılamıyor. Lütfen daha sonra tekrar dene."
    ),
    "confidence.cannot_verify": (
        "Bu yanıtı mevcut resmi kanıtlarla doğrulayamıyorum."
    ),
    "confidence.safe_abstention": (
        "Bu isteği güvenli biçimde yanıtlayamam. Akademik danışmanlık sorunu doğrudan yazabilirsin."
    ),
    "guardrails.input_too_long": (
        "Mesajın güvenli işlem sınırını aşıyor. Lütfen daha kısa bir soru gönder."
    ),
    "guardrails.empty_input": "Lütfen bir soru yaz.",
    "guardrails.invalid_control_character": "Mesaj geçersiz karakterler içeriyor.",
    "guardrails.secret_extraction": (
        "Gizli anahtarları veya sistem talimatlarını paylaşamam. "
        "Akademik bir konuda yardımcı olabilirim."
    ),
    "guardrails.prompt_injection": (
        "Sistem veya güvenlik kurallarını değiştiren talimatları uygulayamam. "
        "Akademik sorunu doğrudan yazabilirsin."
    ),
    "guardrails.unsafe_output": (
        "Bu yanıt güvenlik kontrolünden geçemedi. Lütfen sorunu yeniden ifade et."
    ),
    "guardrails.cross_user_data": "Başka bir öğrencinin özel kayıtlarına erişemem veya bunları paylaşamam.",
    "guardrails.harmful_generation": (
        "Nefret, ayrımcılık veya taciz içeren içerik üretemem. "
        "Saygılı ve eğitsel bilgiyle yardımcı olabilirim."
    ),
    "course_reviews.disabled": "Ders değerlendirmeleri etkin değil.",
    "chat.instructor_opinion_redirect": (
        "Bu tür bir yorumu ya da söylentiyi ne doğrulayabilirim ne de onaylayabilirim — hocalar "
        "hakkındaki öznel görüşler bu sistemin takip ettiği bir şey değil. Asıl yardımcı "
        "olabileceğim konular resmî bilgiler: bir dersin içeriği, ön koşulları, kredisi veya "
        "müfredattaki yeri gibi. Bunlardan herhangi birini sorarsan bakabilirim."
    ),
    "course_reviews.enabled": "Yalnızca ders puanlama özelliği gizlilik politikasıyla kullanılabilir.",
    "course_reviews.consent_required": "Değerlendirme göndermeden önce açık onay vermen gerekir.",
    "course_reviews.invalid_score": "Her ders puanı 1 ile 5 arasında bir tam sayı olmalıdır.",
    "course_reviews.instructor_rating_rejected": "Bu özellik öğretim üyesini değil, dersi değerlendirir.",
    "course_reviews.duplicate": "Bu ders için daha önce değerlendirme gönderdin.",
    "course_reviews.pending": "Değerlendirmen moderasyon bekliyor.",
    "course_reviews.approved": "Ders değerlendirmen kabul edildi.",
    "course_reviews.rejected": "Değerlendirme, ders değerlendirme veri politikasına uygun bulunmadı.",
    "course_reviews.aggregate_suppressed": (
        "Toplu sonuçlar yalnızca en az {minimum} uygun değerlendirmeden sonra gösterilir."
    ),
    "course_reviews.private_chat_disabled": (
        "Özel sohbet ve ders grubu içe aktarımları desteklenmiyor. Ders değerlendirmeleri açık "
        "onay gerektirir ve yalnızca dersi değerlendirebilir; öğretim üyesini değerlendiremez."
    ),
    "course_reviews.invalid": "Ders değerlendirmesi doğrulamadan geçmedi.",
    "course_reviews.deleted": "Ders değerlendirmen silindi.",
    "course_reviews.not_found": "Sana ait bir ders değerlendirmesi bulunamadı.",
    "course_reviews.storage_unavailable": "Ders değerlendirmeleri geçici olarak kullanılamıyor.",
}

MESSAGES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "en": MappingProxyType(_EN),
        "tr": MappingProxyType(_TR),
    }
)


def normalize_language(value: str | None, *, default: ResponseLanguage = "en") -> ResponseLanguage:
    candidate = str(value or "").strip().lower()
    return candidate if candidate in {"tr", "en"} else default


def message(key: str, language: str | None = "en", **values: object) -> str:
    """Return a localized message and fail loudly for unknown resource keys."""

    selected = normalize_language(language)
    template = MESSAGES[selected].get(key) or MESSAGES["en"].get(key)
    if template is None:
        raise KeyError(f"Unknown localization key: {key}")
    return template.format(**values)

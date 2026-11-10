"""Deterministic roll-number normalisation, applied to the transcript before the LLM sees it.

Prisma writes spoken roll numbers as words, in Latin or Devanagari script:
  "S one zero zero six", "एस वन ज़ीरो ज़ीरो थ्री", "S-एक ज़ीरो ज़ीरो चार", "S one thousand two", "S. One thousand four"
In benchmark v1 the LLM mis-converted some of these (S1002 -> S1004, or missed them entirely).
This rewrites any such span to "S1002" etc. and leaves everything else untouched.
"""
import re
import unicodedata


def _k(w):
    return unicodedata.normalize("NFC", w.lower())

DIGIT = {
    # English
    "zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9,
    # English in Devanagari
    "ज़ीरो": 0, "जीरो": 0, "वन": 1, "टू": 2, "थ्री": 3, "फोर": 4, "फ़ोर": 4, "फाइव": 5, "फ़ाइव": 5,
    "सिक्स": 6, "सेवन": 7, "एट": 8, "एइट": 8, "नाइन": 9,
    # Hindi
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "छः": 6, "छे": 6,
    "सात": 7, "आठ": 8, "नौ": 9,
    # Hindi in Latin
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhah": 6, "che": 6,
    "saat": 7, "aath": 8, "nau": 9,
}
TEEN_TENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
             "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
             "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
             "टेन": 10}
THOUSAND = {"thousand", "थाउजेंड", "थाउज़ेंड", "हज़ार", "हजार", "hazaar", "hazar"}
S_TOKEN = {"s", "एस", "ऐस"}
DIGIT = {_k(k): v for k, v in DIGIT.items()}
TEEN_TENS = {_k(k): v for k, v in TEEN_TENS.items()}
THOUSAND = {_k(k) for k in THOUSAND}
S_TOKEN = {_k(k) for k in S_TOKEN}

_TOK = re.compile("[A-Za-z]+|[ऀ-ॣ०-ॿ]+|[0-9]+")  # Devanagari minus danda (U+0964/5)


def _parse_number(words):
    """Parse a word sequence that follows 'S'. Returns (value, n_words_used) or (None, 0)."""
    i, digits = 0, []
    # digit-by-digit: "one zero zero four"
    while i < len(words) and words[i] in DIGIT and len(digits) < 4:
        digits.append(DIGIT[words[i]]); i += 1
    if len(digits) == 4:
        return int("".join(map(str, digits))), i
    # "one thousand [and] two" / "one thousand ten" / "ek hazaar chaar"
    if len(digits) == 1 and i < len(words) and words[i] in THOUSAND:
        val, i = digits[0] * 1000, i + 1
        if i < len(words) and words[i] == "and":
            i += 1
        if i < len(words) and words[i] in TEEN_TENS:
            val += TEEN_TENS[words[i]]; i += 1
            if val % 10 == 0 and i < len(words) and words[i] in DIGIT and DIGIT[words[i]] > 0:
                val += DIGIT[words[i]]; i += 1
        elif i < len(words) and words[i] in DIGIT:
            val += DIGIT[words[i]]; i += 1
        return val, i
    return None, 0


def normalize_roll_numbers(text: str) -> str:
    toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer(text)]
    out, last, k = [], 0, 0
    while k < len(toks):
        w, s, e = toks[k]
        wl = _k(w)
        # "S1004" already fine; "S" + "1004"
        if wl in S_TOKEN and k + 1 < len(toks) and toks[k + 1][0].isdigit() and len(toks[k + 1][0]) == 4:
            out.append(text[last:s] + "S" + toks[k + 1][0]); last = toks[k + 1][2]; k += 2; continue
        if wl in S_TOKEN:
            words = [_k(t[0]) for t in toks[k + 1:k + 8]]
            val, n = _parse_number(words)
            if val is not None and 1000 <= val <= 9999:
                out.append(text[last:s] + f"S{val}"); last = toks[k + n][2]; k += n + 1; continue
        k += 1
    out.append(text[last:])
    return "".join(out)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    cases = {
        "S one zero zero two का scholarship status और fees दोनों बताओ।": "S1002",
        "S one thousand two की fees कितनी due है?": "S1002",
        "roll number S. One thousand four is my fees paid.": "S1004",
        "मेरा application कहाँ तक पहुँचा, roll number एस वन ज़ीरो ज़ीरो थ्री।": "S1003",
        "मेरा रोल नंबर S-एक ज़ीरो ज़ीरो चार है": "S1004",
        "भैया, S one zero zero six की fees due है क्या?": "S1006",
        "roll number S one thousand ten, kya paisa aaya": "S1010",
        "मेरा roll number एस one zero zero four है": "S1004",
        "S1007 scholarship reject kyun hui": "S1007",
        "fees kitni baaki hai?": None,
    }
    bad = 0
    for t, exp in cases.items():
        got = normalize_roll_numbers(t)
        ok = (exp in got) if exp else got == t
        bad += not ok
        print("OK " if ok else "BAD", got)
    print("failures:", bad)

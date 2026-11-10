"""Tools over synthetic data (data/students.json). All data is synthetic."""
import json

from . import config

_DB = None


def _db():
    global _DB
    if _DB is None:
        _DB = json.loads(config.DATA_PATH.read_text(encoding="utf-8"))
    return _DB


# Numbers are rendered to spoken Hinglish in code, not by the LLM: in v1/v2 testing the LLM said
# "aath hazaar" (8,000) for 18,000. Tools return *_spoken fields that the LLM copies verbatim.
_H = ("shunya ek do teen chaar paanch chhah saat aath nau das gyaarah baarah terah chaudah pandrah solah "
      "satrah atharah unnees bees ikkees baaees teis chaubees pachchees chhabbees sattaaees atthaaees untees "
      "tees ikattees battees taintees chautees paintees chhattees saintees adtees untaalees chaalees iktaalees "
      "bayaalees taintaalees chavaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas ikyaavan "
      "baavan tirpan chauvan pachpan chhappan sattaavan atthaavan unsath saath iksath baasath tirsath chausath "
      "painsath chhiyaasath sadsath adsath unhattar sattar ikhattar bahattar tihattar chauhattar pachhattar "
      "chhihattar sathattar athhattar unyaasi assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi "
      "sataasi athaasi navaasi nabbe ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve "
      "atthaanve ninyaanve").split()
_MONTHS = "January February March April May June July August September October November December".split()
_EN = "zero one two three four five six seven eight nine".split()


def hindi_number(n: int) -> str:
    if n == 0:
        return "shunya"
    parts = []
    for size, word in ((100000, "lakh"), (1000, "hazaar"), (100, "sau")):
        if n >= size:
            parts.append(f"{_H[n // size]} {word}")
            n %= size
    if n:
        parts.append(_H[n])
    return " ".join(parts)


def rupees_spoken(n: int) -> str:
    return "zero rupaye" if not n else f"{hindi_number(n)} rupaye"


def date_spoken(iso):
    if not iso:
        return None
    y, m, d = (int(x) for x in iso.split("-"))
    return f"{hindi_number(d)} {_MONTHS[m - 1]}"


def roll_spoken(roll: str) -> str:
    return "S " + " ".join(_EN[int(c)] for c in roll[1:])


def get_scholarship_status(roll_no: str) -> dict:
    roll = roll_no.strip().upper()
    s = _db().get(roll)
    if not s:
        return {"error": "roll number not found", "roll_no_spoken": roll_spoken(roll) if roll[1:].isdigit() else roll}
    sc = s["scholarship"]
    return {"roll_no": roll, "roll_no_spoken": roll_spoken(roll), **sc,
            "amount_spoken": rupees_spoken(sc["amount"])}


def get_fee_status(roll_no: str) -> dict:
    roll = roll_no.strip().upper()
    s = _db().get(roll)
    if not s:
        return {"error": "roll number not found", "roll_no_spoken": roll_spoken(roll) if roll[1:].isdigit() else roll}
    f = s["fees"]
    return {"roll_no": roll, "roll_no_spoken": roll_spoken(roll), **f,
            "due_spoken": rupees_spoken(f["due"]), "due_date_spoken": date_spoken(f["due_date"])}


FUNCS = {"get_scholarship_status": get_scholarship_status, "get_fee_status": get_fee_status}

_ROLL = {"type": "object", "properties": {"roll_no": {"type": "string", "description": "e.g. S1004"}},
         "required": ["roll_no"]}
SPECS = [
    {"name": "get_scholarship_status",
     "description": "Scholarship status for a student: status, amount, next_step.", "schema": _ROLL},
    {"name": "get_fee_status",
     "description": "Fee status for a student: due amount and due_date (due 0 means fully paid).", "schema": _ROLL},
]

SYSTEM_PROMPT = """You are a voice assistant for college students asking about scholarship and fee status.
Reply in short, natural Hinglish (Hindi in Roman script mixed with English), max 2 short sentences.
Identify the student by roll number: the letter S followed by four digits. The text comes from speech
recognition and may be in Devanagari, with the number spoken as words digit by digit ('S two zero three
seven', 'एस टू ज़ीरो थ्री सेवन', 'S-दो शून्य तीन सात') or as a whole number ('S two thousand thirty seven',
'S. Two thousand thirty-seven'); all of these mean S2037 (an example only, not a real student). Convert
exactly the number that was said, never a different one. If the user did not give a roll number, ask for it once instead of guessing.
If the user asks about both scholarship and fees, call both tools and answer both. When you have it, say the roll number back and call the right tool.
Write for speech: no digits, no symbols, no markdown. Tool results contain *_spoken fields (roll_no_spoken,
amount_spoken, due_spoken, due_date_spoken): copy those words exactly for roll numbers, amounts and dates,
never convert numbers yourself. All data is synthetic demo data."""

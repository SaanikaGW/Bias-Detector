#!/usr/bin/env python3
"""One-time generator for detection/patterns.json (Layer 1 rule data).
NOTE: optional regeneration tool. detection/patterns.json is the runtime
source of truth — if you hand-edit the JSON, port changes here before rerunning
(this script overwrites the JSON). Usage: python ml/gen_patterns.py detection/patterns.json
The JSON file it produces is the editable source of truth afterwards.
Severity: 1=low, 2=medium, 3=high. conf = base confidence when rule fires.
amb=True  -> ambiguous term: Layer 2 contextual classifier adjudicates the sentence.
veto      -> sentence-level regexes that suppress the hit outright (clear benign use).
"""
import json, re, sys

CATEGORIES = {
    "masculine_coded": {
        "label": "Masculine-coded wording",
        "weight": 0.8,
        "research": "Research on gender-coded job ads (Gaucher, Friesen & Kay, 2011, Journal of Personality and Social Psychology) found that ads dense in masculine-coded words made women rate the job as less appealing and feel less belonging — without changing whether they felt able to do the job. The wording, not the work, drives self-selection out.",
        "impact": "Masculine-coded trait words signal a male-typed culture, so qualified women are measurably less likely to apply — the posting filters people before skills are ever compared.",
    },
    "feminine_coded": {
        "label": "Feminine-coded wording",
        "weight": 0.5,
        "research": "The same 2011 gender-coding research found feminine-coded wording has a weaker deterrent effect on men than masculine coding has on women, but stacking stereotypically feminine trait words still frames a role by personality type rather than skill, and can signal lower status or pay for the role.",
        "impact": "Heavily feminine-coded trait language types the role by personality rather than skill and can subtly signal lower status, narrowing who pictures themselves in it.",
    },
    "gendered_language": {
        "label": "Gendered pronouns & titles",
        "weight": 1.0,
        "research": "Studies of gendered job titles and pronouns going back to Bem & Bem's 1970s experiments show that gendered wording ('salesman', 'he will…') measurably reduces applications from the excluded gender; many jurisdictions' equal-employment guidance now explicitly recommends neutral titles.",
        "impact": "A gendered title or pronoun tells half your candidate pool the default hire isn't them — the single most direct and most fixable form of gender coding.",
    },
    "stereotype": {
        "label": "Gender & leadership stereotypes",
        "weight": 0.85,
        "research": "Research on agentic/communal stereotypes (e.g., Eagly's role-congruity work) shows trait demands like 'dominant' or 'commanding' evoke a male-typed prototype of leadership, while women who display the same traits are penalized as 'abrasive' — so such wording deters women twice over.",
        "impact": "Stereotyped trait demands describe a persona, not a competency, and the persona is gender-typed — qualified candidates who don't match the persona self-select out.",
    },
    "caregiver_bias": {
        "label": "Caregiver / family-responsibility bias",
        "weight": 0.9,
        "research": "Motherhood-penalty research (Correll, Benard & Paik, 2007, American Journal of Sociology) shows mothers are rated less competent and committed from identical materials. Wording that demands 'total dedication' or penalizes career gaps imports that bias into the ad itself and disproportionately screens out women, who still carry most caregiving load.",
        "impact": "Demands framed around unlimited availability or unbroken work history screen for absence of caregiving duties, not ability to do the job — disproportionately excluding women and parents.",
    },
    "age_coded": {
        "label": "Age-coded wording",
        "weight": 0.7,
        "research": "Age-discrimination audit studies (e.g., Neumark, Burn & Button's large resume-correspondence study) show older applicants — especially older women — receive markedly fewer callbacks; phrases like 'digital native' are flagged by the EEOC-adjacent guidance as proxies for youth.",
        "impact": "Youth-proxy wording deters experienced applicants and creates legal exposure, since it functions as an age screen unrelated to actual job requirements.",
    },
    "appearance_bias": {
        "label": "Appearance bias",
        "weight": 0.7,
        "research": "Appearance requirements are applied and policed more heavily against women (grooming-code case law and lookism research consistently show gendered enforcement), and are rarely genuine job requirements.",
        "impact": "Appearance demands shift evaluation from skills to looks, are enforced more harshly against women, and discourage anyone who suspects they won't match an unstated 'look'.",
    },
    "exclusionary": {
        "label": "Exclusionary wording",
        "weight": 0.75,
        "research": "Language-requirement guidance (including from the EEOC) distinguishes fluency needed for the job from 'native speaker' demands, which exclude fully proficient speakers by origin. Culture-signal phrases like 'work hard, play hard' correlate with male-typed, monoculture workplaces in organizational research on culture-fit hiring.",
        "impact": "These phrases exclude on identity or lifestyle rather than capability — proficient non-native speakers, caregivers, and anyone outside the after-hours social culture read them as 'not for you'.",
    },
    "qualification_inflation": {
        "label": "Qualification inflation",
        "weight": 0.6,
        "research": "A widely cited Hewlett-Packard internal finding (popularized by Mohr, Harvard Business Review, 2014) and later LinkedIn behavioral data indicate women tend to apply only when they meet close to all listed requirements, while men apply at much lower thresholds — so inflated 'must-have' lists disproportionately shrink the female applicant pool.",
        "impact": "Every non-essential 'must' removes more qualified women than men from your pipeline, because application thresholds differ by gender.",
    },
}

# (id, regex, category, severity, conf, ambiguous, veto_list, why, alternatives, gain)
P = []
def e(id, rx, cat, sev, conf, amb=False, veto=None, why="", alt=None, gain=""):
    P.append(dict(id=id, pattern=rx, category=cat, severity=sev, confidence=conf,
                  ambiguous=amb, veto=veto or [], why=why, alternatives=alt or [], gain=gain))

W_PERSON = r"(personalit|person\b|people|candidate|individual|applicant|attitude|character|nature|demeanor|you\b|team\s+member)"
W_BUSINESS = r"(strateg|pricing|market|growth|target|timeline|deadline|expansion|plan\b|plans\b|campaign|goal)"

# ── masculine_coded ───────────────────────────────────────────────────────────
e("mc-aggressive", r"\baggressive(?:ly|ness)?\b", "masculine_coded", 2, 0.65, amb=True, veto=[W_BUSINESS],
  why="'Aggressive' as a personal trait is one of the strongest masculine-coded words; describing strategy or targets as aggressive is fine.",
  alt=["proactive", "determined", "results-focused"], gain="Keeps the drive, drops the gender coding.")
e("mc-dominant", r"\bdominant\b|\bdominate\b|\bdomination\b", "masculine_coded", 2, 0.7, amb=True, veto=[r"market[-\s]dominant position"],
  why="'Dominant/dominate' is masculine-coded trait language.",
  alt=["leading", "influential", "market-leading"], gain="Same ambition, wider appeal.")
e("mc-competitive-person", r"\bcompetitive\b", "masculine_coded", 1, 0.55, amb=True,
  veto=[r"competitive\s+(salary|pay|compensation|benefits|package|rate|wage|pricing|price|market|landscape|advantage|analysis)",
        r"(pricing|prices?|rates?|offer(?:ing)?s?)\s+(is|are|remains?)\s+(highly\s+)?competitive",
        r"competitive\s+(within|in)\s+the\s+market"],
  why="'Competitive' as a required personality trait is masculine-coded; 'competitive salary' is not.",
  alt=["motivated", "goal-oriented"], gain="Keeps the performance bar without the trait demand.")
e("mc-rockstar", r"\brock[\s\-]?star\b", "masculine_coded", 3, 0.95,
  why="'Rockstar' is hyper-masculine-coded job jargon and signals a bravado culture.",
  alt=["skilled", "high-performing", "experienced"], gain="Research-linked masculine jargon removed; requirement unchanged.")
e("mc-ninja", r"\bninja\b", "masculine_coded", 3, 0.95, why="'Ninja' is masculine-coded jargon.",
  alt=["expert", "specialist"], gain="Plain skill language broadens applicants.")
e("mc-guru", r"\bguru\b", "masculine_coded", 2, 0.85, why="'Guru' is coded jargon that also borrows a religious title.",
  alt=["expert", "authority"], gain="Clearer and more inclusive.")
e("mc-wizard", r"\bwizard\b", "masculine_coded", 2, 0.85, why="'Wizard' is masculine-coded jargon.",
  alt=["expert", "specialist"], gain="Plain language, same bar.")
e("mc-superstar", r"\bsuper\s?star\b", "masculine_coded", 2, 0.8, why="'Superstar' is coded bravado jargon.",
  alt=["top performer", "excellent"], gain="Describes the standard, not a persona.")
e("mc-superhero", r"\bsuper\s?hero\b", "masculine_coded", 2, 0.8, why="'Superhero' framing is coded and inflates the role.",
  alt=["dedicated professional"], gain="Sets realistic, inclusive expectations.")
e("mc-warrior", r"\bwarrior\b", "masculine_coded", 3, 0.9, why="Combat metaphors are strongly masculine-coded.",
  alt=["advocate", "committed professional"], gain="Removes combat framing.")
e("mc-alpha", r"\balpha\b", "masculine_coded", 3, 0.9,
  veto=[r"alpha\s+(test|version|release|build|stage)", r"alpha\s+and\s+beta"],
  why="'Alpha' personality language is explicitly male-typed dominance framing.",
  alt=["confident leader"], gain="Leadership without dominance coding.")
e("mc-hustle", r"\bhustl(?:e|er|ing)\b", "masculine_coded", 2, 0.75,
  why="'Hustle' culture language is masculine-coded and signals overwork.",
  alt=["dedicated", "hard-working"], gain="Work ethic stated without overwork signal.")
e("mc-crush", r"\bcrush(?:ing)?\s+(it|goals|targets|quota|the\s+competition)\b", "masculine_coded", 2, 0.8,
  why="Violent achievement metaphors ('crush it') are masculine-coded.",
  alt=["exceed goals", "deliver outstanding results"], gain="Same ambition, neutral phrasing.")
e("mc-killer", r"\bkiller\s+(instinct|attitude|drive)\b|\bkill\s+it\b", "masculine_coded", 3, 0.85,
  why="'Killer instinct' is violent, masculine-coded trait language.",
  alt=["strong drive to succeed"], gain="Drive without violence metaphor.")
e("mc-fearless", r"\bfearless(?:ly|ness)?\b", "masculine_coded", 2, 0.75,
  why="'Fearless' is masculine-coded trait language.", alt=["bold", "willing to take initiative"],
  gain="Initiative stated inclusively.")
e("mc-assertive", r"\bassertive(?:ly|ness)?\b", "masculine_coded", 1, 0.6, amb=True,
  why="'Assertive' is masculine-coded; women displaying assertiveness are also judged more harshly, a double bind.",
  alt=["confident communicator", "comfortable voicing ideas"], gain="Communication skill without the double bind.")
e("mc-decisive", r"\bdecisive(?:ly|ness)?\b", "masculine_coded", 1, 0.5, amb=True,
  why="'Decisive' is mildly masculine-coded when stacked with other agentic traits.",
  alt=["sound judgment", "makes timely decisions"], gain="Behavior instead of persona.")
e("mc-driven-person", r"\bhard[\s\-]driving\b|\bhard[\s\-]charging\b", "masculine_coded", 2, 0.8,
  why="'Hard-charging/hard-driving' is masculine-coded intensity language.",
  alt=["highly motivated"], gain="Motivation without intensity coding.")
e("mc-outspoken", r"\boutspoken\b", "masculine_coded", 1, 0.6,
  why="'Outspoken' is masculine-coded and penalized in women (double bind).",
  alt=["shares ideas openly"], gain="Behavioral phrasing avoids the double bind.")
e("mc-headstrong", r"\bheadstrong\b", "masculine_coded", 2, 0.7, why="'Headstrong' is masculine-coded trait language.",
  alt=["persistent"], gain="Persistence without coding.")
e("mc-relentless", r"\brelentless(?:ly)?\b", "masculine_coded", 1, 0.6, amb=True,
  why="'Relentless' is masculine-coded intensity language.", alt=["persistent", "dedicated"],
  gain="Same persistence, softer coding.")
e("mc-cutthroat", r"\bcut[\s\-]?throat\b", "masculine_coded", 3, 0.9,
  why="'Cutthroat' signals a combative, male-typed culture.", alt=["high-performing", "fast-moving"],
  gain="Pace and standards without combat culture.")
e("mc-thickskin", r"\bthick[\s\-]skin(?:ned)?\b", "masculine_coded", 2, 0.8,
  why="'Thick skin required' signals a harsh culture and deters under-represented candidates who already face more criticism.",
  alt=["comfortable giving and receiving direct feedback"], gain="Feedback culture stated constructively.")
e("mc-trenches", r"\bin\s+the\s+trenches\b", "masculine_coded", 1, 0.7,
  why="War metaphor; mildly masculine-coded.", alt=["hands-on"], gain="Plain description of the work.")
e("mc-battle", r"\bbattle[\s\-]tested\b|\bwar\s+room\b", "masculine_coded", 1, 0.7,
  why="Combat metaphors are masculine-coded.", alt=["proven under pressure"], gain="Same signal, neutral image.")
e("mc-gogetter", r"\bgo[\s\-]getter\b", "masculine_coded", 1, 0.6,
  why="'Go-getter' is mildly masculine-coded persona language.", alt=["self-motivated"], gain="Behavioral phrasing.")
e("mc-highoctane", r"\bhigh[\s\-]octane\b", "masculine_coded", 1, 0.7,
  why="'High-octane' is masculine-coded intensity jargon.", alt=["fast-paced"], gain="Pace without jargon.")
e("mc-stud", r"\bstud\b|\bbeast\s+mode\b|\bbeast\b", "masculine_coded", 3, 0.85, veto=[r"beast\s+of\s+burden"],
  why="'Stud/beast' is explicitly male-typed slang.", alt=["top performer"], gain="Professional and inclusive.")
e("mc-boys-club", r"\bone\s+of\s+the\s+boys\b|\bboys[’']?\s+club\b", "masculine_coded", 3, 0.95,
  why="Directly frames the team as male.", alt=["a close-knit team"], gain="Team culture without gender framing.")
e("mc-swagger", r"\bswagger\b", "masculine_coded", 2, 0.75, why="'Swagger' is masculine-coded persona language.",
  alt=["confidence"], gain="Confidence without persona.")
e("mc-firepower", r"\bfire\s?power\b", "masculine_coded", 1, 0.7, why="Weapon metaphor; masculine-coded.",
  alt=["capability", "capacity"], gain="Neutral capability language.")
e("mc-sharp-elbows", r"\bsharp[\s\-]elbow(?:s|ed)?\b", "masculine_coded", 2, 0.85,
  why="'Sharp-elbowed' celebrates combative workplace politics — masculine-coded trait language.",
  alt=["skilled at navigating complex organizations"], gain="Same skill, no combat framing.")

# ── feminine_coded ────────────────────────────────────────────────────────────
e("fc-nurturing", r"\bnurtur(?:e|ing)\b", "feminine_coded", 2, 0.7, amb=True,
  veto=[r"nurtur\w*\s+(talent|leads|pipeline|relationships|accounts|growth|culture)"],
  why="'Nurturing' as a required trait is feminine-coded; nurturing talent or client relationships as an activity is fine.",
  alt=["supportive", "invested in team growth"], gain="Activity framing instead of persona.")
e("fc-bubbly", r"\bbubbly\b", "feminine_coded", 3, 0.9,
  why="'Bubbly personality' is feminine-coded and often a proxy for hiring young women into front-of-house roles.",
  alt=["friendly and professional"], gain="Service standard without persona proxy.")
e("fc-sweet", r"\bsweet[\s\-](personality|demeanor|nature|tempered|natured)\b", "feminine_coded", 3, 0.9,
  why="'Sweet' as a job requirement is feminine-coded persona language.", alt=["courteous"], gain="Professional standard.")
e("fc-soothing", r"\bsoothing\s+(phone\s+)?(manner|voice|presence)\b", "feminine_coded", 2, 0.85,
  why="'Soothing voice/manner' is feminine-coded persona language historically used to type roles female.",
  alt=["calm, professional phone manner"], gain="Skill-based standard.")
e("fc-cheerful", r"\bcheerful\s+(demeanor|personality|disposition)\b", "feminine_coded", 2, 0.75,
  why="Demanded cheerfulness is feminine-coded emotional labor.", alt=["positive, professional manner"],
  gain="Professional standard without emotional-labor demand.")
e("fc-gentle", r"\bgentle\s+(manner|nature|personality|touch)\b", "feminine_coded", 2, 0.7,
  why="'Gentle nature' is feminine-coded trait language.", alt=["patient", "considerate"], gain="Skill-based phrasing.")
e("fc-softspoken", r"\bsoft[\s\-]spoken\b", "feminine_coded", 2, 0.75,
  why="'Soft-spoken' is feminine-coded and irrelevant to competence.", alt=["clear communicator"], gain="Competence-based.")
e("fc-motherhen", r"\bmother\s+hen\b|\boffice\s+mom\b|\bden\s+mother\b", "feminine_coded", 3, 0.95,
  why="Explicitly casts the role as maternal.", alt=["team coordinator", "operations lead"], gain="Role title reflects the actual work.")
e("fc-empathetic", r"\bempath(?:y|etic|ic)\b", "feminine_coded", 1, 0.45, amb=True,
  why="Empathy is a legitimate skill; flag only when stacked as a persona requirement alongside other feminine-coded traits.",
  alt=["strong listening skills"], gain="Skill framing (often fine as-is).")
e("fc-warm", r"\bwarm\s+(?:and\s+\w+\s+)?(personality|demeanor|presence)\b", "feminine_coded", 2, 0.7,
  why="'Warm personality' as a requirement is feminine-coded.", alt=["welcoming and professional"], gain="Standard, not persona.")
e("fc-pleasant", r"\bpleasant\s+(personality|demeanor|manner|voice)\b", "feminine_coded", 2, 0.7,
  why="'Pleasant personality/voice' is feminine-coded and historically used to type roles female.",
  alt=["professional communication style"], gain="Skill-based requirement.")
e("fc-bubbly2", r"\bpeppy\b|\bperky\b", "feminine_coded", 3, 0.85,
  why="'Peppy/perky' is feminine-coded persona slang.", alt=["energetic and professional"], gain="Neutral energy.")
e("fc-caring-nature", r"\bcaring\s+(nature|personality|disposition)\b", "feminine_coded", 1, 0.6, amb=True,
  why="'Caring nature' is feminine-coded when demanded as personality; care skills for care roles are legitimate.",
  alt=["committed to patient/client wellbeing"], gain="Ties care to the job's real duty.")

# ── gendered_language: pronouns & titles ─────────────────────────────────────
for id_, rx, why, alt in [
    ("gl-he-pronoun", r"\bhe\s+(will|must|should|is\s+expected|has|manages|leads|oversees)\b",
     "Male pronoun assumes the hire is a man.", ["they will", "the successful candidate will"]),
    ("gl-she-pronoun", r"\bshe\s+(will|must|should|is\s+expected|has|manages|leads|oversees)\b",
     "Female pronoun assumes the hire is a woman.", ["they will", "the successful candidate will"]),
    ("gl-his", r"\bhis\s+(team|responsibilities|duties|role|performance|reports)\b",
     "Male possessive pronoun genders the role.", ["their"]),
    ("gl-her", r"\bher\s+(team|responsibilities|duties|role|performance|reports)\b",
     "Female possessive pronoun genders the role.", ["their"]),
    ("gl-heshe", r"\bhe/she\b|\bs/he\b|\bhe\s+or\s+she\b",
     "'He/she' excludes non-binary candidates; 'they' is simpler and inclusive.", ["they"]),
    ("gl-chairman", r"\bchairman\b", "Gendered title.", ["chairperson", "chair"]),
    ("gl-chairwoman", r"\bchairwoman\b", "Gendered title.", ["chairperson", "chair"]),
    ("gl-salesman", r"\bsales(?:man|men)\b", "Gendered title.", ["salesperson", "sales representative"]),
    ("gl-saleswoman", r"\bsales(?:woman|women)\b", "Gendered title.", ["salesperson", "sales representative"]),
    ("gl-businessman", r"\bbusiness(?:man|men)\b", "Gendered title.", ["business professional"]),
    ("gl-businesswoman", r"\bbusiness(?:woman|women)\b", "Gendered title.", ["business professional"]),
    ("gl-spokesman", r"\bspokes(?:man|men)\b", "Gendered title.", ["spokesperson"]),
    ("gl-spokeswoman", r"\bspokes(?:woman|women)\b", "Gendered title.", ["spokesperson"]),
    ("gl-foreman", r"\bfore(?:man|men)\b", "Gendered title.", ["supervisor", "site lead"]),
    ("gl-mailman", r"\bmail(?:man|men)\b", "Gendered title.", ["mail carrier"]),
    ("gl-policeman", r"\bpolice(?:man|men)\b", "Gendered title.", ["police officer"]),
    ("gl-fireman", r"\bfire(?:man|men)\b", "Gendered title.", ["firefighter"]),
    ("gl-stewardess", r"\bsteward(?:ess|esses)\b", "Gendered title.", ["flight attendant"]),
    ("gl-waitress", r"\bwaitress(?:es)?\b", "Gendered title.", ["server"]),
    ("gl-hostess", r"\bhostess(?:es)?\b", "Gendered title.", ["host"]),
    ("gl-handyman", r"\bhandy(?:man|men)\b", "Gendered title.", ["maintenance technician"]),
    ("gl-craftsman", r"\bcrafts(?:man|men)\b", "Gendered title.", ["artisan", "craftsperson"]),
    ("gl-draftsman", r"\bdrafts(?:man|men)\b", "Gendered title.", ["drafter"]),
    ("gl-middleman", r"\bmiddle(?:man|men)\b", "Gendered term.", ["intermediary"]),
    ("gl-tradesman", r"\btrades(?:man|men)\b", "Gendered title.", ["tradesperson", "skilled trades professional"]),
    ("gl-journeyman", r"\bjourney(?:man|men)\b", "Gendered title.", ["journey-level professional"]),
    ("gl-repairman", r"\brepair(?:man|men)\b", "Gendered title.", ["repair technician"]),
    ("gl-manpower", r"\bman\s?power\b", "Gendered term for workforce.", ["workforce", "staffing"]),
    ("gl-manhours", r"\bman[\s\-]?hours?\b", "Gendered unit of work.", ["person-hours", "work hours"]),
    ("gl-mankind", r"\bmankind\b", "Gendered term.", ["humanity", "people"]),
    ("gl-manmade", r"\bman[\s\-]?made\b", "Gendered term.", ["artificial", "manufactured"]),
    ("gl-rightman", r"\bright\s+man\s+for\s+the\s+job\b", "Assumes the hire is a man.", ["right person for the job"]),
    ("gl-careergirl", r"\bcareer\s+girl\b|\boffice\s+girl\b|\bgirl\s+friday\b", "Diminishing gendered term.", ["professional", "administrative assistant"]),
    ("gl-guys", r"\bteam\s+of\s+guys\b|\bthe\s+guys\b|\bour\s+guys\b", "'Guys' frames the team as male.", ["the team", "our people"]),
    ("gl-workmanship", r"\bworkmanship\b", "Gendered term.", ["craftsmanship quality", "quality of work"]),
    ("gl-freshman", r"\bfreshman\b", "Gendered term.", ["first-year"]),
    ("gl-waiter-only", r"\bwaiter\s+wanted\b", "Gendered title in hiring context.", ["server wanted"]),
]:
    e(id_, rx, "gendered_language", 3 if "preferred" in id_ else 2, 0.95, why=why, alt=alt,
      gain="Neutral wording keeps the full candidate pool.")

e("gl-male-preferred", r"\b(male|female)s?\s+(candidates?\s+)?(only|preferred|desired)\b|\bprefer\s+(male|female)s?\b",
  "gendered_language", 3, 0.99,
  why="Explicit gender preference — likely unlawful in most jurisdictions as well as exclusionary.",
  alt=["remove the gender requirement entirely"], gain="Removes explicit (and likely illegal) exclusion.")
e("gl-gendered-role-adj", r"\b(male|female)\s+(sales\s+)?(candidates?|applicants?|staff|employees?|workers?|representatives?|assistants?|engineers?|nurses?|attendants?|receptionists?|managers?)\b",
  "gendered_language", 3, 0.97,
  why="Specifying the gender of the hire ('male sales representative') is explicit gender screening.",
  alt=["drop the gender adjective"], gain="Removes explicit (and likely illegal) exclusion.")
e("gl-delivery-boy", r"\b(delivery|office|shop|paper|errand)\s+(boy|girl)\b", "gendered_language", 3, 0.95,
  why="Gendered (and diminishing) job title.", alt=["courier", "assistant"],
  gain="Neutral title keeps the full candidate pool.")
e("gl-gal-friday", r"\bgal\s+friday\b", "gendered_language", 3, 0.95,
  why="Gendered, diminishing role framing.", alt=["administrative assistant"],
  gain="Professional, neutral title.")
e("gl-right-guy", r"\bright\s+(guy|gal)\s+for\s+(the|this)\s+(job|role|position)\b", "gendered_language", 2, 0.9,
  why="Assumes the hire's gender.", alt=["right person for the role"], gain="Neutral wording.")
e("gl-suits-a-man", r"\b(suits?|needs?|for)\s+a\s+(man|woman)\s+(of|who|with)\b", "gendered_language", 3, 0.9,
  why="Frames the role as belonging to one gender.", alt=["suits a professional with"],
  gain="Same standard, no gender assumption.")
e("gl-young-man-woman", r"\byoung\s+(man|woman|men|women|lady|ladies|gentleman|gentlemen)\b", "gendered_language", 3, 0.95,
  why="Combines gender and age preference.", alt=["motivated candidates"], gain="Removes double exclusion.")
e("gl-gentlemen", r"\bgentlemen\b|\bgentleman\b", "gendered_language", 2, 0.85,
  why="Addresses candidates as male.", alt=["professionals"], gain="Neutral address.")

# ── stereotype (incl. leadership stereotypes) ────────────────────────────────
e("st-emotional", r"\bemotional\b", "stereotype", 2, 0.6, amb=True,
  veto=[r"emotional\s+(intelligence|support|wellbeing|well-being|resilience|labor)"],
  why="'Not emotional' / 'emotional' as a screen invokes the stereotype of women as too emotional; 'emotional intelligence' is a legitimate skill.",
  alt=["composed under pressure"], gain="States the real requirement.")
e("st-bossy", r"\bbossy\b", "stereotype", 3, 0.9, why="'Bossy' is a gendered pejorative applied to women leaders.",
  alt=["directive"], gain="Removes gendered pejorative.")
e("st-abrasive", r"\babrasive\b", "stereotype", 2, 0.8, why="'Abrasive' is disproportionately applied to women in reviews (per analyses of performance-review language).",
  alt=["direct"], gain="Neutral description.")
e("st-manly", r"\bmanly\b|\bmacho\b", "stereotype", 3, 0.95, why="Explicit male stereotype.", alt=["professional"], gain="Removes explicit stereotype.")
e("st-womanly", r"\bwomanly\b|\bfeminine\s+(touch|charm|grace)\b", "stereotype", 3, 0.95,
  why="Explicit female stereotype.", alt=["professional"], gain="Removes explicit stereotype.")
e("st-motherly", r"\bmotherly\b|\bmaternal\s+(instinct|nature)\b", "stereotype", 3, 0.9,
  why="Casts the role as maternal — gender + caregiver stereotype.", alt=["supportive"], gain="Skill framing.")
e("st-fatherly", r"\bfatherly\b|\bpaternal\b", "stereotype", 3, 0.9, why="Male-parent stereotype.", alt=["mentoring"], gain="Skill framing.")
e("st-dominant-leader", r"\b(dominant|commanding|forceful)\s+leader(?:ship)?\b", "stereotype", 3, 0.9,
  why="Dominance-framed leadership evokes a male leadership prototype (role-congruity research).",
  alt=["decisive, collaborative leadership"], gain="Leadership defined by outcomes, not dominance.")
e("st-command-control", r"\bcommand[\s\-]and[\s\-]control\b", "stereotype", 2, 0.85,
  why="'Command-and-control' leadership invokes a dominance-based, male-typed leadership prototype.",
  alt=["clear, accountable leadership"], gain="Leadership defined by outcomes, not dominance.")
e("st-commanding-presence", r"\bcommanding\s+presence\b|\bauthoritative\s+voice\b", "stereotype", 2, 0.8,
  why="'Commanding presence' and 'authoritative voice' are evaluated against a male prototype and are not job skills.",
  alt=["communicates with credibility"], gain="Assessable behavior, no prototype.")
e("st-command-respect", r"\bcommands?\s+respect\b", "stereotype", 2, 0.75,
  why="'Commands respect' invokes dominance-based leadership, a male-typed prototype.",
  alt=["earns trust across teams"], gain="Trust-based leadership framing.")
e("st-born-leader", r"\bborn\s+leader\b|\bnatural[\s\-]born\s+leader\b", "stereotype", 1, 0.65,
  why="'Born leader' implies leadership is an innate persona (stereotypically male-typed) rather than a skill.",
  alt=["proven leadership skills"], gain="Assessable requirement.")
e("st-executive-presence", r"\bexecutive\s+presence\b", "stereotype", 1, 0.55, amb=True,
  why="'Executive presence' is vague, unassessable, and in practice evaluated against a male prototype.",
  alt=["communicates credibly with senior stakeholders"], gain="Concrete, assessable behavior.")
e("st-gravitas", r"\bgravitas\b", "stereotype", 1, 0.55, amb=True,
  why="'Gravitas' is a vague prototype-based demand applied unevenly by gender.",
  alt=["credibility with senior audiences"], gain="Concrete behavior.")
e("st-iron-fist", r"\biron\s+fist\b|\btake[\s\-]no[\s\-]prisoners\b", "stereotype", 3, 0.9,
  why="Dominance/combat leadership stereotype.", alt=["holds the team to high standards"], gain="Standards without dominance.")
e("st-manup", r"\bman\s+up\b|\bgrow\s+a\s+pair\b", "stereotype", 3, 0.98, why="Explicitly genders toughness.",
  alt=["be resilient"], gain="Removes explicit gendering.")
e("st-hysterical", r"\bhysterical\b|\bhysteria\b", "stereotype", 3, 0.9,
  why="Historically gendered pejorative.", alt=["calm under pressure (state the positive requirement)"], gain="Neutral requirement.")
e("st-strongman", r"\bstrong\s+man\b", "stereotype", 3, 0.9, why="Genders strength.", alt=["strong candidate"], gain="Neutral.")
e("st-assertive-man", r"\bassertive\s+(man|male|guy)\b", "stereotype", 3, 0.98, why="Genders assertiveness explicitly.",
  alt=["confident communicator"], gain="Neutral skill.")

# ── caregiver_bias ────────────────────────────────────────────────────────────
e("cb-nofamily", r"\bno\s+(family|personal|caregiving|childcare)\s+(commitments?|responsibilit(?:y|ies)|obligations?)\b",
  "caregiver_bias", 3, 0.98, why="Directly screens out caregivers — disproportionately women, and likely unlawful in many places.",
  alt=["remove; state the actual schedule requirement instead"], gain="Legal, inclusive, and clearer about the real demand.")
e("cb-fully-dedicated", r"\b(must\s+be\s+)?(fully|totally|completely)\s+(dedicated|committed)\b", "caregiver_bias", 2, 0.8,
  why="'Total dedication' is a proxy for unlimited availability, which screens for absence of caregiving duties.",
  alt=["committed to delivering results during working hours"], gain="Commitment framed by output, not availability.")
e("cb-lifestyle-fit", r"\blifestyle\s+fit\b", "caregiver_bias", 2, 0.8,
  why="'Lifestyle fit' invites screening on family status rather than skills.",
  alt=["alignment with our working practices (state them)"], gain="Transparent, skills-relevant criterion.")
e("cb-247", r"\b(available\s+)?24\s*/\s*7\b|\bavailable\s+(at\s+)?all\s+times\b|\balways\s+(on|available)\b",
  "caregiver_bias", 2, 0.8, amb=True,
  veto=[r"(support|service|system|platform|hotline|coverage|operations?)\s+(is\s+|runs?\s+)?(available\s+)?24\s*/\s*7",
        r"24\s*/\s*7\s+(\w+\s+){0,2}(support|service|system|operations?|center|centre|coverage|monitoring|network)",
        r"(rotating|rotation|shift)\s+schedule"],
  why="Demanding personal 24/7 availability screens out caregivers; describing a 24/7 service with defined shifts is fine.",
  alt=["participates in an on-call rotation (approx. X nights/month)"], gain="Real requirement, honestly scoped.")
e("cb-nogaps", r"\bno\s+(employment\s+|work\s+|career\s+)?gaps\b|\bgaps?\s+in\s+(employment|work|career)\s+history\b|\bcontinuous\s+(employment|work)\s+history\b",
  "caregiver_bias", 3, 0.95,
  why="Career-gap screens penalize parenting and caregiving breaks (mostly taken by women) with no evidence of job relevance.",
  alt=["remove; assess skills and recent accomplishments instead"], gain="Wider pool, skills-based screen.")
e("cb-career-break", r"\bno\s+career\s+breaks?\b|\bmust\s+not\s+have\s+taken\s+(a\s+)?break\b", "caregiver_bias", 3, 0.95,
  why="Penalizes caregiving breaks directly.", alt=["remove"], gain="Skills-based screen.")
e("cb-work-first", r"\b(put\s+)?work\s+(comes\s+)?first\b|\bjob\s+comes\s+first\b|\bwork\s+is\s+your\s+life\b",
  "caregiver_bias", 3, 0.9, why="Demands work take priority over family — screens for absence of caregiving duties.",
  alt=["state actual hours and peak periods honestly"], gain="Honest scope, no lifestyle screen.")
e("cb-relocate-now", r"\bwilling\s+to\s+relocate\s+immediately\b|\bimmediate\s+relocation\s+required\b|\brelocate\s+(internationally\s+)?within\s+\d+\s+days\b|\buproot\s+their\s+li(?:fe|ves)\b", "caregiver_bias", 2, 0.75,
  why="Immediate-relocation demands disproportionately exclude people with family ties; if relocation is real, give a timeline.",
  alt=["relocation to X required within N months (assistance provided)"], gain="Keeps the requirement with humane scope.")
e("cb-evenings-always", r"\bevenings?\s+and\s+weekends?\s+(required\s+)?(with\s+)?no\s+exceptions\b", "caregiver_bias", 2, 0.75,
  why="Blanket 'no exceptions' availability screens caregivers; specify the actual schedule.",
  alt=["scheduled evening/weekend shifts (rota shared in advance)"], gain="Real schedule, predictable for caregivers.")
e("cb-unencumbered", r"\bunencumbered\b|\bfree\s+(of|from)\s+(family|personal|caregiving)\s+(obligations?|duties|commitments?)\b", "caregiver_bias", 3, 0.95,
  why="Directly demands freedom from family obligations.", alt=["remove"], gain="Removes unlawful-leaning screen.")
e("cb-single-pref", r"\bsingle\s+(candidates?\s+)?(preferred|only)\b|\bunmarried\b|\bno\s+(kids|children)\b", "caregiver_bias", 3, 0.99,
  why="Marital/family-status screening — exclusionary and unlawful in many jurisdictions.", alt=["remove"], gain="Legal and inclusive.")
e("cb-no-parents", r"\bno\s+(mothers?|fathers?|parents?)\b", "caregiver_bias", 3, 0.99,
  why="Directly excludes parents — unlawful family-status discrimination in many jurisdictions.",
  alt=["remove"], gain="Legal and inclusive.")
e("cb-unbroken", r"\bunbroken\b[^.\n]{0,40}\b(employment|work|career)\b|\b(employment|work|career)\b[^.\n]{0,20}\bunbroken\b",
  "caregiver_bias", 3, 0.9,
  why="'Unbroken employment record' penalizes caregiving breaks with no evidence of job relevance.",
  alt=["assess skills and recent accomplishments instead"], gain="Skills-based screen, wider pool.")
e("cb-weekends-unpaid", r"\bweekends?\s+without\s+(additional\s+)?(compensation|pay|overtime)\b",
  "caregiver_bias", 2, 0.85,
  why="Unpaid weekend expectations disproportionately screen out caregivers.",
  alt=["paid weekend shifts on a published rota"], gain="Honest, fair scheduling requirement.")
e("cb-travel-100", r"\b100%\s+travel\b|\btravel\s+100%\b", "caregiver_bias", 1, 0.6, amb=True,
  why="Extreme travel demands exclude caregivers; if genuine, keep but state support offered.",
  alt=["extensive travel (~75–100%); itineraries shared N weeks ahead"], gain="Requirement preserved, predictability added.")
e("cb-flexible-forus", r"\bflexib(?:le|ility)\s+(to\s+work\s+)?(long|extra|additional)\s+hours\b", "caregiver_bias", 1, 0.6,
  why="One-way 'flexibility' (for the employer) signals unpredictable hours that deter caregivers.",
  alt=["occasional overtime during launches (paid/TOIL)"], gain="Honest, bounded expectation.")

# ── age_coded ────────────────────────────────────────────────────────────────
e("ac-over-need-not", r"\b(over|under|above|below)\s+\d{2}\b[^.\n]{0,40}\bneed\s+not\s+apply\b|\bneed\s+not\s+apply\s+if\s+(over|under)\s+\d{2}\b",
  "age_coded", 3, 0.98, why="Explicit age screen — unlawful in many jurisdictions.",
  alt=["remove"], gain="Legal and inclusive.")
e("ac-skews-young", r"\bskews?\s+young\b", "age_coded", 2, 0.9,
  why="Advertising a young age profile signals older candidates aren't wanted.",
  alt=["describe the culture without age"], gain="Culture signal without age screen.")
e("ac-out-of-university", r"\b(just|straight|fresh|right)\s+out\s+of\s+(university|college|school)\b",
  "age_coded", 2, 0.85, why="Youth proxy — 'just out of university' functions as an age cap.",
  alt=["early-career candidates welcome"], gain="Experience band without age proxy.")
e("ac-twentysomething", r"\btwenty[\s\-]somethings?\b|\bthirty[\s\-]somethings?\b", "age_coded", 2, 0.9,
  why="Describes the team or expectation by age bracket.", alt=["energetic team"],
  gain="Culture described without age.")
e("ac-digital-native", r"\bdigital\s+native\b", "age_coded", 3, 0.95,
  why="'Digital native' is a proxy for 'young' — an age screen, not a skill.",
  alt=["proficient with modern digital tools (name them)"], gain="Names the real skill, drops the age proxy.")
e("ac-young-energetic", r"\byoung\s+(and\s+)?(energetic|dynamic|vibrant|hungry)\b", "age_coded", 3, 0.95,
  why="Explicitly demands youth.", alt=["energetic"], gain="Keeps energy, drops age.")
e("ac-fresh-grad", r"\bfresh\s+graduates?\s+(only|preferred)\b|\brecent\s+graduates?\s+(only|preferred)\b", "age_coded", 3, 0.9,
  why="Restricting to fresh graduates functions as an age cap.",
  alt=["0–2 years' experience welcome"], gain="Experience band without age proxy.")
e("ac-under-age", r"\bunder\s+\d{2}\s*(years\s+old|yrs?)?\s*(preferred|only)?\b|\b\d{2}\s+years?\s+or\s+younger\b|\bmaximum\s+age\b",
  "age_coded", 3, 0.98, why="Explicit age cap — unlawful in many jurisdictions.", alt=["remove"], gain="Legal and inclusive.")
e("ac-young-team", r"\b(join\s+our\s+)?young(?:,?\s+\w+)?\s+team\b|\byouthful\s+(team|culture|environment)\b", "age_coded", 2, 0.85,
  why="Advertising a 'young team' signals older candidates aren't wanted.",
  alt=["energetic, close-knit team"], gain="Culture described without age.")
e("ac-millennial", r"\bmillennial\s+(mindset|energy|vibe)\b|\bgen[\s\-]?z\s+(energy|vibe)\b", "age_coded", 2, 0.85,
  why="Generation labels are age proxies.", alt=["current with digital trends"], gain="Skill, not birth year.")
e("ac-born-after", r"\bborn\s+(after|before)\s+(19|20)\d{2}\b", "age_coded", 3, 0.98, why="Explicit age screen.",
  alt=["remove"], gain="Legal and inclusive.")
e("ac-highenergy-young", r"\bhigh[\s\-]energy\s+young\b", "age_coded", 3, 0.95, why="Explicit youth demand.",
  alt=["high-energy"], gain="Energy without age.")
e("ac-newgrad-only", r"\bnew\s+grads?\s+only\b", "age_coded", 3, 0.9, why="Age-cap proxy.",
  alt=["early-career candidates welcome"], gain="Inclusive experience band.")
e("ac-overqualified", r"\boverqualified\s+(candidates?\s+)?(need\s+not|will\s+not|should\s+not)\b", "age_coded", 2, 0.85,
  why="'No overqualified candidates' is commonly an age screen in disguise.", alt=["remove"], gain="Skills-based screen.")
e("ac-recent-college", r"\brecent\s+college\s+(grad|graduate)s?\b", "age_coded", 1, 0.6, amb=True,
  why="Fine for a defined new-grad program; an age proxy elsewhere.", alt=["early-career professionals"],
  gain="Keeps program intent without age proxy.")
e("ac-energetic-alone", r"\benergetic\b", "age_coded", 1, 0.4, amb=True,
  why="'Energetic' alone is usually fine; it becomes an age proxy when stacked with youth signals.",
  alt=["motivated"], gain="Usually fine as-is; Layer 2 decides from context.")

# ── appearance_bias ───────────────────────────────────────────────────────────
e("ap-attractive", r"\battractive\b|\bgood[\s\-]looking\b|\bphotogenic\b", "appearance_bias", 3, 0.95,
  veto=[r"attractive\s+(salary|compensation|package|benefits|offer|opportunity|proposition)"],
  why="Looks requirements are rarely job-relevant and are policed more heavily against women.",
  alt=["professional presentation"], gain="Skills-relevant standard.")
e("ap-wellgroomed", r"\bwell[\s\-]groomed\b|\bclean[\s\-]cut\b", "appearance_bias", 1, 0.6, amb=True,
  why="Grooming standards can be legitimate (client-facing/safety) but are enforced unevenly by gender; state specifics.",
  alt=["adheres to our dress and safety standards (linked)"], gain="Specific, evenly-applicable standard.")
e("ap-presentable", r"\bpresentable\b", "appearance_bias", 1, 0.6, amb=True,
  why="'Presentable' is vague and gender-unevenly enforced.", alt=["professional dress standard"], gain="Specific standard.")
e("ap-physfit", r"\bphysically\s+fit\b", "appearance_bias", 1, 0.55, amb=True,
  veto=[r"(lift|carry|move)\s+\d+\s*(lbs?|kg|pounds)", r"physical\s+demands?\s+(include|of\s+the\s+role)"],
  why="'Physically fit' is vague; if the job has physical demands, state them measurably.",
  alt=["able to lift 25 kg repeatedly during a shift"], gain="Measurable requirement replaces vague screen.")
e("ap-slim", r"\bslim\b|\bpetite\b|\bheight\s*[:\-]?\s*\d|\bheight\s+requirements?\b|\b(minimum|maximum)\s+height\b|\bweight\s+requirements?\b", "appearance_bias", 3, 0.95,
  why="Body-type requirements are appearance screens with disparate gender impact.", alt=["remove or state validated physical job demands"],
  gain="Removes body-type screening.")
e("ap-smile", r"\bgreat\s+smile\b|\bwinning\s+smile\b", "appearance_bias", 3, 0.9,
  why="Appearance demand, typically aimed at women in service roles.", alt=["friendly customer manner"], gain="Service skill, not looks.")
e("ap-youthful-look", r"\byouthful\s+(appearance|look)\b|\byoung[\s\-]looking\b", "appearance_bias", 3, 0.95,
  why="Combines appearance and age screening.", alt=["remove"], gain="Removes double screen.")
e("ap-neat-appearance", r"\bneat\s+(and\s+tidy\s+)?appearance\b", "appearance_bias", 1, 0.55, amb=True,
  why="Vague appearance demand; specify the actual standard if one exists.", alt=["follows workplace dress code"], gain="Specific standard.")

# ── exclusionary ──────────────────────────────────────────────────────────────
e("ex-native-speaker", r"\bnative\s+(english\s+)?speaker\b|\benglish\s+(as\s+a\s+)?(mother\s+tongue|first\s+language)\b|\bnative[\s\-]level\s+english\b",
  "exclusionary", 3, 0.95,
  why="'Native speaker' excludes fully proficient speakers by origin; the job needs proficiency, not birthplace. (EEOC guidance distinguishes fluency requirements from national-origin proxies.)",
  alt=["professional English proficiency (C1/C2)"], gain="Keeps the real language bar, drops the origin screen.")
e("ex-whph", r"\bwork\s+hard[\s,]*play\s+hard\b", "exclusionary", 2, 0.9,
  why="'Work hard, play hard' signals an after-hours social culture (often drinking-centered) that excludes caregivers and non-drinkers, and correlates with male-typed cultures.",
  alt=["we work hard and celebrate wins as a team"], gain="Culture described without the after-hours screen.")
e("ex-beer", r"\bbeer\s+(on\s+tap|fridge|o[’']?clock)\b|\bkegerator\b|\bhappy\s+hours?\b", "exclusionary", 1, 0.6, amb=True,
  why="Alcohol-centered perks as headline culture signals exclude non-drinkers and caregivers; fine as one perk among many.",
  alt=["team events and celebrations"], gain="Inclusive social signal.")
e("ex-brotherhood", r"\bbrotherhood\b|\bfraternit(?:y|ies)\b|\bfrat\s+(house|culture|vibe)\b", "exclusionary", 3, 0.95,
  why="Frames the workplace as a male social unit.", alt=["a tight-knit team"], gain="Belonging without gendering.")
e("ex-culture-fit", r"\bculture\s+fit\b|\bcultural\s+fit\b", "exclusionary", 1, 0.5, amb=True,
  why="Unstructured 'culture fit' screens replicate the existing (often homogeneous) team; 'culture add' or explicit values are better.",
  alt=["alignment with our values (list them)"], gain="Assessable values instead of similarity bias.")
e("ex-boys-fit", r"\bfit\s+in\s+with\s+the\s+(boys|guys|lads)\b", "exclusionary", 3, 0.98,
  why="Explicitly requires fitting a male group.", alt=["collaborate well with the team"], gain="Removes explicit exclusion.")
e("ex-bro", r"\bbro\s+culture\b|\btech\s+bros?\b|\bbrogrammer\b", "exclusionary", 3, 0.9,
  why="Male-typed culture signal.", alt=["collaborative engineering culture"], gain="Inclusive culture signal.")
e("ex-family-we-are", r"\bwe[’']?re\s+(like\s+)?a\s+family\b|\bwork\s+family\b", "exclusionary", 1, 0.5,
  why="'We're a family' blurs boundaries and pressures unpaid overtime — deters people with actual family obligations.",
  alt=["a supportive, close-knit team"], gain="Warmth without boundary-blurring.")
e("ex-ablebodied", r"\bable[\s\-]bodied\b", "exclusionary", 3, 0.95,
  why="Excludes disabled candidates categorically; state actual physical requirements.",
  alt=["able to perform the physical tasks listed (with or without accommodation)"], gain="Legal, specific, inclusive.")
e("ex-no-9to5", r"\bnot\s+a\s+9[\s\-]?(to|–|-)[\s\-]?5\b|\bthis\s+isn[’']?t\s+a\s+9[\s\-]?(to|–|-)[\s\-]?5\b", "exclusionary", 2, 0.8,
  why="Signals unbounded hours — screens out caregivers without stating a real requirement.",
  alt=["core hours X–Y with occasional evening launches"], gain="Honest hours, wider pool.")
e("ex-work-weekends-social", r"\bmandatory\s+(team\s+)?(drinks|socials?|happy\s+hours?)\b", "exclusionary", 2, 0.85,
  why="Mandatory social/drinking events exclude caregivers and non-drinkers.", alt=["optional team socials"], gain="Social culture made optional.")
e("ex-grit-hazing", r"\bsink\s+or\s+swim\b|\btrial\s+by\s+fire\b|\bbaptism\s+(of|by)\s+fire\b", "exclusionary", 2, 0.8,
  why="Hazing-style onboarding language deters candidates from under-supported groups.",
  alt=["structured onboarding with real ownership from week one"], gain="Challenge without hazing signal.")

# ── qualification_inflation ───────────────────────────────────────────────────
e("qi-all-reqs", r"\bmust\s+meet\s+(all|100%|every(\s+single)?\s+one)\s+(of\s+)?(the\s+)?(requirements?|qualifications?|criteria)\b",
  "qualification_inflation", 3, 0.9,
  why="Demanding 100% of requirements shrinks the female applicant pool most, since women tend to apply only near full qualification.",
  alt=["if you meet most of these, we encourage you to apply"], gain="Directly counteracts the documented application-threshold gap.")
e("qi-only-qualified", r"\bonly\s+(fully\s+)?qualified\s+candidates?\s+(need|should)\s+apply\b|\bdo\s+not\s+apply\s+unless\b",
  "qualification_inflation", 2, 0.85,
  why="Gatekeeping phrasing deters near-qualified candidates unevenly by gender.",
  alt=["we welcome applicants who meet most requirements"], gain="Keeps the bar, widens the funnel.")
e("qi-expert-everything", r"\bexpert\s+in\s+(all|every|everything)\b", "qualification_inflation", 2, 0.8,
  why="Nobody is an expert in everything; inflated demands deter women disproportionately.",
  alt=["strong in several of: …"], gain="Realistic requirements.")
e("qi-worldclass", r"\bworld[\s\-]class\b|\bbest\s+of\s+the\s+best\b|\btop\s+1%\b", "qualification_inflation", 1, 0.6,
  why="Superlative bars are unassessable and deter candidates with realistic self-assessments.",
  alt=["excellent"], gain="Assessable standard.")
e("qi-ivy", r"\bivy\s+league\b|\btop[\s\-]tier\s+(university|school|college)\b|\bprestigious\s+(university|institution|school)\b|\belite\s+(university|school|college)\b",
  "qualification_inflation", 2, 0.85,
  why="Pedigree screens exclude qualified candidates from less privileged paths and narrow diversity on every axis.",
  alt=["degree or equivalent practical experience"], gain="Skills-based credential requirement.")
e("qi-unicorn", r"\bunicorn\s+candidate\b|\bpurple\s+squirrel\b", "qualification_inflation", 1, 0.7,
  why="'Unicorn' framing signals an inflated, unrealistic wish list.", alt=["strong generalist"], gain="Realistic role definition.")
e("qi-jack", r"\bjack\s+of\s+all\s+trades\b", "qualification_inflation", 1, 0.7,
  why="Inflated generalist demand — and 'jack' is also gendered.", alt=["versatile generalist"], gain="Neutral and realistic.")
e("qi-rockstar-team", r"\bA[\s\-]players?\s+only\b", "qualification_inflation", 2, 0.8,
  why="'A-players only' gatekeeping deters near-qualified applicants unevenly.", alt=["high standards, strong support"], gain="Standards without gatekeeping.")
e("qi-years-inflated", r"\b(1[5-9]|[2-9]\d)\+?\s+years[’']?\s+(of\s+)?experience\s+required\b|\b(fifteen|twenty|twenty[\s\-]five|thirty)[\s\-]?(plus|\+)?\s+years\s+(of\s+)?experience\s+required\b", "qualification_inflation", 1, 0.55, amb=True,
  why="Very long tenure demands are rarely evidence-based and function as both age and gender screens.",
  alt=["significant experience (10+ years) or equivalent depth"], gain="Keeps seniority, drops arbitrary bar.")
e("qi-advanced-degree", r"\b(advanced\s+degree|master[’']?s(\s+degree)?|mba)\s+required\b", "qualification_inflation", 1, 0.5, amb=True,
  why="Degree requirements not tied to job tasks shrink diverse pipelines; consider 'or equivalent experience'.",
  alt=["Master's degree or equivalent experience"], gain="Skills-based alternative path.")
e("qi-no-training", r"\bmust\s+hit\s+the\s+ground\s+running\b|\bno\s+hand[\s\-]?holding\b", "qualification_inflation", 1, 0.6,
  why="'No hand-holding' signals absent onboarding/support, deterring candidates from under-supported groups.",
  alt=["quick ramp-up with structured onboarding"], gain="Support signal widens the pool.")
e("qi-perfection", r"\bperfectionist\b|\bflawless\s+execution\b", "qualification_inflation", 1, 0.6,
  why="Perfection demands raise the self-selection threshold unevenly by gender.",
  alt=["high attention to detail"], gain="Realistic standard.")

INCLUSIVE_SIGNALS = [
    dict(id="is-eeo", pattern=r"\bequal\s+opportunit(?:y|ies)\s+employer\b|\bEEO\b", label="Equal-opportunity statement", points=8),
    dict(id="is-encourage", pattern=r"\bencouraged?\s+to\s+apply\b|\bapply\s+even\s+if\b|\bdon[’']?t\s+meet\s+every\b", label="Encourages near-qualified applicants", points=10),
    dict(id="is-flex", pattern=r"\bflexible\s+(work(?:ing)?|hours|schedule)\b|\bflex\s+time\b|\bflexitime\b", label="Flexible working", points=8),
    dict(id="is-remote", pattern=r"\bremote(?:\s|-)?(friendly|first|work)?\b|\bhybrid\b|\bwork\s+from\s+home\b", label="Remote/hybrid option", points=6),
    dict(id="is-parental", pattern=r"\bparental\s+leave\b|\bmaternity\b|\bpaternity\b|\bfamily\s+leave\b", label="Parental leave mentioned", points=8),
    dict(id="is-they", pattern=r"\bthey\s+will\b|\bthe\s+successful\s+candidate\s+will\b", label="Gender-neutral pronouns", points=5),
    dict(id="is-allgenders", pattern=r"\ball\s+genders\b|\bregardless\s+of\s+gender\b|\bany\s+gender\b", label="Explicit gender inclusion", points=8),
    dict(id="is-accommodation", pattern=r"\breasonable\s+accommodations?\b|\baccessibilit(?:y|ies)\b|\baccommodations?\s+available\b", label="Accessibility/accommodations", points=6),
    dict(id="is-underrep", pattern=r"\bunderrepresented\b|\bdiverse\s+backgrounds?\b|\bdiversity\s+and\s+inclusion\b|\bD&I\b|\bDEI\b", label="Diversity commitment", points=6),
    dict(id="is-salary", pattern=r"\bsalary\s+(range|band)\b|\b\$\s?\d[\d,]*\s*(-|–|to)\s*\$\s?\d[\d,]*\b|\bpay\s+range\b", label="Pay transparency", points=8),
    dict(id="is-parttime", pattern=r"\bpart[\s\-]time\s+(option|available|considered)\b|\bjob[\s\-]shar(?:e|ing)\b", label="Part-time/job-share option", points=6),
    dict(id="is-returners", pattern=r"\breturnship\b|\bcareer\s+returners?\b|\breturning\s+to\s+work\b", label="Career-returner friendly", points=8),
    dict(id="is-neutral-titles", pattern=r"\bchairperson\b|\bspokesperson\b|\bsalesperson\b|\bfirefighter\b|\bpolice\s+officer\b|\bmail\s+carrier\b|\bserver\b", label="Gender-neutral job titles", points=4),
    dict(id="is-skills-based", pattern=r"\bor\s+equivalent\s+(practical\s+)?experience\b", label="Skills-based alternative to credentials", points=6),
    dict(id="is-oncall-scoped", pattern=r"\bon[\s\-]call\s+rotation\b", label="Honestly scoped on-call", points=4),
]

# Validate all regexes compile
for p in P:
    re.compile(p["pattern"], re.IGNORECASE)
    for v in p["veto"]:
        re.compile(v, re.IGNORECASE)
for s in INCLUSIVE_SIGNALS:
    re.compile(s["pattern"], re.IGNORECASE)

ids = [p["id"] for p in P]
assert len(ids) == len(set(ids)), "duplicate ids"
print(f"{len(P)} patterns, {len(INCLUSIVE_SIGNALS)} inclusive signals")
assert len(P) >= 150, f"need >=150 patterns, have {len(P)}"

out = dict(version="2.0",
           note="Editable source of truth for Layer 1. severity: 1=low 2=medium 3=high. 'ambiguous' entries are adjudicated by the Layer 2 contextual classifier. 'veto' regexes suppress a hit when the sentence clearly uses the term benignly.",
           categories=CATEGORIES, patterns=P, inclusive_signals=INCLUSIVE_SIGNALS)
path = sys.argv[1] if len(sys.argv) > 1 else "patterns.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
print("wrote", path)

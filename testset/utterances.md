# Test set to record (20 utterances, then repeat once with noise for the noisy condition)

Record 16 kHz mono WAV if you can (any rate works, it is resampled). Leave ~0.3 s of silence at
the start; stop talking, do not trail off. Name files `clean_01.wav` ... `clean_20.wav`.
The noisy condition: play fan/traffic audio in the room, or mix noise in digitally
(`noisy_01.wav` ...). Use only synthetic roll numbers S1001-S1010.

| # | Utterance | Lang | intent | roll_no |
|---|---|---|---|---|
| 01 | meri scholarship approve hui ya nahi? | hinglish | scholarship | (none: agent must ask) |
| 02 | fees kitni baaki hai? | hinglish | fee | (none) |
| 03 | mera roll number S1004 hai, scholarship ka status batao | hinglish | scholarship | S1004 |
| 04 | S1002 ki fees kitni due hai? | hinglish | fee | S1002 |
| 05 | mera roll number S1001 hai, meri fees kab tak bharni hai? | hinglish | fee | S1001 |
| 06 | S1007 scholarship reject kyun hui, ab kya karna hoga? | hinglish | scholarship | S1007 |
| 07 | roll number S1010, kya meri scholarship ka paisa aa gaya? | hinglish | scholarship | S1010 |
| 08 | mujhe apni fee ka status chahiye, S1006 | hinglish | fee | S1006 |
| 09 | scholarship ka next step kya hai? | hinglish | scholarship | (none) |
| 10 | S1003 ka scholarship status kya hai | hinglish | scholarship | S1003 |
| 11 | मेरी स्कॉलरशिप का स्टेटस क्या है, रोल नंबर S1005 | hindi | scholarship | S1005 |
| 12 | मेरी फीस कितनी बाकी है | hindi | fee | (none) |
| 13 | मेरा रोल नंबर S1008 है, फीस की आखिरी तारीख बताइए | hindi | fee | S1008 |
| 14 | what is my scholarship status, my roll number is S1009 | english | scholarship | S1009 |
| 15 | how much fee do I still have to pay | english | fee | (none) |
| 16 | roll number S1004, is my fee paid | english | fee | S1004 |
| 17 | S1001 scholarship approve ho gayi na? paisa kab milega | hinglish | scholarship | S1001 |
| 18 | bhaiya S1006 ki fees due hai kya | hinglish | fee | S1006 |
| 19 | mera application kahan tak pahuncha, roll number S1003 | hinglish | scholarship | S1003 |
| 20 | S1002 ka scholarship status aur fees dono batao | hinglish | scholarship | S1002 |

Utterance 20 asks about both intents; the correctness check only requires the scholarship tool
(extra calls are allowed). Copy `manifest.example.csv` to `manifest.csv` once recorded.

# ch05 movie-search qrels: review summary

Judged 2026-10-01. Binary relevance over the pooled (query, movie) pairs only. 427 pairs total; 31 judged relevant. One relevance file written to `qrels.json`.

Rule applied: a movie is relevant only if it genuinely satisfies the query information need. Topical adjacency or a shared keyword alone is not enough.

## Per-query results

### q01 GoldenEye (exact title): 1 / 29 relevant
- GoldenEye (1995)

### q02 Twelve Monkeys (exact title): 1 / 23 relevant
- Twelve Monkeys (1995)

### q03 Heat (exact-title-ambiguous): 1 / 26 relevant
- Heat (1995)
- The 1995 Michael Mann crime film is the only target. Films that merely carry the word "heat" in their text were not judged relevant.

### q04 Samuel L. Jackson (exact cast name): 9 / 30 relevant
- Die Hard: With a Vengeance (1995), as Zeus Carver
- Fluke (1995), voice of Rumbo
- Fresh (1994), as Sam the chess player
- Jurassic Park (1993), as Ray Arnold
- Kiss of Death (1995), as Calvin Hart
- Lightning Jack (1994), as Ben Doyle
- Losing Isaiah (1995), as Kadar Lewis
- Menace II Society (1993), as Tat Lawson
- Pulp Fiction (1994), as Jules Winnfield
- This matches the probe note's expectation of roughly nine Jackson films in the slice.

### q05 movies starring Robin Williams (cast name, natural language): 5 / 27 relevant
- Being Human (1994), as Hector
- Jumanji (1995), as Alan Parrish
- Mrs. Doubtfire (1993), as Daniel Hillard
- Nine Months (1995), as Dr. Kosevich
- The Birdcage (1996), as Armand Goldman

### q06 Robert De Niro as a Las Vegas casino boss (mixed cast + plot): 1 / 25 relevant
- Casino (1995)
- Strict reading. De Niro distractors (Heat, Taxi Driver, A Bronx Tale) and other Las Vegas films (Leaving Las Vegas, Showgirls, Destiny Turns on the Radio) do not match the casino-boss role and were marked not relevant.

### q07 animated film about toys that come to life (genre + plot): 1 / 34 relevant
- Toy Story (1995)

### q08 a dark thriller about a detective hunting a serial killer (genre + plot): 3 / 27 relevant
- Copycat (1995)
- Se7en (1995)
- Virtuosity (1995), borderline (see below)

### q09 two investigators track a methodical murderer punishing people for the cardinal vices (paraphrase of Se7en): 1 / 34 relevant
- Se7en (1995)
- Copycat is in this pool too, but its killer copies famous serial killers rather than punishing the cardinal vices, so it does not match this paraphrase.

### q10 a prisoner is sent to the past to trace a plague... (paraphrase of Twelve Monkeys): 1 / 37 relevant
- Twelve Monkeys (1995)
- Outbreak (virus but no time travel) and Timecop (time travel but no plague) do not match both elements.

### q11 a self-destructive man... bonds with a sex worker during his final days (paraphrase of Leaving Las Vegas): 1 / 38 relevant
- Leaving Las Vegas (1995)

### q12 a medieval highland warrior rallies his people to overthrow their foreign rulers (paraphrase of Braveheart): 1 / 27 relevant
- Braveheart (1995)
- Rob Roy and Highlander: The Final Dimension share the highland/Highlander keyword but do not match the "rallies his people to overthrow foreign rulers" plot; marked not relevant (Rob Roy flagged borderline below).

### q13 secret agent spy thriller with gadgets and a villain plotting global sabotage (OPEN): 4 / 29 relevant
- GoldenEye (1995)
- True Lies (1994)
- Clear and Present Danger (1994), borderline (CIA espionage; see below)
- Bad Company (1995), borderline (CIA / private intelligence espionage; see below)
- Judged generously as "genuine spy or espionage thriller," but not every action film. GoldenEye and True Lies are the clearest Bond-style fits. In the Line of Fire, Fair Game, and Executive Decision were considered and marked not relevant (see borderline notes).

### q14 a lighthearted comedy about a failed athlete who becomes an unlikely golf star (paraphrase of Happy Gilmore): 1 / 41 relevant
- Happy Gilmore (1996)

## Borderline calls flagged for author review

- q05, Speechless (1994): marked 0. Robin Williams has only an uncredited cameo; the query asks for movies "starring" him, so a cameo does not qualify.
- q07, The Indian in the Cupboard (1995): marked 0. Toy figures do come to life, but the film is live action, not animated, so it fails the "animated film" constraint.
- q08, Virtuosity (1995): marked 1. A former cop literally hunts a serial killer (SID 6.7, a synthetic composite of serial killers), which fits "a detective hunting a serial killer," though the film is science-fiction action rather than a conventional dark thriller.
- q08, Beyond Bedlam (1994) and Hideaway (1995): marked 0. Both involve a killer and an investigator or victim with a psychic link, but neither is cleanly a detective hunting a serial killer; they read as horror.
- q12, Rob Roy (1995): marked 0. A Scottish Highland warrior, but set in the 1700s (not medieval) and driven by a personal honor and revenge feud rather than rallying his people to overthrow foreign rulers.
- q13, Clear and Present Danger (1994): marked 1. A genuine CIA espionage thriller, but the antagonist is a Colombian drug cartel rather than a villain plotting global sabotage, and it lacks the gadget flavor.
- q13, Bad Company (1995): marked 1. A genuine espionage thriller about CIA operatives and a private intelligence firm running covert operations; less Bond-style, so flagged.
- q13, In the Line of Fire (1993): marked 0. A Secret Service agent versus a lone presidential assassin; a federal-agent thriller, but not an espionage or spy story with gadgets or global sabotage.
- q13, Fair Game (1995) and Executive Decision (1996): marked 0. Rogue KGB antagonists (Fair Game) and a counter-terrorism hijacking plot (Executive Decision) give a spy flavor, but the protagonists are a cop and an intelligence analyst in action settings rather than secret-agent spy thrillers.

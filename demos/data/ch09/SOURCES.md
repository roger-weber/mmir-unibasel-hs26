# Chapter 9 Demo Data: Sources

All files come from Wikimedia Commons. Images are the 960 px renditions served by Commons
(the ginger cat is the 736 px original); audio files are 30-second excerpts converted to mono,
16 kHz, 16-bit WAV with ffmpeg.

## Images (`images/`)

| File | Commons file | Author | License |
|---|---|---|---|
| `taj_a.jpg` | [20191204 Taj Mahal Agra 0622 6533 DxO.jpg](https://commons.wikimedia.org/wiki/File:20191204_Taj_Mahal_Agra_0622_6533_DxO.jpg) | Jakub Hałun | CC BY-SA 4.0 |
| `taj_b.jpg` | [20191204 Taj Mahal Agra 0623 6541 DxO.jpg](https://commons.wikimedia.org/wiki/File:20191204_Taj_Mahal_Agra_0623_6541_DxO.jpg) | Jakub Hałun | CC BY-SA 4.0 |
| `mt_forbes.jpg` | [Alpenglow on Mt. Forbes in the Canadian Rockies.jpg](https://commons.wikimedia.org/wiki/File:Alpenglow_on_Mt._Forbes_in_the_Canadian_Rockies.jpg) | BrettA343 | CC BY-SA 4.0 |
| `red_car.jpg` | [Ferarri Ferrari red matte red (6778354615).jpg](https://commons.wikimedia.org/wiki/File:Ferarri_Ferrari_red_matte_red_(6778354615).jpg) | Ben | CC BY-SA 2.0 |
| `sunset.jpg` | [Red-orange sunset over Jura.jpg](https://commons.wikimedia.org/wiki/File:Red-orange_sunset_over_Jura.jpg) | Jan Remund | CC BY 2.0 |
| `black_cat.jpg` | [Black cat lying on a blue bed quilt.jpg](https://commons.wikimedia.org/wiki/File:Black_cat_lying_on_a_blue_bed_quilt.jpg) | Candeadly | CC BY 4.0 |
| `ginger_cat.jpg` | [Small Cat between the grass.jpg](https://commons.wikimedia.org/wiki/File:Small_Cat_between_the_grass.jpg) | Waledyousef | CC BY-SA 4.0 |

## Audio (`audio/`)

| File | Commons file | Performer / reader | License | Excerpt |
|---|---|---|---|---|
| `speech_munsell.wav` | [Albert Henry Munsell - A Color Notation (LibriVox) 02.mp3](https://commons.wikimedia.org/wiki/File:Albert_Henry_Munsell_-_A_Color_Notation_(LibriVox)_02.mp3) | LibriVox volunteer | CC0 | 60 to 90 s |
| `speech_tucker.wav` | [LibriVox - The Sorcery of Aphlar (Lovecraft, Rimel).mp3](https://commons.wikimedia.org/wiki/File:LibriVox_-_The_Sorcery_of_Aphlar_(Lovecraft,_Rimel).mp3) | Ben Tucker (LibriVox) | Public domain | 60 to 90 s |
| `speech_hedrick.wav` | [Librivox - House on the Cliff (1927) Chapter 1.mp3](https://commons.wikimedia.org/wiki/File:Librivox_-_House_on_the_Cliff_(1927)_Chapter_1.mp3) | James R. Hedrick (LibriVox) | Public domain | 60 to 90 s |
| `music_morning.wav` | [Musopen - Morning.ogg](https://commons.wikimedia.org/wiki/File:Musopen_-_Morning.ogg) | Musopen Symphony Orchestra (Grieg) | Public domain | 30 to 60 s |
| `music_mountain_king.wav` | [Musopen - In the Hall Of The Mountain King.ogg](https://commons.wikimedia.org/wiki/File:Musopen_-_In_the_Hall_Of_The_Mountain_King.ogg) | Musopen Symphony Orchestra (Grieg) | Public domain | 30 to 60 s |
| `music_goldberg_aria.wav` | [Bach, Goldberg Variations, Aria (Musopen version).ogg](https://commons.wikimedia.org/wiki/File:Bach,_Goldberg_Variations,_Aria_(Musopen_version).ogg) | Musopen (Bach) | CC0 | 30 to 60 s |

## Web pages

`news_page.html` and `hotel_ad_page.html` were written for demo ch09-01.

## Recorded model answers

`llm_metadata.json` holds the answers of Claude Sonnet 5.5 on AWS Bedrock (recorded 2026-11-02)
to the prompts of demo ch09-03: descriptions and JSON metadata for every photo, plus three runs
each of the forced and hedged naming prompts. The demo replays them when `LIVE` is False.

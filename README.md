# Intro
This project is in development process so scripts are not ready and working properly. It aimed to finish the project's first phase by the end of the july/2024, purpose is creating a Turkish Sign Language Recognizer.
It aimed to be based on Mediapipe's "Holositcs" LandMark Detection model.

* The model is will be based on words. Not to actual sentences.
* python version, as far I'm using 3.12.0

# Why I Do Develop This Project ?
* It is predicted that Türkiye has 836 k people that are deaf and mute.  Those people are using Turkish Sign Language to communicate. Türkiye's population is 836k out of  80 million is deaf and mute and  is just %1 percent
So, many people does not know Turkish Sign Language(TSL) and there's only little percent that obligated to learn this language.
For the people that does not have deaf, mute relatives it is hard to understand TSL. In daily life you may need any language in any time.
Basically you can open translate and use these languages, What I aim for the future, that you can open your mobile app and understand sign language.
 

---
# Aimed Audience:
* People that does not know Turkish Sign Language and people that having hard time while learning it.
---
# What is the data ?
* Data is coming from Mediapipe's holistics model that provides landmarks for Face, Pose(Posture) and for both hands (left hand and right hand).
---
# Future Developments
* Model can be converted in to tensorflow lite version and integrated in to mobile application.
* Always, can be added new words

# You can check/learn Turkish Sign Language from here:
* https://www.youtube.com/@isaretdiliegitimi5504 (some YouTube channel that provides many words and letters)
* https://orgm.meb.gov.tr/dosyalar/00012/tid_sozluk.pdf ( Türkiye's Governments Official Turkish Sign Language paper)
* Also, I'm collecting data according to those two sources. To be clear, I'm imitating the signs myself as described, shown and collecting data via mediapipe's holistics.
---
# Libraries:
python- 3.12 (as far I'm stick with it, maybe it can be lowered to 3.10)
* numpy 
* keras
* open-cv
* scikit-learn
* yt-dlp (https://github.com/yt-dlp/yt-dlp)
* mediapipe (https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md)
---


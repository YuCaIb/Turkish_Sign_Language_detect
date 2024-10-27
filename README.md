# Intro
This project is in development process so scripts are not ready and working properly. It aimed to finish the project's first phase by the end of the july/2024, purpose is creating a Turkish Sign Language Recognizer.
It aimed to be based on Mediapipe's "Holositcs" LandMark Detection model.

* The model is will be based on words. Not to actual sentences.
* ~~python version, as far I'm using 3.12.0~~.
* I downgraded to python 3.11.0 it is better for ML-deep learning algorithms.

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
~~python- 3.12 (as far I'm stick with it, maybe it can be lowered to 3.10)~~
* I downgraded to  Python version 3.11 to compilable with torch and keras.
* flask
* numpy 
* keras
* open-cv
* scikit-learn
* torch
* yt-dlp (https://github.com/yt-dlp/yt-dlp)
* mediapipe (https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md)
---
# Data
* you can download 55 words, that collected by me from [here](https://drive.google.com/drive/folders/1OlvLZqXLRrz1OcG1FgIaLKxNSDGFRCaB?usp=sharing)
---
# Model
* best model that is %91 categorical_accuracy while training but with my evaluations it scored as %54 accuracy score you can download from [here](https://drive.google.com/file/d/1vZubaYU6Hsywn7a7QG4az84W1qMfPjPU/view?usp=sharing)
---
# Guide:
* `git clone https://github.com/YuCaIb/Turkish_Sign_Language_detect.git`
* Download [model](https://drive.google.com/file/d/1vZubaYU6Hsywn7a7QG4az84W1qMfPjPU/view?usp=sharing), [folder_names](https://drive.google.com/file/d/10iK4knCaSd0GX5_at4U-7tNLFD2TRFWK/view?usp=sharing), [X](https://drive.google.com/file/d/19WKcmDOSoD3twRdYHoYgiXUMzEl3X7bR/view?usp=sharing), [y](https://drive.google.com/file/d/1ys_LffT-W9YNZeoI-bfJjz8sUNuUfcl0/view?usp=sharing). Take them somewhere you can find.
* PS : X is data set and y is labels. folder_names are labels. For now Model is not good, I will be uploading this in the future.
* handle paths of; X , y ,folder_names, in the needed place  
* ` conda create -n signlang python = 3.11`
* `conda activate signlang`
* `pip install mediapipe keras tensorflow yt-dlp scikit-learn opencv-python flask numpy`
* for torch https://pytorch.org/get-started/locally/
* cd path content
* python web_UI.py
* So you can try the model.
---

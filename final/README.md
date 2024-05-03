# Music Generation with Generative Adversarial Networks 

It is strongly suggested that using "model.ipynb" on google colab to run the project. Codes are divided into blocks according to the function and comments are written in each part to help understand. "model_local.py" is a simplified version which we use to debug and test when the "model.ipynb" is running on the colab.

If the user choose "model.ipynb", here are instructions.

## Steps:

1. Download the GiantMIDI-Piano from https://github.com/bytedance/GiantMIDI-Piano and upload it to the google drive.
2. Import the "model.ipynb" to the colab.
3. Click "Run All". The authority to connect with your google drive will be requested. Select "Yes".
4. Wait until the model finishes training. All results will be printed out.

It is also acceptable if the user wants to train the model locally using "model_local.py". 

## Steps:

1. Download the GiantMIDI-Piano from https://github.com/bytedance/GiantMIDI-Piano.
2. Install dependencies and change related file paths.
3. Input "python3 train.py" in the terminal or click "RUN" in IDE(such as VSCode).

Two demos, "piano.wav" and "mixed.wav", are offered. The first one is an audio only includes piano while the second one is a mixed version, which has the same melody but consists of different kinds of instruments. "mixer.ipynb" is the file to produce mixed audio.

## Index
- [Project Overview](#project-overview)
- [Implementation Plan](#implementation-plan)
- [Project Structure](#project-structure)
- [Resources](#resources)
- [Taining Instructions](#how-to-train-the-models)
- [Environment Setup](#setting-up-environment)
- [Final Result](#final-result)
- [Appendix](#appendix-a)

## Project Overview
In recent years, environmental management problems has received increasing attention from the public. One of the key problems people are trying to solve is the ability to efficiently identify
different types of waste materials during the waste recycling process. Automatic waste detection becomes a necessity when tons of waste materials have to be processed at the waste recycle center every day. Our project aims to utilize deep learning and computer vision techniques to efficiently and accurately detect different waste materials.

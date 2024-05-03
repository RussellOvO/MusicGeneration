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

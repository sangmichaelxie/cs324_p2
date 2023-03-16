# Project 2 (Building Large Language Models) for Stanford CS324: Understanding and Developing Large Language Models (Winter 2022)

This repository contains the code for project 2 on training language models in the Winter 2022 iteration of [CS324: Understanding and Developing Large Language Models](https://stanford-cs324.github.io/winter2022/) at Stanford. This code was developed by Sang Michael Xie and Percy Liang.
- In the first part of the project, we use fine-tuning/continued-pretraining to instill length controllability into GPT-2 small, where we can control the word length of the model's generated response by prepending metadata about length. The data, preprocessing code, and training code are provided, but some hyperparameters need to be tuned.
- In the second part of the project, we aim to instill a new capability of choice into a language model. The project code is built on top of HuggingFace, and the data for the first part is based on OpenWebText. By default, we provide scripts for running the project on CodaLab, a platform for reproducibile research, but the scripts and code can be adapted to run locally or on other platforms as well.

Useful links:
- [Project document and starter guide](https://stanford-cs324.github.io/winter2022/projects/CS324_P2.pdf)
- [Data assets](https://worksheets.codalab.org/worksheets/0x21906fed417147e4a24ef486086b9178)

Setup and quickstart for CodaLab:
- [CodaLab guide](https://docs.google.com/document/d/1rWgWyYJc36Vow2eU8oZ0pfVU7SzFRg3Eb1dB1XHZzvE/edit#heading=h.3oa3f2ydysz2)

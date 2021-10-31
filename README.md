
<h1 align="center">
  <br>
  <a href="https://pf.kakao.com/_FlDxgs"><img src="https://i.loli.net/2021/10/31/jIPeZOcNkg6nB8o.png" alt="PEEP-Talk Logo"></a>
</h1>

<h4 align="center">PEEP-Talk: Deep Learning-based English Education Platform for Personalized Foreign Language Learning</h4>
<p align="center">Human-inspired AI, Korea University</p>

<p align="center">
    <img alt="python-3.7.7" src="https://img.shields.io/badge/python-3.7.7-blue"/>
    <img alt="django-2.2.5" src="https://img.shields.io/badge/KakaoTalk-offline-yellow"/>
    <img alt="chromedriver-79.0.3945" src="https://img.shields.io/badge/chromedriver-79.0.3945-blueviolet"/>
    <img alt="GitHub" src="https://img.shields.io/github/license/metterian/redbttn-seoul-studio"/>
</p>


> PEEP-Talk is an educational platform with a deep learning-based persona conversation system and a feedback function for correcting English grammar. In addition, unlike the existing persona conversation system, a Context Detector (CD) module that can automatically determine the flow of conversation and change the conversation topic in real time can be applied to give the user a feeling of talking to a real person.

> The source code is open so that you can download the source code and set it up with ease if you would like to have your own exclusive environment, and this platform is deployed by Kakao i Open Builder.

## Screenshots

screenshot1             |  screenshot2
:-------------------------:|:-------------------------:
![](https://i.loli.net/2021/10/31/nYtvxABGIHQsDL2.png)  |  ![](https://i.loli.net/2021/10/31/4BHTGFmatUACcP2.png)

- screenshot1: This is an example of a screen of PEEP-Talk that can be used through the KakaoTalk channel. The answer of the conversation system to the user input, the result of the CD module, and the information on correcting grammar errors are provided in the form of feedback as shown in the figure above.
- screenshot2: This is an example of a case where the CD module detects a conversation of a user out of context and changes the conversation topic, that is, a persona.



## Project interests

### Conversational Agent
By considering persona as a situation, English conversation learning for each situation becomes possible. To make conversational agent model mainly, we use Hugging Face's [TransferTransfo](https://github.com/huggingface/transfer-learning-conv-ai) code.

### Context Detector
This module can detect whether user speak properly in suggested situation or not. This module contains two BERT based models. Evaluate the conversation using the following two functions. Based on that score, we decide whether to change the conversation.
- **Context Similarity**(상황 유사도): fine-tuinig the MRPC(Microsoft Research Paraphrase Corpus) dataset to detect user's context similarity in suggested situation.
- **Linguistic Acceptability**(문장 허용도): fine-tuning the CoLA(The Corpus of Linguistic Acceptability) dataset to detect user's input is acceptable in human conversation.



## Folder Structure
    .
    ├── .idea
    ├── homepage                # Main page app
    ├── redbttn_home            # Django project settings
    ├── static                  # Static folder
    ├── templates               # HTML template folder
    ├── insta_login.py          # Selenium module code
    ├── parser.py               # Instagram crawling code
    ├── manage.py
    ├── requirements.txt
    ├── LICENSE
    └── README.md




## Run
to interact with PEEP-talk :
```
python run.py
```
to kakao server:
```
python kakao.py
```


## Video
- [PC Version](https://youtu.be/w9NuSj_xY1o)
- [Mobile Version](https://youtu.be/pgPuoi7n1Uc)





## License
The MIT License


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



## Site

### Home Page




## Project interests

### Front-end

#### HLS Streaming
- Need real-time video streaming technology to match the self-video studio concept
- AWS CloudFront for simultaneous streaming to multiple connectors
- Video JS
#### Instagram Grid Style
- Real-time synchronized Instagram feed layout
#### Optimized for mobile(android & ios) and web
- CSS for devices which has multiple resolutions
- Video format with Android and iOS support

### Back-end
#### Crawling Bot
- Develop a crawling bot for real-time Instagram feed synchronization
- Modularize each function like a bot e.g. Log in, Crawling, Terminate
#### Server
- AWS EC2, Ubuntu 18.04
- Crontab(crawling bot with real-time scheduling)





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



## Dependencies
<img alt="python-3.7.7" src="https://img.shields.io/badge/python-3.7.7-blue"/>
<img alt="django-2.2.5" src="https://img.shields.io/badge/Django-2.2.5-brightgreen"/>
<img alt="chromedriver-79.0.3945" src="https://img.shields.io/badge/chromedriver-79.0.3945-blueviolet"/>



to run parser.py:
- download chromedriver, unzip, move to  `HOME_PATH/chromedriver` (mac os / linux)


create a secret.json file with variables:

```
SECRET_KEY = 'secret_key_in_settings.py'
username = 'instagram_username'
password = 'instagram_password'
```

## Run
to make instagram database using crawling:
```
python3 parser.py
```
to run server:
```
python3 manage.py runserver
```


## Video
- [PC Version](https://youtu.be/w9NuSj_xY1o)
- [Mobile Version](https://youtu.be/pgPuoi7n1Uc)


## Skill
<p align="left">
    <img alt="python" src="https://img.shields.io/badge/Python- -black"/>
    <img alt="python-3.7.7" src="https://img.shields.io/badge/CSS-%20-blue"/>
    <img alt="" src="https://img.shields.io/badge/HTML-%20-orange"/>
    <img alt="javascript" src="https://img.shields.io/badge/JavaScript-%20-yellow"/>
    <img alt="JQuery" src="https://img.shields.io/badge/JQuery- -blue"/>
    <img alt="hls" src="https://img.shields.io/badge/HLS-%20-red"/>
    <img alt="videojs" src="https://img.shields.io/badge/VideoJS-%20-yellowgreen"/>
    <img alt="sqlite3" src="https://img.shields.io/badge/sqlite3- -blue"/>
    <img alt="selenuum" src="https://img.shields.io/badge/selenuum- -black"/>
    <img alt="beautifulsoup4" src="https://img.shields.io/badge/beautifulsoup4- -green"/>
    <img alt="AWS" src="https://img.shields.io/badge/AWS-%20-orange"/>
</p>


## Wiki
📕[WIKI](https://www.notion.so/Back-end-fc842cd3273a4e10b82a9e7d550826ae)

The wiki include concerns about technical issues and solutions to the development of the homepage.

## Reference
- [VideoJS](https://videojs.github.io/videojs-contrib-hls/)

- [Web Scraping Instagram with Selenium](https://medium.com/analytics-vidhya/web-scraping-instagram-with-selenium-b6b1f27b885) | by Mariyasha | Analytics Vidhya | Medium
- [Automate TINDER with Python tutorial](https://github.com/aj-4/tinder-swipe-bot) | AJ-4 | Youtube

- [Django  Project Deploy - 1. AWS](https://nachwon.github.io/django-deploy-1-aws/) | nachwon

- [Amazon AWS :: Configuring HLS Streaming with Cloudfront](https://salesmore.tistory.com/851) | SalesMore

- [Building Video Streaming Services (AWS s3/cloudFront, HLS, video.js)](http://lab.naminsik.com/3960) | Dev. Nam Lab.

## License
The MIT License

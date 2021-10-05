#%%
import pandas as pd
import json
import sys
import os
import urllib.request
from tqdm.auto import tqdm
import time
import googletrans
#%%
data_path = os.path.join(os.path.pardir, 'data/translation_eng_kor.xlsx')
trans = pd.read_excel(data_path, engine='openpyxl')
# %%
trans.head()
# %%
len(trans['상황'].unique())
# %%

def translate(kor_text: str, wait : bool = False) -> dict:
    client_id = "Bcuz8MOTWPbsDanaYreD" # 개발자센터에서 발급받은 Client ID 값
    client_secret = "JjamGQA6_4" # 개발자센터에서 발급받은 Client Secret 값
    encText = urllib.parse.quote(kor_text)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)

    if wait:
        time.sleep(0.5)
    response = urllib.request.urlopen(request, data=data.encode("utf-8"))

    with urllib.request.urlopen(request, data=data.encode("utf-8")) as response:
        rescode = response.getcode()
        if(rescode==200):
            translation = json.loads(response.read().decode('utf8'))
        else:
            raise Exception
    return translation['message']['result']['translatedText']

# %%


translator = googletrans.Translator()

tqdm.pandas()
# %%
# %%
trans['situation'] = trans['상황'].apply(lambda x: translator.translate(x).text )

# %%
from collections import OrderedDict

situation_trans = OrderedDict()

translate = lambda x : translator.translate(x).text


from tqdm import tqdm
for situation in tqdm(trans['상황'].unique()):
    situation_trans[situation] = translate(situation)
    time.sleep(0.5)

# %%
import os
from google.cloud import translate_v2 as translate
import json

client = translate.Client()
result = client.translate('안녕하세요', target_language='en')

secret_path = "/root/client_secret.json"

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = secret_path
# %%
# %%

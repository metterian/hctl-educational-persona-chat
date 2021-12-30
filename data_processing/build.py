#%%
from typing import List
import pandas as pd
import json
import os
import random
import spacy
from tqdm import tqdm
from setproctitle import setproctitle
from utils import get_absolute_path, space_before_eos

#%%
# environment setting
def set_config(proc_title: str, gpu_id: int) -> None:
    tqdm.pandas()
    setproctitle(proc_title)  # set the pid
    spacy.prefer_gpu(gpu_id)  # set gpu id


def load_excel(file_path: str) -> pd.DataFrame:
    """
    It fetches Excel files and outputs them as Pandas.
    """
    dataset_path = get_absolute_path(file_path)
    dataset = pd.read_excel(dataset_path, engine="openpyxl")
    return dataset


def preprocess_dataset():
    """
    After loading the dataset, pre-process the lowercase and space the eos.
    And output the dataset as an Excel file.
    """
    # load dataset
    trans = load_excel("data/translation_eng_kor.xlsx")

    trans["번역문"] = trans["번역문"].progress_apply(space_before_eos)
    trans.to_excel(get_absolute_path("data/translation_eng_kor_eos.xlsx"))


def get_topk_situation(k: int) -> List:
    """
    This function outputs the top k situations based on the number of conversation turns.
    For data matching, data work is performed on the top k situations with a large number of conversation turns.
    """
    situ_count = trans.groupby("상황").count().sort_values(by="소분류", ascending=False)
    situ_count["대화수"] = situ_count["번역문"] / 4
    situations = situ_count.head(k).index.tolist()  # 대화수 상위 15개의 상황
    return situations


def sample_candidate(candidates: List, num: int = 18) -> List:
    """
    Create a candidate sentence used to predict the next utterance answer.
    Samples 18 from candidates in different situations.

    return:
        List
    example:
        "my mom was single with 3 boys , so we never left the projects .",
        "i try to wear all black every day . it makes me feel comfortable .",
        "well nursing stresses you out so i wish luck with sister",
        "..."

    """
    return random.sample(candidates, num)


def load_dataset() -> tuple:
    """
    The data set in which the preprocessing operation is completed and the data set labeled with the situation are retrieved, and preprocessing is performed.
    The output result is in dictionary form.

    returns :
        Dict
    example:
        situation: ['description1','description2','description3','description4']
    """
    # dataset reload
    dialogue = load_excel("data/translation_eng_kor_eos.xlsx")
    # load situation labels
    situation_label_path = get_absolute_path("data_processing/situation_label.json")
    with open(situation_label_path) as fp:
        situation_labels = json.load(fp)

    # space the punctuation in <eos>
    for situation, description in situation_labels.items():
        description = list(map(space_before_eos, description))
        situation_labels[situation] = description

    return dialogue, situation_labels


def match_situation() -> List[dict]:
    """
    Match conversation with situation label.

    """
    dataset = []
    dialogue, situation_labels = load_dataset()
    for situation_label, persona in situation_labels.items():
        situation_label = situation_label.replace("(", r"\(")

        is_contain = lambda text: dialogue[dialogue["상황"].str.contains(text)]
        is_not_contain = lambda text: dialogue[~dialogue["대분류"].str.contains(text)]

        situation = is_contain(situation_label)
        top_situation = is_contain(situation_label)["대분류"].iloc[0]
        candidates = is_not_contain(top_situation)["번역문"].to_list()
        conversations = situation.groupby("Set Nr.")["번역문"].apply(list).tolist()

        for conversation in conversations:
            utterances = []
            for i in range(len(conversation) - 1):
                candidate = sample_candidate(candidates) + [conversation[i + 1]]
                utterances.append(
                    {"candidates": candidate, "history": conversation[: i + 1]}
                )
            dialogue_entry = {"personality": persona, "utterances": utterances}
            dataset.append(dialogue_entry)
    return dataset


def main():
    nlp = spacy.load(
        "en_core_web_sm",  # set the tokenizer
    )
    set_config("joon_persona", 4)
    preprocess_dataset()
    # match_situation()


if __name__ == "__main__":
    main()


# %%
with open("test_dataset.json", "w+") as fp:
    json.dump(dataset, fp, indent=4)
# %%

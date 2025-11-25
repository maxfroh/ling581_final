import pandas as pd
import csv
import json
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter

import chardet


def snip():
    with open("blogtext.csv", mode="r", encoding="utf-8") as f:
        with open("snippet.csv", mode="w", encoding="utf-8") as out:
            for _ in range(500):
                out.write(f.readline())


def process_full_data(chunk_size=400000):
    chunks = pd.read_csv("blogtext.csv", chunksize=chunk_size)

    ages = Counter()
    genders = Counter()
    topics = Counter()
    signs = Counter()

    for chunk in tqdm(chunks, desc="Reading in data..."):
        # print(df.info)
        ages = ages + Counter(chunk["age"].value_counts().to_dict())
        genders = genders + Counter(chunk["gender"].value_counts().to_dict())
        topics = topics + Counter(chunk["topic"].value_counts().to_dict())
        signs = signs + Counter(chunk["sign"].value_counts().to_dict())

    obj = {
        "ages": ages,
        "genders": genders,
        "topics": topics,
        "signs": signs
    }

    with open("counts.json", mode="w", encoding="utf-8") as f:
        json.dump(obj, f)


def make_graphs():
    with open("counts.json", mode="r", encoding="utf-8") as f:
        data = json.load(f)

    if not os.path.isdir("graphs"):
        os.mkdir("graphs")

    ages = data["ages"]
    genders = data["genders"]
    topics = data["topics"]
    signs = data["signs"]
    colors = ["red", "orange", "gold", "lightyellow", "greenyellow", "limegreen",
              "mediumspringgreen", "lightskyblue", "royalblue", "mediumpurple", "violet", "pink"]

    plt.pie(list(genders.values()), labels=list(
        genders.keys()), autopct="%.2f%%", startangle=90)
    plt.title("Gender Distribution in Dataset")
    plt.savefig("graphs/gender.png", bbox_inches="tight", dpi=300)

    plt.clf()
    plt.pie(list(signs.values()), labels=list(signs.keys()),
            autopct="%.2f%%", colors=colors, startangle=90)
    plt.title("Zodiac Sign Distribution in Dataset")
    plt.savefig("graphs/zodiac.png", bbox_inches="tight", dpi=300)

    plt.clf()
    plt.pie(list(topics.values()), labels=list(topics.keys()),
            autopct="%.2f%%", colors=colors, startangle=90)
    plt.title("Topic Distribution in Dataset")
    plt.savefig("graphs/topics.png", bbox_inches="tight", dpi=300)

    plt.clf()
    ages = sorted(ages.items(), key=lambda a: int(a[0]))
    plt.bar([a[0] for a in ages], [a[1] for a in ages])
    plt.title("Age Distribution in Dataset")
    plt.savefig("graphs/age.png", bbox_inches="tight", dpi=300)


def age_to_label(age_ranges, age):
    for k, (low, high) in age_ranges.items():
        if low <= age <= high:
            return k
    return -1


def clean_data():
    age_ranges = {0: (13, 17), 1: (23, 27), 2: (33, 42)}
    allowed_ages = set().union(*[set(range(low, high + 1))
                                 for low, high in age_ranges.values()])
    with open("cleaned_data.csv", mode="w", encoding="utf-8") as fout:
        with open("blogtext.csv", mode="r", encoding="utf-8") as fin:
            old_header = fin.readline()
            new_header = "id,gender,age,label,topic,sign,date,text\n"
            fout.write(new_header)
            line = fin.readline()
            while len(line) != 0:
                vals = line.split(",")
                #  original order:
                #  id,gender,age,topic,sign,date,text
                #  0  1      2   3     4    5-7  8+
                if int(vals[2]) in allowed_ages:
                    clean_text = ",".join(vals[8:]).strip().strip("\"").strip().replace("\x00", "")
                    if len(clean_text) > 0:
                        clean_text = "\"" + clean_text + "\""
                        cleaned_vals = [vals[0], vals[1], vals[2], str(age_to_label(age_ranges, int(
                            vals[2]))), vals[3], vals[4], ",".join(vals[5:8]), clean_text]
                        cleaned_line = ",".join(cleaned_vals) + "\n"
                        fout.write(cleaned_line)
                line = fin.readline()


# process_full_data()
# make_graphs()
clean_data()

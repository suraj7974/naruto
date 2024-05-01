from nltk import sent_tokenize
import nltk
import en_core_web_sm
import spacy
import pandas as pd
from glob import glob
nltk.download('punkt')

path_to_subtitles = sorted(glob("./subtitles/*.ass"))
path_to_subtitles[:10]
scripts = []
episode_number = []
for path in path_to_subtitles:
    with open(path, 'r') as file:
        lines = file.readlines()
        lines = lines[27:]

        rows = [",".join(line.split(",")[9:]) for line in lines]

    rows = [line.replace("\\N", " ") for line in rows]
    script = " ".join(rows)

    episode = int(path.split('_')[0].split('/')[2].strip())
    scripts.append(script)
    episode_number.append(episode)


df = pd.DataFrame.from_dict({'episode': episode_number, 'script': scripts})

df.head()
nlp = en_core_web_sm.load()
doc = nlp("Sasuke went to konoha")
for ent in doc.ents:
    print(ent.text, ent.label_)


def get_ners(script):
    script_sentences = sent_tokenize(script)

    ner_output = []

    for sentence in script_sentences:
        doc = nlp(sentence)
        ners = set()
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                full_name = ent.text
                first_name = full_name.split(' ')[0]
                ners.add(first_name)
        ner_output.append(list(ners))
    return ner_output


df['ners'] = df['script'].apply(get_ners)
window = 10
entity_relationship = []

for row in df['ners']:
    previous_entities_in_window = []

    for sentence in row:
        previous_entities_in_window.append(sentence)
        previous_entities_in_window = previous_entities_in_window[-10:]

        previous_entities_flattend = sum(previous_entities_in_window, [])

        for entity in sentence:
            for entity_in_window in previous_entities_flattend:
                if entity != entity_in_window:
                    entity_rel = sorted([entity, entity_in_window])
                    entity_relationship.append(entity_rel)

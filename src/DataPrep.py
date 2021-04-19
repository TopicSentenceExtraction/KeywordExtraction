import json
import pandas as pd
from rake_nltk import Rake


def get_metadata():
    with open('data/arxiv-metadata-oai-snapshot.json') as f:
        for line in f:
            yield line


def print_variable_names(metadata):
    for paper in metadata:
        first_paper = json.loads(paper)
        break

    print("We have the following variables in each record(one paper):")
    for key in first_paper:
        print(key)


def extract_keywords(r, text):
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases()
    return ranked_phrases


def export_metadata(metadata):
    titles = []
    abstracts = []
    keywords = []
    total_items = 0
    r = Rake()
    for paper in metadata:
        paper = json.loads(paper)
        total_items += 1
        titles.append(paper['title'])
        # get abstract and its corresponding keywords
        abstract = paper['abstract']
        abstracts.append(abstract)
        keywords.append(extract_keywords(r, abstract))

    print(f'Total number of items is: {total_items}')

    d = {
        'title': titles,
        'abstract': abstracts,
        'keyword': keywords
    }

    arxiv_metadata_dataset = pd.DataFrame(d)
    arxiv_metadata_sample_dataset = arxiv_metadata_dataset.sample(frac=0.01, random_state=1)
    arxiv_metadata_dataset.to_csv(
        'data/arxiv_metadata_dataset.csv')  # dataset contains all the titles and abstracts of papers
    arxiv_metadata_sample_dataset.to_csv(
        'data/arxiv_metadata_sample_dataset.csv')  # sample dataset, contains 1% of the original dataset


def main():
    metadata = get_metadata()
    print_variable_names(metadata)
    export_metadata(metadata)


if __name__ == "__main__":
    main()

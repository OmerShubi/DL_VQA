import json
import os
from collections import Counter
import itertools

from preprocessing import data_preprocessing
import utils


def extract_vocab(iterable, top_k=None, start=0):
    """ Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """
    # TODO change to pytorch Vocab
    all_tokens = itertools.chain.from_iterable(iterable)
    counter = Counter(all_tokens)
    if top_k:
        most_common = counter.most_common(top_k)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    # descending in count, then lexicographical order
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start)}
    return vocab


def create_vocab(data_base_path, data_paths, vocab_path, max_answers=3000):
    questions = os.path.join(data_base_path, data_paths['questions'])
    answers = os.path.join(data_base_path, data_paths['answers'])

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data_preprocessing.prepare_questions(questions)
    answers = data_preprocessing.prepare_answers(answers)


    question_vocab = extract_vocab(questions, start=1)

    answer_vocab = extract_vocab(answers, top_k=max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    vocabulary_path = vocab_path  # path where the used vocabularies for question and answers are saved to

    with open(vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    create_vocab()

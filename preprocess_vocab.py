import json
from collections import Counter
import itertools

import data_preprocessing
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


def main():
    questions = utils.path_for(train=True, question=True)
    answers = utils.path_for(train=True, answer=True)

    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    questions = data_preprocessing.prepare_questions(questions)
    answers = data_preprocessing.prepare_answers(answers)


    question_vocab = extract_vocab(questions, start=1)

    max_answers = 3000  # TODO param
    answer_vocab = extract_vocab(answers, top_k=max_answers)

    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }
    vocabulary_path = './vocab.json'  # path where the used vocabularies for question and answers are saved to  # TODO param

    with open(vocabulary_path, 'w') as fd:
        json.dump(vocabs, fd)


if __name__ == '__main__':
    main()
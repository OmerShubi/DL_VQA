import json
import os
from collections import Counter
import itertools

from preprocessing import data_preprocessing


def extract_vocab(word_lst, num_most_frequent=None, start=0):
    """
        Turns an iterable of list of tokens into a vocabulary.
        These tokens could be single answers or word tokens in questions.
    """

    word_dict = itertools.chain.from_iterable(word_lst)
    word_counter = Counter(word_dict)
    print(len(word_counter))

    if num_most_frequent:
        most_common_words = word_counter.most_common(num_most_frequent)
        most_common_words = (word for word, _ in most_common_words)
    else:
        most_common_words = word_counter.keys()

    # descending in count
    tokens = sorted(most_common_words, key=lambda x: (word_counter[x], x), reverse=True)

    # Reverse dictionary
    vocab = {t: i for i, t in enumerate(tokens, start=start)}

    return vocab


def create_vocab(data_base_path, data_paths, vocab_path, max_answers=3000):

    # Get Paths of answers and questions
    questions = os.path.join(data_base_path, data_paths['questions'])
    answers = os.path.join(data_base_path, data_paths['answers'])

    # Load jsons
    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)

    # Preprocess Questions
    questions = data_preprocessing.prepare_questions(questions)
    question_vocab = extract_vocab(questions, start=1)

    # Preprocess Answers
    answers = data_preprocessing.prepare_answers(answers)
    answer_vocab = extract_vocab(answers, num_most_frequent=max_answers, start=1)

    # Save vocab
    vocabs = {
        'question': question_vocab,
        'answer': answer_vocab,
    }

    with open(vocab_path, 'w') as fd:
        json.dump(vocabs, fd)

import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
UNKOWN_TOKEN = 0
from collections import namedtuple

class VQA_dataset(torch.utils.data.Dataset):
    """ VQA_dataset dataset, open-ended """

    def __init__(self, data_paths, other_paths, logger, answerable_only=False):
        super(VQA_dataset, self).__init__()
        # self.semi_dense = namedtuple('semi_dense', ['indices', 'values', 'size'])

        base_path = other_paths['base_path']
        questions_path = os.path.join(base_path, data_paths['questions'])
        answers_path = os.path.join(base_path, data_paths['answers'])
        vocabulary_path = other_paths['vocab_path']
        self.image_path = data_paths['processed_imgs']
        logger.write("Opening files")
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        with open(vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)
        logger.write("Checking integrity")

        self._check_integrity(questions_json, answers_json)
        # vocab
        self.vocab = vocab_json
        self.question_token_to_index = self.vocab['question']
        self.answer_to_index = self.vocab['answer']

        logger.write("preparing and encoding questions")

        # questions - Turn a question into a padded vector of vocab indices and a question length
        self.questions_list = list(prepare_questions(questions_json))
        self.questions = [self._encode_question(q) for q in self.questions_list]

        logger.write("preparing and encoding answers")
        # answers - Turn an answer into a vector of counts of all possible answers
        # self.answers = [self._encode_answers(a) for a in prepare_answers(answers_json)]

        self.answer_indices = []
        self.answer_values = []
        self.answer_lengths = []
        for a in prepare_answers(answers_json):
            index, values, answer_length = self._encode_answers(a)
            self.answer_indices.append(index)
            self.answer_values.append(values)
            self.answer_lengths.append(answer_length)

        self.answer_indices = torch.nn.utils.rnn.pad_sequence(self.answer_indices, batch_first=True)
        self.answer_values = torch.nn.utils.rnn.pad_sequence(self.answer_values, batch_first=True)

        logger.write("indexing images")
        # imgs
        self.coco_ids = [q['image_id'] for q in questions_json['questions']]
        self.coco_id_to_h5index = self._create_coco_id_to_index()

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            logger.write("answerable_only")
            self.answerable = self._find_answerable()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions_list))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.question_token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_path, 'r') as features_file:
            coco_ids = features_file['ids'][()]
        coco_id_to_index = {id: i for i, id in enumerate(coco_ids)}
        return coco_id_to_index

    def _check_integrity(self, questions, answers):
        """ Verify that we are using the correct data """
        qa_pairs = list(zip(questions['questions'], answers['annotations']))
        assert all(q['question_id'] == a['question_id'] for q, a in qa_pairs), 'Questions not aligned with answers'
        assert all(q['image_id'] == a['image_id'] for q, a in qa_pairs), 'Image id of question and answer don\'t match'
        assert questions['data_type'] == answers['data_type'], 'Mismatched data types'
        assert questions['data_subtype'] == answers['data_subtype'], 'Mismatched data subtypes'

    def _find_answerable(self):
        """ Create a list of indices into questions that will have at least one answer that is in the vocab """
        answerable = []
        for i, answer_length in enumerate(self.answer_lengths):
            # store the indices of anything that is answerable
            if answer_length > 0:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.question_token_to_index.get(token, UNKOWN_TOKEN)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer

        # get indices of answers that have an id in answer vocab
        answers_with_id_from_vocab = [self.answer_to_index.get(answer) for answer in answers if self.answer_to_index.get(answer) is not None]

        # get unique indices and how many counts of each
        unique_indices, counts = np.unique(answers_with_id_from_vocab, return_counts=True)
        # return torch.sparse_coo_tensor(indices=torch.tensor([unique_indices]), values=torch.tensor(counts), size=(len(self.answer_to_index),))
        return torch.tensor(unique_indices), torch.tensor(counts), len(unique_indices)

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'processed_images'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.processed_images = h5py.File(self.image_path, 'r')
        index = self.coco_id_to_h5index[image_id]
        img = self.processed_images['features'][index].astype('float32')
        return torch.from_numpy(img)

    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]

        q, q_length = self.questions[item]
        a_indices = self.answer_indices[item]
        a_values = self.answer_values[item]
        a_length = self.answer_lengths[item]
        image_id = self.coco_ids[item]
        v = self._load_image(image_id)


        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return v, q, a_indices, a_values, a_length, item, q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)

period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}',
                '(', ')', '=', '+', '\\', '_', '-',
                '>', '<', '@', '`', ',', '?', '!']
contractions = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve":
    "could've", "couldnt": "couldn't", "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've", "didnt": "didn't", "doesnt":
    "doesn't", "dont": "don't", "hadnt": "hadn't", "hadnt've":
    "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent":
    "haven't", "hed": "he'd", "hed've": "he'd've", "he'dve":
    "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im":
    "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd", "itd've":
    "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've":
    "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat":
    "'ow's'at", "shant": "shan't", "shed've": "she'd've", "she'dve":
    "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt":
    "shouldn't", "shouldnt've": "shouldn't've", "shouldn'tve":
    "shouldn't've", "somebody'd": "somebodyd", "somebodyd've":
    "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll":
    "somebody'll", "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd":
    "something'd", "somethingd've": "something'd've", "something'dve":
    "something'd've", "somethingll": "something'll", "thats":
    "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres":
    "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve":
    "they'd've", "theyll": "they'll", "theyre": "they're", "theyve":
    "they've", "twas": "'twas", "wasnt": "wasn't", "wed've":
    "we'd've", "we'dve": "we'd've", "weve": "we've", "werent":
    "weren't", "whatll": "what'll", "whatre": "what're", "whats":
    "what's", "whatve": "what've", "whens": "when's", "whered":
    "where'd", "wheres": "where's", "whereve": "where've", "whod":
    "who'd", "whod've": "who'd've", "who'dve": "who'd've", "wholl":
    "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't", "wouldve":
    "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll":
    "y'all'll", "y'allll": "y'all'll", "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd":
    "you'd", "youd've": "you'd've", "you'dve": "you'd've", "youll":
    "you'll", "youre": "you're", "youve": "you've"
}

manual_map = { 'none': '0',
              'zero': '0',
              'one': '1',
              'two': '2',
              'three': '3',
              'four': '4',
              'five': '5',
              'six': '6',
              'seven': '7',
              'eight': '8',
               'nine': '9',
              'ten': '10'}
articles = ['a', 'an', 'the']


def prepare_questions(questions_json):
    """
        Tokenize and normalize questions from a given question json in the usual VQA_dataset format.
    """
    questions = [q['question'] for q in questions_json['questions']]
    for question in questions:
        if question[-1] != '?':
            raise Exception("Final char not ?")
        question = question.lower()[:-1]
        yield question.split(' ')


def prepare_answers(answers_json):
    """
        Normalize answers from a given answer json in the usual VQA_dataset format.
    """
    answers = [[a['answer'] for a in ans_dict['answers']] for ans_dict in answers_json['annotations']]

    # The only normalization that is applied to both machine generated answers as well as
    # ground truth answers is replacing most punctuation with space (see [0] and [1]).
    # Since potential machine generated answers are just taken from most common answers, applying the other
    # normalizations is not needed, assuming that the human answers are already normalized.
    # [0]: http://visualqa.org/evaluation.html
    # [1]: https://github.com/VT-vision-lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96
    # https://github.com/GT-Vision-Lab/VQA/blob/3849b1eae04a0ffd83f56ad6f70ebd0767e09e0f/PythonEvaluationTools/vqaEvaluation/vqaEval.py#L96

    for answer_list in answers:
        yield list(map(preprocess_answer, answer_list))

def preprocess_answer(answer):
    answer = process_digit_article(process_punctuation(answer))
    answer = answer.replace(',', '')
    return answer

def process_punctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) \
                or (re.search(comma_strip, inText) != None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = period_strip.sub("", outText, re.UNICODE)
    return outText

def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    outText = ' '.join(outText)
    return outText

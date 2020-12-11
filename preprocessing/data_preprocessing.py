import json
import os
import os.path
import re

from PIL import Image
import h5py
import torch
import torch.utils.data


class VQA_dataset(torch.utils.data.Dataset):
    """ VQA_dataset dataset, open-ended """
    def __init__(self, data_paths, other_paths, answerable_only=False):
        super(VQA_dataset, self).__init__()

        questions_path = data_paths['questions']
        answers_path = data_paths['answers']
        vocabulary_path = other_paths['vocab_path']
        with open(questions_path, 'r') as fd:
            questions_json = json.load(fd)
        with open(answers_path, 'r') as fd:
            answers_json = json.load(fd)
        with open(vocabulary_path, 'r') as fd:
            vocab_json = json.load(fd)
        self._check_integrity(questions_json, answers_json)

        # vocab
        self.vocab = vocab_json
        self.question_token_to_index = self.vocab['question']
        self.answer_token_to_index = self.vocab['answer']

        # questions - Turn a question into a padded vector of vocab indices and a question length
        questions = list(prepare_questions(questions_json))
        self.questions = [self._encode_question(q) for q in questions]
        # answers - Turn an answer into a vector of counts of all possible answers
        answers = list(prepare_answers(answers_json))
        self.answers = [self._encode_answers(a) for a in answers]

        # v - for each question store image id in coco_ids
        self.image_features_path = other_paths['processed_imgs']
        self.coco_id_to_h5index = self._create_coco_id_to_index()
        self.coco_ids = [q['image_id'] for q in questions_json['questions']]

        # only use questions that have at least one answer?
        self.answerable_only = answerable_only
        if self.answerable_only:
            self.answerable = self._find_answerable()

    @property
    def max_question_length(self):
        if not hasattr(self, '_max_length'):
            self._max_length = max(map(len, self.questions))
        return self._max_length

    @property
    def num_tokens(self):
        return len(self.question_token_to_index) + 1  # add 1 for <unknown> token at index 0

    def _create_coco_id_to_index(self):
        """ Create a mapping from a COCO image id into the corresponding index into the h5 file """
        with h5py.File(self.image_features_path, 'r') as features_file:
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
        for i, answers in enumerate(self.answers):
            answer_has_index = len(answers.nonzero()) > 0
            # store the indices of anything that is answerable
            if answer_has_index:
                answerable.append(i)
        return answerable

    def _encode_question(self, question):
        """ Turn a question into a vector of indices and a question length """
        # TODO change to pytorch padding
        vec = torch.zeros(self.max_question_length).long()
        for i, token in enumerate(question):
            index = self.question_token_to_index.get(token, 0)
            vec[i] = index
        return vec, len(question)

    def _encode_answers(self, answers):
        """ Turn an answer into a vector """
        # TODO change to sparse vector!
        # answer vec will be a vector of answer counts to determine which answers will contribute to the loss.
        # this should be multiplied with 0.1 * negative log-likelihoods that a model produces and then summed up
        # to get the loss that is weighted by how many humans gave that answer
        answer_vec = torch.zeros(len(self.answer_token_to_index))
        for answer in answers:
            index = self.answer_token_to_index.get(answer)
            if index is not None:
                answer_vec[index] += 1
        return answer_vec

    def _load_image(self, image_id):
        """ Load an image """
        if not hasattr(self, 'features_file'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.features_file = h5py.File(self.image_features_path, 'r')
        index = self.coco_id_to_h5index[image_id]
        dataset = self.features_file['features']
        img = dataset[index].astype('float32')
        return torch.from_numpy(img)

    def __getitem__(self, item):
        if self.answerable_only:
            # change of indices to only address answerable questions
            item = self.answerable[item]

        q, q_length = self.questions[item]
        a = self.answers[item]
        image_id = self.coco_ids[item]
        v = self._load_image(image_id)
        # since batches are re-ordered for PackedSequence's, the original question order is lost
        # we return `item` so that the order of (v, q, a) triples can be restored if desired
        # without shuffling in the dataloader, these will be in the order that they appear in the q and a json's.
        return v, q, a, item, q_length

    def __len__(self):
        if self.answerable_only:
            return len(self.answerable)
        else:
            return len(self.questions)


# this is used for normalizing questions
_special_chars = re.compile('[^a-z0-9 ]*')

# these try to emulate the original normalization scheme for answers
_period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
_comma_strip = re.compile(r'(\d)(,)(\d)')
_punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
_punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
_punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))




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

    def process_punctuation(s):
        # the original is somewhat broken, so things that look odd here might just be to mimic that behaviour
        # this version should be faster since we use re instead of repeated operations on str's

        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        res = s.strip()

        return res


    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))


class CocoImages(torch.utils.data.Dataset):
    def __init__(self, path, transform=None):
        """
        Dataset for MSCOCO images located in a folder on the filesystem

        :param path: path to image dir
        :param transform: transforms.Compose, transformations to apply
        """
        super(CocoImages, self).__init__()
        self.path = path
        self.id_to_filename = self._find_images()
        self.sorted_ids = sorted(self.id_to_filename.keys())  # used for deterministic iteration order
        print('found {} images in {}'.format(len(self), self.path))
        self.transform = transform

    def _find_images(self):
        id_to_filename = {}
        for filename in os.listdir(self.path):
            if not filename.endswith('.jpg'):
                continue
            id_and_extension = filename.split('_')[-1]
            id = int(id_and_extension.split('.')[0])
            id_to_filename[id] = filename
        return id_to_filename

    def __getitem__(self, item):
        id = self.sorted_ids[item]
        path = os.path.join(self.path, self.id_to_filename[id])
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return id, img

    def __len__(self):
        return len(self.sorted_ids)


class Composite(torch.utils.data.Dataset):
    """ Dataset that is a composite of several Dataset objects. Useful for combining splits of a dataset. """
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, item):
        current = self.datasets[0]
        for d in self.datasets:
            if item < len(d):
                return d[item]
            item -= len(d)
        else:
            raise IndexError('Index too large for composite dataset')

    def __len__(self):
        return sum(map(len, self.datasets))



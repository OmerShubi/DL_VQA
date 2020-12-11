import os


def path_for(train=False, val=False, test=False, question=False, answer=False):
    task = 'OpenEnded'
    dataset = 'mscoco'
    qa_path = 'vqa'  # directory containing the question and annotation jsons
    base_path = '/datashare'
    assert train + val + test == 1
    assert question + answer == 1
    assert not (test and answer), 'loading answers from test split not supported'  # if you want to eval on test, you need to implement loading of a VQA Dataset without given answers yourself
    if train:
        split = 'train2014'
    elif val:
        split = 'val2014'
    else:
        split = 'test2015'
    if question:
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        fmt = 'v2_{1}_{2}_annotations.json'
    s = fmt.format(task, dataset, split)
    s = os.path.join(base_path, s)
    return os.path.join(qa_path, s)

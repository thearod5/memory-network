"""
Responsible for parsing the sample training data
"""
from functools import reduce

from constants import PATH_TO_SAMPLE_DATA


def parse_stories(lines):
    """
     - Parse stories provided in the bAbI tasks format
     - A story starts from line 1 to line 15. Every 3rd line,
       there is a question &amp;amp;amp;amp;amp; answer.
     - Function extracts sub-stories within a story and
       creates tuples
    :param lines:
    :return:
    """

    data = []
    story = []
    for line in lines:
        # line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            # reset story when line ID=1 (start of new story)
            story = []
        if '\t' in line:
            # this line is tab separated Q, A &amp;amp;amp;amp;amp; support fact ID
            q, a, supporting = line.split('\t')
            # Provide all the sub-stories till this question
            substory = " ".join([x for x in story if x])
            # A story ends and is appended to global story data-set
            data.append((substory, q, a))
            story.append('')
        else:
            # this line is a sentence of story
            story.append(line)
    return data


def get_stories(f):
    """
    argument: filename
    returns list of all stories in the argument data-set file
    :param f:
    :return:
    """

    data = parse_stories(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data]
    return data


def load_sample_data():
    train_file = open(PATH_TO_SAMPLE_DATA.format('train'))
    test_file = open(PATH_TO_SAMPLE_DATA.format('test'))

    train = get_stories(train_file)
    test = get_stories(test_file)
    return train, test

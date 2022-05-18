import json
from gensim.models import KeyedVectors

a = KeyedVectors.load_word2vec_format('../mappings/glove-wiki-gigaword-50.txt')


def main(word, class_names):
    fd = {}
    for i in range(len(class_names)):
        fd[str(i + 1)] = str(a.similarity(word, class_names[i]))

    s = json.dumps(fd)
    fp = open('../mappings/' + word + '.json', 'w+')
    fp.write(s)
    fp.close()

class_names = ['crab', 'eel', 'animal', 'fish', 'shells', 'starfish', 'plant', 'robot', 'bag', 'bottle', 'branch',
               'can', 'clothing', 'container', 'cup', 'net', 'pipe', 'rope', 'wrapper', 'tarp', 'trash', 'wreckage']
word = 'tangle'

main(word, class_names)
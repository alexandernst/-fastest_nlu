import unicodedata
import re

def normalize(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()

def tokenize(s):
    return [x for x in re.split(r'[\s,.!?;:([\]\'"¡¿)/]+', s) if x]

class Encoder:
    def __init__(self, processor=None):
        self.processor = processor or (lambda s: tokenize(normalize(s)))
        self.featureMap = {}
        self.numFeature = 0
        self.intentMap = {}
        self.intents = []

    def learnIntent(self, intent):
        if intent not in self.intentMap:
            self.intentMap[intent] = len(self.intents)
            self.intents.append(intent)

    def learnFeature(self, feature):
        if feature not in self.featureMap:
            self.featureMap[feature] = self.numFeature
            self.numFeature += 1

    def encodeText(self, text, learn=False):
        d = {}
        keys = []
        features = self.processor(text)
        for feature in features:
            if learn:
                self.learnFeature(feature)
            index = self.featureMap.get(feature)
            if index is not None and index not in d:
                d[index] = 1
                keys.append(index)
        return keys

    def encode(self, text, intent, learn=False):
        if learn:
            self.learnIntent(intent)
        return {
            'input': self.encodeText(text, learn),
            'output': self.intentMap.get(intent),
        }

    def encodeCorpus(self, corpus):
        result = {'train': [], 'validation': []}
        for data in corpus:
            utterances = data.get('utterances')
            intent = data.get('intent')
            if utterances:
                for utterance in utterances:
                    result['train'].append(self.encode(utterance, intent, True))
            tests = data.get('tests')
            if tests:
                for test in tests:
                    result['validation'].append(self.encode(test, intent))
        return result

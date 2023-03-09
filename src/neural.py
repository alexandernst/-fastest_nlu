from .encoder import Encoder
import time

def defaultLogFn(status, time_):
    print(f"Epoch {status['iterations']} loss {status['error']} time {time_}ms")

def runInputPerceptron(weights, input):
    sum_ = 0
    for key in input:
        sum_ += weights[key]
    return sum_ if sum_ > 0 else 0


class Neural:
    def __init__(self, settings = {}):
        self.settings = settings
        self.settings.setdefault('maxIterations', 150)
        self.settings.setdefault('learningRate', 0.002)
        self.logFn = defaultLogFn if self.settings.get('log', False) else self.settings.get('log')

    def prepareCorpus(self, corpus):
        self.encoder = self.settings.get('encoder', Encoder(self.settings.get('processor')))
        self.encoded = self.encoder.encodeCorpus(corpus)

    def initialize(self, corpus):
        self.prepareCorpus(corpus)
        self.status = {'error': float('inf'), 'iterations': 0}
        self.perceptrons = [{
            'intent': intent,
            'id': self.encoder.intentMap.get(intent),
            'weights': [0.0] * self.encoder.numFeature
        } for intent in self.encoder.intents]

    def trainPerceptron(self, perceptron, data):
        learningRate = self.settings['learningRate']
        weights = perceptron['weights']
        error = 0
        for chunk in data:
            input_ = chunk['input']
            output = chunk['output']
            actualOutput = runInputPerceptron(weights, input_)
            expectedOutput = 1 if output == perceptron['id'] else 0
            currentError = expectedOutput - actualOutput
            if currentError:
                error += currentError ** 2
                change = currentError * learningRate
                for v in input_:
                    weights[v] += change
        return error

    def train(self, corpus):
        self.initialize(corpus if isinstance(corpus, list) else corpus['data'])
        data = self.encoded['train']
        maxIterations = self.settings['maxIterations']
        while self.status['iterations'] < maxIterations:
            hrstart = time.time()
            self.status['iterations'] += 1
            totalError = 0
            for perceptron in self.perceptrons:
              totalError += self.trainPerceptron(perceptron, data)
            self.status['error'] = totalError / (len(data) * len(self.perceptrons))
            if self.logFn:
                hrend = time.time()
                self.logFn(self.status, int((hrend - hrstart) * 1000))
        return self.status

    def run(self, text):
        input_ = self.encoder.encodeText(text)
        result = []
        for perceptron in self.perceptrons:
            score = runInputPerceptron(perceptron['weights'], input_)
            if score:
                result.append({'intent': perceptron['intent'], 'score': score})
        if not result:
            return [{'intent': 'None', 'score': 0}]
        return sorted(result, key=lambda x: x['score'], reverse=True)

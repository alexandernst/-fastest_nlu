import json, time
from src.neural import Neural
from benchmark.bench import Bench

def execFn(net, data):
    good = 0
    for item in data:
        classifications = net.run(item["utterance"])
        if classifications[0]["intent"] == item["intent"]:
            good += 1
    return {"good": good, "total": len(data)}

def measureCorpus(corpus):
    test_data = []
    for item in corpus['data']:
        for test in item['tests']:
            test_data.append({'utterance': test, 'intent': item['intent']})
    net = Neural({ 'log': True })
    hrstart = time.perf_counter()
    net.train(corpus)
    hrend = time.perf_counter()
    print(f'Time for training: {hrend - hrstart}s')
    result = execFn(**{'net': net, 'data': test_data})
    print(f'Accuracy: {(result["good"] * 100) / result["total"]}')
    bench = Bench({'transactionsPerRun': len(test_data)})
    bench_result = bench.measure(execFn, lambda: {'net': net, 'data': test_data})
    print(f'Transactions per second: {bench_result}')


print("English corpus")
with open("./benchmark/corpus-massive-en.json", "r") as f:
    corpusEn = json.load(f)
measureCorpus(corpusEn)

print("\nSpanish corpus")
with open("./benchmark/corpus-massive-es.json", "r") as f:
    corpusEs = json.load(f)
measureCorpus(corpusEs)

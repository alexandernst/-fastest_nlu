import time

class Bench:
    def __init__(self, settings={}):
        self.duration = settings.get("duration", 10000)
        self.transactionsPerRun = settings.get("transactionsPerRun", 1)

    def measure(self, fn, initfn):
        initValue = initfn()
        hrstart = time.perf_counter()
        runs = 0
        elapsed = 0
        resultIteration = None
        while elapsed < self.duration:
            resultIteration = fn(**initValue)
            runs += 1
            hrend = time.perf_counter()
            elapsed = (hrend - hrstart) * 1000
        timePerTransaction = elapsed / (runs * self.transactionsPerRun)
        print("Runs", runs)
        return 1000 / timePerTransaction

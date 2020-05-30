import functools
import copy

class Memoize:
    def __init__(self, f):
        self.f = f
        self.memo = {}
    def __call__(self, *args):
        if not args in self.memo:
            self.memo[args] = self.f(*args)
        # Warning: You may wish to do a deepcopy here if returning objects
        return copy.deepcopy(self.memo[args])

def empty_counter():
    print("empty")

@functools.lru_cache(maxsize=128)
def fn1(x):
    print("fn1")
    empty_counter()
    return x


def fn2(x):
    print("fn2")
    empty_counter()
    return x

if __name__ == "__main__":
    print(fn1(5))
    print(fn1(5)) # WARNING: Does not print fn1!
    print("\n")
    print(fn2(6))
    print(fn2(6))

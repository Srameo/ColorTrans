import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        ans = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ + f" spending time: {end - start}")
        return ans

    return wrapper

# instruction: python that counts words in a file
import collections
import sys


def count_words(path: str) -> collections.Counter:
    with open(path, encoding="utf-8") as handle:
        words = handle.read().lower().split()
    return collections.Counter(words)


if __name__ == "__main__":
    for word, n in count_words(sys.argv[1]).most_common(10):
        print(word, n)

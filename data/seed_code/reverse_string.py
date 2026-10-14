# instruction: python to reverse a string
def reverse(text: str) -> str:
    """Return the characters of text in reverse order."""
    return text[::-1]


if __name__ == "__main__":
    print(reverse("hello world"))

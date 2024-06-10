def is_palindrome(s):
    print(f"s={s} s[::-1]={s[::-1]}")
    return s == s[::-1]

def palindrome_pairs(words):
    word_dict = {word: i for i, word in enumerate(words)}
    result = []

    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix, suffix = word[:j], word[j:]

            if is_palindrome(prefix):
                reversed_suffix = suffix[::-1]
                if reversed_suffix in word_dict and word_dict[reversed_suffix] != i:
                    result.append([word_dict[reversed_suffix], i])

            if j != len(word) and is_palindrome(suffix):
                reversed_prefix = prefix[::-1]
                if reversed_prefix in word_dict and word_dict[reversed_prefix] != i:
                    result.append([i, word_dict[reversed_prefix]])

    return result

# Example usage:
words = ["abcd", "dcba", "lls", "s", "sssll"]
print(palindrome_pairs(words))
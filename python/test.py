def swap(seq, i, j):
    seq[i], seq[j] = seq[j], seq[i]

def generate_permutations(seq, seqLen, resLen):
    c = [0] * seqLen
    yield seq[:resLen]

    i = 0
    while i < seqLen:
        if c[i] < i:
            if i % 2 == 0:
                swap(seq, 0, i)
            else:
                swap(seq, c[i], i)
            yield seq[:resLen]
            c[i] += 1
            i = 0
        else:
            c[i] = 0
            i += 1

def permutations(seq, resLen=None):
    if not resLen: resLen = len(seq)
    return generate_permutations(seq, len(seq), resLen)

for p in permutations(['a', 'b', 'c', 'd']): print(p)
# for p in permutations([1,2,3],2): print(p)
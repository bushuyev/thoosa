class Solution:
    def shortestPalindrome(self, s: str) -> str:
        print(f" str={[f'{c:02}' for c in s]}")
        t = s #+ "#" + s[::-1] + "$"
        n = len(t)
        next = [0] * n
        next[0] = -1
        i, j = 2, 0
        while i < n:
            if t[i - 1] == t[j]:
                j += 1
                next[i] = j
                i += 1
            elif j:
                j = next[j]
            else:
                next[i] = 0
                i += 1

        print(f"next={[f'{n:02}' for n in next]}")

        return s[::-1][: -next[n-1]] + s


print(Solution().shortestPalindrome("abcdab"))
#abcdeb
#abcdab
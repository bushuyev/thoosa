from functools import cache
from typing import List


class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:

        @cache
        def dfs(i, j, k):
            print(f"0 i: {i}, j: {j}, k: {k}")
            if i > j:
                return 0

            while i < j and boxes[j] == boxes[j - 1]:
                j = j - 1
                k = k + 1

            print(f"1 i: {i}, j: {j}, k: {k}")

            if i + 1 < j:
                ans = dfs(i, j - 1, 0)
            else:
                ans = 0

            v = (k + 1) * (k + 1)
            print(f"i={i} j={j} k={k} v: {v}  ans={ans}")
            ans = ans + v

            for h in range(i, j):
                if boxes[h] == boxes[j]:
                    print(f"h={h}, j={j}")

                    ans_1 = dfs(h + 1, j - 1, 0)
                    ans_2 = dfs(i, h, k + 1)

                    print(f"ans={ans}, ans_1={ans_1}, ans_2={ans_2}")

                    ans = max(ans, ans_1 + ans_2)
            return ans


        n = len(boxes)
        ans = dfs(0, n - 1, 0)
        dfs.cache_clear()
        return ans


# s = Solution().removeBoxes([1,3,2,2,2,3,4,3,1])
s = Solution().removeBoxes([1,2,1])
print(f"s={s}")
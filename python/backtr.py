def generate_pairs_indices(length, num_pairs):
    if length < 2 or num_pairs < 1:
        return []

    def backtrack(curr_index, remaining_pairs, combination):
        if remaining_pairs == 0:
            combinations.append(combination.copy())
            return

        for next_index in range(curr_index + 1, length):
            combination.append((curr_index, next_index))
            backtrack(next_index, remaining_pairs - 1, combination)
            combination.pop()

    combinations = []
    backtrack(0, num_pairs, [])
    return combinations

# Example usage:
length = 5
num_pairs = 3
pairs_indices = generate_pairs_indices(length, num_pairs)

print(f"pairs_indices={pairs_indices}")
# for combination in pairs_indices:
#     print(", ".join([f"{index},{index + 1}" for index in combination]))
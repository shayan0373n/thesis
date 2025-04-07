import numpy as np

def find_plateaus(arr, min_length):
    arr = np.asarray(arr)
    
    # Find indices where the value changes
    change_idx = np.diff(arr).nonzero()[0]
    
    # Compute start and end indices for each run
    starts = np.concatenate(([0], change_idx + 1))
    ends = np.concatenate((change_idx, [arr.size - 1]))
    
    # Return runs where the length meets or exceeds min_length
    return [(s, e, arr[s]) for s, e in zip(starts, ends) if (e - s + 1) >= min_length]

# def find_plateaus(arr, min_length):
#     plateaus = []
#     curr_start = 0
#     curr_val = arr[0]
#     for i in range(1, len(arr)):
#         if arr[i] != curr_val:
#             if i - curr_start >= min_length:
#                 plateaus.append((curr_start, i - 1, curr_val))
#             curr_start = i
#             curr_val = arr[i]
#     if len(arr) - curr_start >= min_length:
#         plateaus.append((curr_start, len(arr) - 1, curr_val))
#     return plateaus
        

# Example usage:
if __name__ == '__main__':
    data = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    print(find_plateaus(data, 3))
    # Expected output: [(0, 2, 1), (5, 8, 3), (9, 13, 4)]
    print(find_plateaus(data, 4))
    # Expected output: [(5, 8, 3), (9, 13, 4)]
    print(find_plateaus(data, 5))
    # Expected output: [(9, 13, 4)]
    print(find_plateaus(data, 6))
    # Expected output: []

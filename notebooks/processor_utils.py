import numpy as np

def one_hot_encode(data):
    """ One-hot encode a specified attribute."""
    
    unique_values = set()
    for row in data:
        for term in row:
            if isinstance(term, str) or isinstance(term, str):
                unique_values.add(term)
            else:
                raise ValueError(f"Invalid term type: {type(term)}. Expected string.")

    # Map unique values to integers
    value_to_int = {value: idx for idx, value in enumerate(unique_values)}

    # Convert attribute_data to integer indices
    integer_encoded = []
    for row in data:
        row_indices = [value_to_int[term] for term in row if term in value_to_int]
        integer_encoded.append(row_indices)

    # Create one-hot encoded array
    one_hot_encoded = np.zeros((len(data), len(unique_values)))
    for i, row_indices in enumerate(integer_encoded):
        one_hot_encoded[i, row_indices] = 1

    return one_hot_encoded
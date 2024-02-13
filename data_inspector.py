
import datasets

"""coarse_label (ClassLabel): Coarse class label. Possible values are:
'ABBR' (0): Abbreviation.
'ENTY' (1): Entity.
'DESC' (2): Description and abstract concept.
'HUM' (3): Human being.
'LOC' (4): Location.
'NUM' (5): Numeric value.
"""

label_mappings = {'ABBR': 0, 'ENTY': 1, 'DESC': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}
reversed_label_mappings = {v: k for k, v in label_mappings.items()}

def print_data_sample(ds, text_field, label_field, print_count=5, label_mappings=None):
    count_by_label = {e: 0 for e in set(ds[label_field])}
    print(count_by_label)
    for label in count_by_label:
        for example, example_label in zip(ds[text_field], ds[label_field]):
            if example_label == label:
                if count_by_label[label] == -1:
                    continue
                count_by_label[label] += 1
                label_text = label_mappings[label] if label_mappings else label
                print(f"{label_text}:  {example}")
                if count_by_label[example_label] == print_count:
                    count_by_label[example_label] = -1

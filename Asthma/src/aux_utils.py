import re
import numpy as np

def list_string_to_list(data):
    input_string = str(data).split()
    cleaned_string = ' '.join(input_string).replace("\n", " ").replace("[", "").replace("]", "").replace("'", "").replace("\\n", " ")
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string).strip()  # Remove extra whitespaces
    # Convert to list of floats
    float_list = [float(x) for x in cleaned_string.split()]
    return np.array(float_list)
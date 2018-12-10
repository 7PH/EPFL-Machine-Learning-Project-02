def load_from_file(filepath, clean_fn=None):
    """
    Load a file from the filesystem. If provided, map each line with clean_fn function

    :param filepath: File path
    :param clean_fn: Clean function
    :return: List of every lines
    """
    lines = []
    with open(filepath, 'rb') as file:
        for line in file:
            line = line.decode("UTF-8")[:-1]
            if clean_fn is not None:
                line = clean_fn(line)
            lines.append(line)
    return lines

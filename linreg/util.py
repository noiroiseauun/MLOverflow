def clean_tokenize(text):
    return text.replace('\n', ' ').replace('\'', '').split()

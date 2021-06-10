from typing import Dict
from nltk.tokenize import sent_tokenize
from functools import reduce
import json


class Page:
    def __init__(self, json_data: Dict):
        self.page_info = json_data["page"]
        self.tokens_info = json_data["tokens"]
        self.token_list = [token["text"] for token in self.tokens_info]
        self.text = " ".join(self.token_list)
        self.sentences = Page.split_to_sentences(self.text)
        self.sent_token_list = [sent.split(" ") for sent in self.sentences]

    @classmethod
    def split_to_sentences(cls, text):
        return sent_tokenize(text)

    def count_tokens(self):
        return len(self.token_list)

    def count_sent_tokens(self):
        number: int = reduce(lambda x, y: x + y, [len(sent) for sent in self.sent_token_list])
        return number

    def __str__(self):
        return self.__dict__.__str__()


class Sentence:
    def __init__(self):
        pass


class TokenList:
    def __init__(self, tokens: Dict):
        self.token_list = [token["text"] for token in tokens]


if __name__ == "__main__":
    with open("../test_data/pdf_structure.json", "r") as fr:
        data = fr.read()
        json_data = json.loads(data)
        page = Page(json_data[0])
        print(page.count_tokens())
        print(page.count_sent_tokens())

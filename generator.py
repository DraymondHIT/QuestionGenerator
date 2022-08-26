import stanza
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from structure import SentenceStructure
import random


class QuestionGenerator:
    def __init__(self):
        self.parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,ner,lemma,depparse')
        self.openie = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")

    def filter_from_openie_result(self, tokens, results):
        results = results["verbs"]
        filtered = []
        for result in results:
            # 1. too many 'O's
            if result["tags"].count("O") > len(result["tags"]) // 2:
                continue

            # 2. ignore auxiliary verb(have, be and do sometimes have meanings)
            if result["verb"] in {"may", "shall", "will", "can", "must"}:
                continue

            # 3. judge whether have, be or do has meanings
            if list(tokens)[result["tags"].index("B-V") + 1].upos == "VERB":
                continue

            filtered.append(result)

        return filtered

    def is_head_of(self, token1, token2, doc):
        while token2.head != 0:
            token2 = list(doc)[token2.head-1]
            if token1.id == token2.id:
                return True
        return False

    def get_wh_word(self, word_doc, token_doc, subject):
        entities = []
        for index, token in enumerate(subject.getValue()):
            if list(token_doc)[token.id-1].ner != "O":
                entities.append(token)

        if len(entities) == 0:
            return "What"

        if len(entities) == 1:
            token = entities[0]
        elif len(entities) == 2:
            entity1 = entities[0]
            entity2 = entities[1]
            if self.is_head_of(entity1, entity2, word_doc):
                token = entity1
            else:
                token = entity2

        if list(token_doc)[token.id-1].ner.__contains__("PERSON"):
            return "Who"
        elif list(token_doc)[token.id-1].ner.__contains__("DATE"):
            return "What time"
        elif list(token_doc)[token.id-1].ner.__contains__("LOC"):
            return "What place"
        else:
            return "What"

    def get_wh_word_in_tmp(self, tmp):
        for token in tmp.getValue():
            if token.text.lower() in {"every", "each"}:
                return "How often"
            elif token.text.lower() in {"for"}:
                return "How long"
            elif token.text.lower() in {"from", "until"}:
                return token.text.capitalize() + " " + "when"
        return "When"

    def lower_when_QG(self, token_doc, subject):
        start = subject.getStart()
        if list(token_doc)[start.id-1].ner != "O":
            return subject.__str__()
        else:
            _subject = [token.text for token in subject.getValue()[1:]]
            _subject = " ".join(_subject)
            return start.text.lower() + " " + _subject

    def generate_question(self, token_doc, structure):
        questions = []
        original_verb = structure.verb
        if not structure.mod.isEmpty():
            auxiliary = structure.mod
            verb_lemma = structure.verb
        else:
            if structure.verb.__str__() in {"is", "are", "was", "were"}:
                auxiliary = structure.verb.__str__()
                verb_lemma = ""
            else:
                if structure.verb.xpos() == "VBD":
                    auxiliary = "did"
                elif structure.verb.xpos() == "VBP":
                    auxiliary = "do"
                else:
                    auxiliary = "does"
                verb_lemma = structure.verb.lemma()

        # 1. subject question
        if len(structure.mods) > 0:
            mod = random.choice(structure.mods)
            _mod = self.lower_when_QG(token_doc, mod)
        else:
            _mod = ""
        wh_word = self.get_wh_word(structure.doc, token_doc, structure.subject)
        if structure.verb.xpos() in {"VBG", "VBN"}:
            print(wh_word + " " + auxiliary.__str__() + " " + original_verb.__str__() + " " + structure.object.__str__() + " " + _mod + "?")
        else:
            print(wh_word + " " + original_verb.__str__() + " " + structure.object.__str__() + " " + _mod + "?")

        # 2. object question
        if not structure.object.isEmpty():
            if len(structure.mods) > 0:
                mod = random.choice(structure.mods)
                _mod = self.lower_when_QG(token_doc, mod)
            else:
                _mod = ""
            wh_word = self.get_wh_word(structure.doc, token_doc, structure.object)
            _subject = self.lower_when_QG(token_doc, structure.subject)

            # if object starts with PREP
            if structure.object.getStart().upos == "ADP":
                prep = structure.object.getStart().text.capitalize()
                wh_word = wh_word.lower()
                # convert 'who' to 'whom'
                if wh_word == "who":
                    wh_word = "whom"
                print(prep + " " + wh_word + " " + auxiliary.__str__() + " " + _subject + " " + verb_lemma.__str__() + " " + _mod + "?")
            else:
                print(wh_word + " " + auxiliary.__str__() + " " + _subject + " " + verb_lemma.__str__() + " " + _mod + "?")

        # 3. TMP question
        # Three main types: When, How long and How often
        # When: in, at, on, ...
        # How long: for
        # How often: every, each
        if not structure.tmp.isEmpty():
            wh_word = self.get_wh_word_in_tmp(structure.tmp)
            _subject = self.lower_when_QG(token_doc, structure.subject)
            print(wh_word + " " + auxiliary.__str__() + " " + _subject + " " + verb_lemma.__str__() + " " + structure.object.__str__() + "?")

    def create(self, sentence):
        doc = self.parser(sentence)
        token_doc = doc.sentences[0].tokens
        word_doc = doc.sentences[0].words
        openie_result = self.openie.predict(sentence=sentence)

        # filter some bad results
        filtered_result = self.filter_from_openie_result(word_doc, openie_result)
        print(filtered_result)

        # break up sentences into many elements
        structure = SentenceStructure(word_doc, filtered_result)
        print(structure)

        self.generate_question(token_doc, structure)


generator = QuestionGenerator()
generator.create("Charles W. Eliot, president 1869-1909, removed Christianity from the school curriculum.")

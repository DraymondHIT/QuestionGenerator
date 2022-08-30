import stanza
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
from structure import SentenceStructure, PhraseStructure
import random
from collections import Counter


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

    def delete_continuous_space(self, text):
        return " ".join(text.split())

    def get_center_word(self, word_doc, subject):
        tokens = subject.getValue()
        limit = range(tokens[0].id, tokens[-1].id+1)
        center_words = []
        for token in tokens:
            temp = token
            while temp.head in limit:
                temp = list(word_doc)[temp.head-1]
            center_words.append(temp)
        counts = Counter(center_words)
        center_word = counts.most_common(1)[0][0]

        return center_word

    def get_entity(self, word_doc, token_doc, subject):
        center_token = self.get_center_word(word_doc, subject)
        token_list = list(token_doc)
        tag = token_list[center_token.id - 1].ner
        if tag == "O":
            return None
        elif tag.startswith("S-"):
            return center_token.text
        else:
            entity = PhraseStructure()
            entity.right_add(center_token)
            index1 = index2 = center_token.id - 1
            while token_list[index1].ner != "B" + tag[1:] and index1 > 0:
                index1 -= 1
                entity.left_add(token_list[index1])
            while token_list[index2].ner != "E" + tag[1:] and index2 < len(token_list) - 1:
                index2 += 1
                entity.right_add(token_list[index2])
            return entity.__str__()

    def get_wh_word(self, word_doc, token_doc, subject):

        center = self.get_center_word(word_doc, subject)
        if center.upos in {"NOUN", "PROPN", "NUM"}:
            if list(token_doc)[center.id - 1].ner.__contains__("PERSON"):
                return "Who"
            elif list(token_doc)[center.id - 1].ner.__contains__("DATE"):
                return "What time"
            elif list(token_doc)[center.id - 1].ner.__contains__("LOC"):
                return "What place"
            elif list(token_doc)[center.id - 1].ner.__contains__("CARDINAL"):
                return "How many"
            elif list(token_doc)[center.id - 1].ner.__contains__("PERCENT"):
                return "How much"
            else:
                return "What"
        elif center.upos in {"ADJ", "ADV"}:
            return "How"
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
        qas = []
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
        neg = structure.neg

        # 1. subject question
        _answer = self.get_entity(structure.doc, token_doc, structure.subject)
        if _answer is not None:
            if len(structure.mods) > 0:
                mod = random.choice(structure.mods)
                _mod = self.lower_when_QG(token_doc, mod)
            else:
                _mod = ""
            wh_word = self.get_wh_word(structure.doc, token_doc, structure.subject)

            if structure.verb.xpos() in {"VBG", "VBN"}:
                question = wh_word + " " + auxiliary.__str__() + " " + neg.__str__() + " " + original_verb.__str__() + " " + structure.object.__str__() + " " + _mod + "?"
            else:
                question = wh_word + " " + original_verb.__str__() + " " + neg.__str__() + " " + structure.object.__str__() + " " + _mod + "?"
            qas.append({"question": self.delete_continuous_space(question), "answer": _answer})

        # 2. object question
        if not structure.object.isEmpty() and structure.sconj is None:
            _answer = self.get_entity(structure.doc, token_doc, structure.object)
            if _answer is not None:
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
                    question = prep + " " + wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + " " + _mod + "?"
                else:
                    question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + " " + _mod + "?"
                qas.append({"question": self.delete_continuous_space(question), "answer": _answer})

        # 3. TMP question
        # Three main types: When, How long and How often
        # When: in, at, on, ...
        # How long: for
        # How often: every, each
        if not structure.tmp.isEmpty():
            # sometimes tmp is not a real tmp
            center = self.get_center_word(structure.doc, structure.tmp)
            if not list(token_doc)[center.id - 1].ner.__contains__("DATE"):
                wh_word = self.get_wh_word(structure.doc, token_doc, structure.tmp)
                _answer = self.get_entity(structure.doc, token_doc, structure.tmp)
                _subject = self.lower_when_QG(token_doc, structure.subject)

                # if object starts with PREP
                if structure.tmp.getStart().upos == "ADP":
                    prep = structure.tmp.getStart().text.capitalize()
                    wh_word = wh_word.lower()
                    # convert 'who' to 'whom'
                    if wh_word == "who":
                        wh_word = "whom"
                    question = prep + " " + wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + "?"
                else:
                    question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + "?"
                qas.append({"question": self.delete_continuous_space(question), "answer": _answer})
            else:
                wh_word = self.get_wh_word_in_tmp(structure.tmp)
                _subject = self.lower_when_QG(token_doc, structure.subject)
                question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + " " + structure.object.__str__() + "?"
                qas.append({"question": self.delete_continuous_space(question), "answer": structure.tmp.__str__()})

        # 4. LOC question
        if not structure.loc.isEmpty():
            _answer = self.get_entity(structure.doc, token_doc, structure.loc)
            if _answer is not None:
                wh_word = "Where"
                _subject = self.lower_when_QG(token_doc, structure.subject)
                question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + " " + structure.object.__str__() + "?"
                qas.append({"question": self.delete_continuous_space(question), "answer": _answer})

        # 5. PRP(purpose) question
        # if not structure.prp.isEmpty():
        #     wh_word = "For what purpose"
        #     _subject = self.lower_when_QG(token_doc, structure.subject)
        #     question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + verb_lemma.__str__() + " " + structure.object.__str__() + "?"
        #     qas.append({"question": self.delete_continuous_space(question), "answer": structure.prp.__str__()})

        # 6. PRD(secondary predication) question
        if structure.prd is not None and not structure.prd.object.isEmpty():
            _answer = self.get_entity(structure.loc, token_doc, structure.prd.object)
            if _answer is not None:
                secondary_verb = structure.prd.verb.lemma()
                if len(structure.prd.mods) > 0:
                    mod = random.choice(structure.prd.mods)
                    _mod = self.lower_when_QG(token_doc, mod)
                else:
                    _mod = ""
                wh_word = self.get_wh_word(structure.doc, token_doc, structure.prd.object)
                _subject = self.lower_when_QG(token_doc, structure.subject if structure.prd.subject.isEmpty() else structure.prd.subject)

                # if object starts with PREP
                if structure.prd.object.getStart().upos == "ADP":
                    prep = structure.prd.object.getStart().text.capitalize()
                    wh_word = wh_word.lower()
                    # convert 'who' to 'whom'
                    if wh_word == "who":
                        wh_word = "whom"
                    question = prep + " " + wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + secondary_verb.__str__() + " " + _mod + "?"
                else:
                    question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + secondary_verb.__str__() + " " + _mod + "?"
                qas.append({"question": self.delete_continuous_space(question), "answer": _answer})

        # 7. SCONJ question
        if structure.sconj is not None:
            # incomplete structure
            if structure.sconj.subject.isEmpty():
                _answer = self.get_entity(structure.doc, token_doc, structure.sconj.object)
                if _answer is not None:
                    sconj_verb = structure.sconj.verb.lemma()
                    if len(structure.sconj.mods) > 0:
                        mod = random.choice(structure.sconj.mods)
                        _mod = self.lower_when_QG(token_doc, mod)
                    else:
                        _mod = ""
                    wh_word = self.get_wh_word(structure.doc, token_doc, structure.sconj.object)
                    _subject = self.lower_when_QG(token_doc, structure.subject)

                    # if object starts with PREP
                    if structure.sconj.object.getStart().upos == "ADP":
                        prep = structure.sconj.object.getStart().text.capitalize()
                        wh_word = wh_word.lower()
                        # convert 'who' to 'whom'
                        if wh_word == "who":
                            wh_word = "whom"
                        question = prep + " " + wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + sconj_verb.__str__() + " " + _mod + "?"
                    else:
                        question = wh_word + " " + auxiliary.__str__() + " " + _subject + " " + neg.__str__() + " " + sconj_verb.__str__() + " " + _mod + "?"
                    qas.append({"question": self.delete_continuous_space(question), "answer": _answer})

            # complete structure
            else:
                qas.extend(self.generate_question(token_doc, structure.sconj))

        return qas

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

        result = self.generate_question(token_doc, structure)
        print(result)


generator = QuestionGenerator()
generator.create("Super Bowl 50 was played to see who would be the National Football League ( NFL ) champion.")

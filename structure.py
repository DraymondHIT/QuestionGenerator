class PhraseStructure:
    def __init__(self):
        self.phrase = []

    def left_add(self, token):
        self.phrase.insert(0, token)

    def right_add(self, token):
        self.phrase.append(token)

    def isEmpty(self):
        return len(self.phrase) == 0

    def getValue(self):
        return self.phrase

    def getStart(self):
        return self.phrase[0]

    def lemma(self):
        assert len(self.phrase) == 1
        return self.phrase[0].lemma

    def xpos(self):
        assert len(self.phrase) == 1
        return self.phrase[0].xpos

    def upos(self):
        assert len(self.phrase) == 1
        return self.phrase[0].upos

    def __str__(self):
        temp = [token.text for token in self.phrase]
        return " ".join(temp)


class SentenceStructure:
    def __init__(self, doc, filtered):
        self.doc = doc
        self.subject = PhraseStructure()
        self.object = PhraseStructure()
        self.complement = PhraseStructure()
        self.verb = PhraseStructure()
        self.neg = PhraseStructure()
        self.adv = PhraseStructure()
        self.mod = PhraseStructure()
        self.loc = PhraseStructure()
        self.tmp = PhraseStructure()
        self.prp = PhraseStructure()
        self.prd = None
        self.sconj = None
        self.analysize(doc, filtered)
        self.mods = self.get_mods()

    def analysize(self, doc, filtered):
        import re
        text1 = re.compile(r"B-ARG[0-9]$")
        text2 = re.compile(r"I-ARG[0-9]$")

        if len(filtered) == 1:
            main_structure = filtered[0]
        else:
            # find main structure(main verb)
            main_structure = None
            for result in filtered:
                if list(doc)[result["tags"].index("B-V")].head == 0:
                    main_structure = result

            if main_structure is None:
                min_O = 100
                for result in filtered:
                    if result["tags"].count("O") <= min_O:
                        min_O = result["tags"].count("O")
                        main_structure = result

            # find possible secondary predication
            if main_structure["tags"].count("B-ARGM-PRD") > 0:
                secondary_verb = list(doc)[main_structure["tags"].index("B-ARGM-PRD")].text
                for result in filtered:
                    if result["verb"] == secondary_verb:
                        self.prd = SentenceStructure(doc, [result])
                filtered = filter(lambda x: x["verb"] != secondary_verb, filtered)

            # find possible sconj structure
            verb_follow_index = main_structure["tags"].index("B-V")+1
            verb_follow = list(doc)[verb_follow_index]
            verb_follow_tag = main_structure["tags"][verb_follow_index]
            if verb_follow.upos == "SCONJ":
                sconj_verb = list(doc)[verb_follow.head-1].text
                for result in filtered:
                    if result["verb"] == sconj_verb:
                        self.sconj = SentenceStructure(doc, [result])
                        break
                if self.sconj is None:
                    for result in filtered:
                        if result["tags"].index("B-V") > verb_follow_index and main_structure["tags"][result["tags"].index("B-V")] == "I"+verb_follow_tag[1:]:
                            self.sconj = SentenceStructure(doc, [result])
                            break

        _verb = list(doc)[main_structure["tags"].index("B-V")]
        for index, token in enumerate(doc):
            temp = main_structure["tags"][index]
            if text1.match(temp):
                if self.subject.isEmpty() and index < _verb.id - 1:
                    self.subject.right_add(token)
                if index > _verb.id - 1:
                    if self.object.isEmpty():
                        self.object.right_add(token)
                    elif self.complement.isEmpty():
                        self.complement.right_add(token)
            elif text2.match(temp):
                assert not self.subject.isEmpty() or not self.object.isEmpty() or not self.complement.isEmpty()
                if self.object.isEmpty():
                    self.subject.right_add(token)
                elif self.complement.isEmpty():
                    self.object.right_add(token)
                else:
                    self.complement.right_add(token)
            elif temp.__contains__("B-V"):
                self.verb.right_add(token)
            elif temp.__contains__("NEG"):
                self.neg.right_add(token)
            elif temp.__contains__("ADV"):
                self.adv.right_add(token)
            elif temp.__contains__("MOD"):
                self.mod.right_add(token)
            elif temp.__contains__("LOC"):
                self.loc.right_add(token)
            elif temp.__contains__("TMP"):
                self.tmp.right_add(token)
            elif temp.__contains__("PRP"):
                self.prp.right_add(token)
            elif temp == "O":
                if token.upos == "AUX" and list(doc)[token.head-1].upos == "VERB":
                    self.mod.right_add(token)

    def get_mods(self):
        mods = []
        if not self.complement.isEmpty():
            mods.append(self.complement)
        if not self.adv.isEmpty():
            mods.append(self.adv)
        if not self.loc.isEmpty():
            mods.append(self.loc)
        if not self.tmp.isEmpty():
            mods.append(self.tmp)
        if not self.prp.isEmpty():
            mods.append(self.prp)
        return mods

    def __str__(self):
        return f"""subject: {self.subject}
verb:    {self.verb}
object:  {self.object}
comple:  {self.complement}
adv:     {self.adv}
mod:     {self.mod}
loc:     {self.loc}
tmp:     {self.tmp}"""


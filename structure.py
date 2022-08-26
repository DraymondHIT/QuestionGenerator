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
        self.analysize(doc, filtered)
        self.mods = self.get_mods()

    def analysize(self, doc, filtered):
        import re
        text1 = re.compile(r"B-ARG[0-9]$")
        text2 = re.compile(r"I-ARG[0-9]$")
        filtered = filtered[-1]
        for index, token in enumerate(doc):
            temp = filtered["tags"][index]
            if text1.match(temp):
                if self.subject.isEmpty():
                    self.subject.right_add(token)
                elif self.object.isEmpty():
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


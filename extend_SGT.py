from nltk.corpus import wordnet as wn

SGT = [s.replace("\n", "") for s in open("SGT.txt", "r").readlines()]
extended = list()
for s in SGT:
    extended.append(s)
    extended.extend([w.name().split(".")[0] for w in wn.synsets(s)])
with open("extended_SGT.txt", "w") as writer:
    for s in set(extended):
        writer.write(s.replace("_", " ") + "\n")

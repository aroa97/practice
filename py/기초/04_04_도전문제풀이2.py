# 도전문제2
s = input("염기 서열을 입력해주세요: ")
alphabet = {}

for i in s:
    if not i in alphabet:
        alphabet[i] = 1
    else:
        alphabet[i] += 1

print("a의 개수: {}".format(alphabet["a"]))
print("t의 개수: {}".format(alphabet["t"]))
print("g의 개수: {}".format(alphabet["g"]))
print("c의 개수: {}".format(alphabet["c"]))


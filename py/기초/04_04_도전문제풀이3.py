# 도전문제3
s = input("염기 서열을 입력해주세요: ")
codon = {}

for i in range(len(s) // 3):
    if not s[i * 3 : i * 3 + 3] in codon:
        codon[s[i * 3 : i * 3 + 3]] = 1
    else:
        codon[s[i * 3 : i * 3 + 3]] += 1

print(codon)
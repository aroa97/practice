# 도전문제4

list_value = [1,2,[3,4],5,[6,7],[8,9]]
res = []

for i in range(len(list_value)):
    if type(list_value[i]) is list:
        res.extend(list_value[i])
    else :
        res.append(list_value[i])

print(f"{list_value}를 평탄화하면")
print(f"{res}입니다.")
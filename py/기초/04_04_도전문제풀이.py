# 도전문제1
list_value = [1,2,3,4,1,2,3,1,4,1,2,3]
dict_value = {}
for i in list_value:
    if not i in dict_value:
        dict_value[i] = 1
    else:
        dict_value[i] += 1

print(f"{list_value}에서")
print(f"사용된 숫자의 종류는 {len(dict_value)}개 입니다.")
print(f"참고: {dict_value}")
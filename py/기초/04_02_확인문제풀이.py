# 확인문제1
# dict_a = {}
# dict_a["name"] = "구름"
# del dict_a["name"]
# print(dict_a)

# 확인문제2
numbers = [1,2,6,8,4,3,2,1,9,5,4,9,7,2,1,3,5,4,8,9,7,2,3]
counter = {}

for number in numbers: 
    if number in counter:
        counter[number] += 1
    else:
        counter[number] = 1

print(counter)

# import collections
# numbers = [1,2,6,8,4,3,2,1,9,5,4,9,7,2,1,3,5,4,8,9,7,2,3]

# print(collections.Counter(numbers))
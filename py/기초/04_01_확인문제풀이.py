# 확인문제1
# list_a = [0,1,2,3,4,5,6,7]
# list_a.extend(list_a) -> [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]
# list_a.append(10) -> [0,1,2,3,4,5,6,7,10]
# list_a.insert(3, 0) -> [0,1,2,0,3,4,5,6,7]
# list_a.remove(3) -> [0,1,2,4,5,6,7]
# list_a.pop(3) -> [0,1,2,4,5,6,7]
# list_a.clear() -> []


# 확인문제2
numbers = [273,103,5,32,65,9,72,800,99]

for number in numbers:
    if number >= 100:
        print("- 100 이상의 수:", number)
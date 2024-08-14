# 확인문제3
numbers = [273,103,5,32,65,9,72,800,99]

for i in numbers:
    if i % 2 == 0: 
        print(i, "는 짝수입니다.")
    else: 
        print(i, "는 홀수입니다.")

print()

for i in numbers:
    print(i, "는", len(str(i)), "자릿수입니다.")
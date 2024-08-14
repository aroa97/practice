score = int(input("점수는 0 ~ 100까지만 입력하세요: "))

if score >= 90 and score <= 100: print("A")
elif score >= 80 and score <= 89: print("B")
elif score >= 70 and score <= 79: print("C")
elif score >= 0 and score <= 69: print("F")
else:
    print("0~100의 점수만 입력하세요")
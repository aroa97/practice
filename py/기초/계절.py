# import datetime
# mon = datetime.datetime.now().month

mon = int(input("현재 몇 월: "))

if mon >= 3 and mon <= 5:
    print("봄")
elif mon >= 6 and mon <= 8:
    print("여름")
elif mon >= 9 and mon <= 11:
    print("가을")
else:
    print("겨울")
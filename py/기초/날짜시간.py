import datetime

now = datetime.datetime.now()
print(f"{now.year}년 {now.month}월 {now.day}일")
print("{}시 {}분 {}초".format(now.hour, now.minute, now.second))

if now.hour < 12:
    print("오전")
else:
    print("오후")
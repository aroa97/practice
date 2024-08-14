import datetime

talk = input("입력: ").strip()

if "안녕" in talk : print("> 안녕하세요.")
if "몇 시" in talk or "몇시" in talk: 
    print(f"> 지금은 {datetime.datetime.now().hour}시입니다.")
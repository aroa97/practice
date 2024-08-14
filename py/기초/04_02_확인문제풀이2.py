# 확인문제3
character = {
    "name": "기사",
    "level": 12,
    "items": {
        "sword": "불꽃의 검",
        "armor": "풀플레이트"
    },
    "skill": ["베기", "세게 베기", "아주 세게 베기"]
}

for key in character:
    if type(character[key]) is list:
        for i in character[key]:
            print(key, ":", i)
    elif type(character[key]) is dict:
        for key2, value in character[key].items():
            print(key2, ":", value)
    else:
        print(key, ":", character[key])
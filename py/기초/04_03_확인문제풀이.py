# 확인문제1
# range(4,6) -> [4,5]
# range(7, 0, -1) -> [7,6,5,4,3,2,1]
# range(3,10,3) -> [3,6,9]

# 확인문제2 
key_list = ["name", "hp", "mp", "level"]
value_list = ["기사", 200, 30, 5]
charater = {}

for i in range(len(key_list)):
    charater[key_list[i]] = value_list[i]

print(charater)
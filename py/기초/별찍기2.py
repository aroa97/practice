for i in range(5, 0,-1):
    print('*' * i)

for i in range(5):
    for j in range(5 - i):
        print('*', end="")
    print()
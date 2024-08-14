for i in range(6):
    print(' ' * (5 - i), end='')
    print('*' * (2 * i + 1))
    
for i in range(2):
    print(' ' * 4, end='')
    print('*' * 3)


for i in range(6):
    for j in range(5 - i):
        print(' ', end='')
    for j in range(2 * i + 1):
        print('*', end='')
    print()
    
for i in range(2):
    for j in range(4):
        print(' ', end='')
    for j in range(3):
        print('*', end='')
    print()
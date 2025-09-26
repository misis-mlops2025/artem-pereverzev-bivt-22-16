
def main(*args):
    return sum(args)

if __name__ == '__main__':
    a, b, c = -4, 9, 10
    res = main(a, b, c)
    print(res)

    res = main(a, b)
    print(res)

    res = main(1, 2, 4, 5, -1)
    print(res)

    res = main(1, 2, 4, 5, -1, 10.44, -100.11)
    print(res)
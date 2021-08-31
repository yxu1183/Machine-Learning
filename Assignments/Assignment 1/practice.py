def factorial(n):
    result = 1
    if n == 1:
        return result
    else:
        result = result * factorial(n)
    return result


print(factorial(3))

def factorial(n):
    result = 1
    if n == 1 or n == 0:
        return result
    else:
        result = n * factorial(n-1)
        return result


print(factorial(4))

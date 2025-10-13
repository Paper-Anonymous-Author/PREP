import numpy as np

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def month_plus(start_month, months):
    month_now = start_month.copy()
    month_now[0] += int(months / 12)
    month_now[1] += months % 12
    if month_now[1] >= 12:
        month_now[1] -= 12
        month_now[0] += 1
    return [int(month_now[0]), int(month_now[1])]

if __name__ == '__main__':
    print(month_plus([2010, 1], 11))
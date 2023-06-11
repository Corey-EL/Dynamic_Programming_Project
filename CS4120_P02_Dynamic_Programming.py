#   Corey Lang
#   Project 2: Practice with Dynamic Programming
#   Task: 
#       write three ways to solve the fibonacci problem and matrix chain problem.
#       (1) straight recursive approach
#       (2) memoized top-down approach
#       (3) bottom-up approach
#       (4) then analysis the time taken for all 6 methods and compare
#-------------------------------------------------------------------------------
import time
import sys
import pandas as pd


# Table is created to store the observable data with headers
df = pd.DataFrame(columns=['Algorithm_Used', 'n', 'Value', 'Timer (ns)'])

# fibonacci number (1) recursive (2) memorized top-down (3) bottom-up
def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
def fibonacci_top_down(n, memo={0: 0, 1: 1}):
    if n in memo:
        return memo[n]
    elif n <= 1:
        return n
    else:
        memo[n] = fibonacci_top_down(n-1, memo) + fibonacci_top_down(n-2, memo)
        return memo[n]


def fibonacci_bottom_up(n):
    if n <= 1:
        return n
    else:
        fib = [0, 1]
        for i in range(2, n+1):
            fib.append(fib[i-1] + fib[i-2])
        return fib[n]
    

# Table is created to store the observable data with headers
df_2 = pd.DataFrame(columns=['Algorithm_Used', 'Matrix Dimensions', 'Minimum number of multiplications', 'Optimal parenthesization', 'Timer (ns)'])

# matrix chain order (1) recursive (2) memorized top-down (3) bottom-up
def print_optimal_parens(s, i, j):
    if i == j:
        return f"A{i}"
    else:
        return f"({print_optimal_parens(s, i, s[i][j])}{print_optimal_parens(s, s[i][j] + 1, j)})"

# (1)
def MatrixChainOrder_recursive(p, i, j, s):
    if i == j:
        return 0

    _min = sys.maxsize
    split_point = None

    for k in range(i, j):
        count = (MatrixChainOrder_recursive(p, i, k, s)
                 + MatrixChainOrder_recursive(p, k + 1, j, s)
                 + p[i - 1] * p[k] * p[j])

        if count < _min:
            _min = count
            split_point = k

    if split_point:
        s[i][j] = split_point

    return _min

# (2)
def MatrixChainOrder_memo(p,n):
    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        m[i][i] = 0

    return Look_Up(p, m, s, 1, n), s

def Look_Up(p, m, s, i, j):
    if m[i][j] != 0:
        return m[i][j]

    if i == j:
        m[i][j] = 0
    else:
        _min = sys.maxsize
        for k in range(i, j):
            count_left = Look_Up(p, m, s, i, k)
            count_right = Look_Up(p, m, s, k + 1, j)
            count = count_left + count_right + p[i - 1] * p[k] * p[j]

            if count < _min:
                _min = count
                m[i][j] = _min
                s[i][j] = k

    return m[i][j]

# (3)
def MatrixChainOrder_bottom(p,n):
    m = [[0 for _ in range(n + 1)] for _ in range(n + 1)]
    s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        m[i][i] = 0

    for l in range(2, n + 1):
        for i in range(1, n - l + 2):
            j = i + l - 1
            m[i][j] = float('inf')
            for k in range(i, j):
                q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m[1][n], s

# Fibonaci calls and times for [2, 7, 30, 10, 0, 8, 40]
start_time_r1 = time.perf_counter_ns()
fib_comp_r1 = fibonacci_recursive(2)
end_time_r1 = time.perf_counter_ns()

start_time_r2 = time.perf_counter_ns()
fib_comp_r2 = fibonacci_recursive(7)
end_time_r2 = time.perf_counter_ns()

start_time_r3 = time.perf_counter_ns()
fib_comp_r3 = fibonacci_recursive(30)
end_time_r3 = time.perf_counter_ns()

start_time_r4 = time.perf_counter_ns()
fib_comp_r4 = fibonacci_recursive(10)
end_time_r4 = time.perf_counter_ns()

start_time_r5 = time.perf_counter_ns()
fib_comp_r5 = fibonacci_recursive(0)
end_time_r5 = time.perf_counter_ns()

start_time_r6 = time.perf_counter_ns()
fib_comp_r6 = fibonacci_recursive(8)
end_time_r6 = time.perf_counter_ns()

start_time_r7 = time.perf_counter_ns()
fib_comp_r7 = fibonacci_recursive(40)
end_time_r7 = time.perf_counter_ns()

fib_r = [{'Algorithm_Used': 'fibonacci_recursive', 
         'n': '2',
         'Value': fib_comp_r1, 
         'Timer (ns)': end_time_r1 - start_time_r1},

        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '7',
         'Value': fib_comp_r2, 
         'Timer (ns)': end_time_r2 - start_time_r2},

        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '30',
         'Value': fib_comp_r3, 
         'Timer (ns)': end_time_r3 - start_time_r3},

        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '10',
         'Value': fib_comp_r4, 
         'Timer (ns)': end_time_r4 - start_time_r4},

        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '0',
         'Value': fib_comp_r5, 
         'Timer (ns)': end_time_r5 - start_time_r5},

        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '8',
         'Value': fib_comp_r6, 
         'Timer (ns)': end_time_r6 - start_time_r6},
        
        {'Algorithm_Used': 'fibonacci_recursive', 
         'n': '40',
         'Value': fib_comp_r7, 
         'Timer (ns)': end_time_r7 - start_time_r7}]

df_fib_r = pd.concat([pd.DataFrame([d]) for d in fib_r], ignore_index=True)
df = pd.concat([df, df_fib_r], ignore_index=True)

# calls and times for [2, 7, 30, 10, 0, 8, 40]
start_time_t1 = time.perf_counter_ns()
fib_comp_t1 = fibonacci_top_down(2)
end_time_t1 = time.perf_counter_ns()

start_time_t2 = time.perf_counter_ns()
fib_comp_t2 = fibonacci_top_down(7)
end_time_t2 = time.perf_counter_ns()

start_time_t3 = time.perf_counter_ns()
fib_comp_t3 = fibonacci_top_down(30)
end_time_t3 = time.perf_counter_ns()

start_time_t4 = time.perf_counter_ns()
fib_comp_t4 = fibonacci_top_down(10)
end_time_t4 = time.perf_counter_ns()

start_time_t5 = time.perf_counter_ns()
fib_comp_t5 = fibonacci_top_down(0)
end_time_t5 = time.perf_counter_ns()

start_time_t6 = time.perf_counter_ns()
fib_comp_t6 = fibonacci_top_down(8)
end_time_t6 = time.perf_counter_ns()

start_time_t7 = time.perf_counter_ns()
fib_comp_t7 = fibonacci_top_down(40)
end_time_t7 = time.perf_counter_ns()

fib_t = [{'Algorithm_Used': 'fibonacci_top_down', 
         'n': '2',
         'Value': fib_comp_t1, 
         'Timer (ns)': end_time_t1 - start_time_t1},

        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '7',
         'Value': fib_comp_t2, 
         'Timer (ns)': end_time_t2 - start_time_t2},

        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '30',
         'Value': fib_comp_t3, 
         'Timer (ns)': end_time_t3 - start_time_t3},

        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '10',
         'Value': fib_comp_t4, 
         'Timer (ns)': end_time_t4 - start_time_t4},

        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '0',
         'Value': fib_comp_t5, 
         'Timer (ns)': end_time_t5 - start_time_t5},

        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '8',
         'Value': fib_comp_t6, 
         'Timer (ns)': end_time_t6 - start_time_t6},
        
        {'Algorithm_Used': 'fibonacci_top_down', 
         'n': '40',
         'Value': fib_comp_t7, 
         'Timer (ns)': end_time_t7 - start_time_t7}]

df_fib_t = pd.concat([pd.DataFrame([d]) for d in fib_t], ignore_index=True)
df = pd.concat([df, df_fib_t], ignore_index=True)

# calls and times for fibonacci_bottom_up [2, 7, 30, 10, 0, 8, 40]
start_time_b1 = time.perf_counter_ns()
fib_comp_b1 = fibonacci_bottom_up(2)
end_time_b1 = time.perf_counter_ns()

start_time_b2 = time.perf_counter_ns()
fib_comp_b2 = fibonacci_bottom_up(7)
end_time_b2 = time.perf_counter_ns()

start_time_b3 = time.perf_counter_ns()
fib_comp_b3 = fibonacci_bottom_up(30)
end_time_b3 = time.perf_counter_ns()

start_time_b4 = time.perf_counter_ns()
fib_comp_b4 = fibonacci_bottom_up(10)
end_time_b4 = time.perf_counter_ns()

start_time_b5 = time.perf_counter_ns()
fib_comp_b5 = fibonacci_bottom_up(0)
end_time_b5 = time.perf_counter_ns()

start_time_b6 = time.perf_counter_ns()
fib_comp_b6 = fibonacci_bottom_up(8)
end_time_b6 = time.perf_counter_ns()

start_time_b7 = time.perf_counter_ns()
fib_comp_b7 = fibonacci_bottom_up(40)
end_time_b7 = time.perf_counter_ns()

fib_b = [{'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '2',
         'Value': fib_comp_b1, 
         'Timer (ns)': end_time_b1 - start_time_b1},

        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '7',
         'Value': fib_comp_b2, 
         'Timer (ns)': end_time_b2 - start_time_b2},

        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '30',
         'Value': fib_comp_b3, 
         'Timer (ns)': end_time_b3 - start_time_b3},

        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '10',
         'Value': fib_comp_b4, 
         'Timer (ns)': end_time_b4 - start_time_b4},

        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '0',
         'Value': fib_comp_b5, 
         'Timer (ns)': end_time_b5 - start_time_b5},

        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '8',
         'Value': fib_comp_b6, 
         'Timer (ns)': end_time_b6 - start_time_b6},
        
        {'Algorithm_Used': 'fibonacci_bottom_up', 
         'n': '40',
         'Value': fib_comp_b7, 
         'Timer (ns)': end_time_b7 - start_time_b7}]

df_fib_b = pd.concat([pd.DataFrame([d]) for d in fib_b], ignore_index=True)
df = pd.concat([df, df_fib_b], ignore_index=True)
print(df, end='\n')


# calls and times for matrix dimensions of [30, 35, 15, 5, 10, 20, 25]
# inputs
p = [30, 35, 15, 5, 10, 20, 25]
n = len(p) - 1
s = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

start_time_r = time.perf_counter_ns()
rm = MatrixChainOrder_recursive(p, 1, n, s)
rs = print_optimal_parens(s, 1, n)
end_time_r = time.perf_counter_ns()

start_time_m = time.perf_counter_ns()
mm, s2 = MatrixChainOrder_memo(p,n)
ms = print_optimal_parens(s2, 1, n)
end_time_m = time.perf_counter_ns()

start_time_b = time.perf_counter_ns()
bm, s3 = MatrixChainOrder_bottom(p, n)
bs = print_optimal_parens(s3, 1, n)
end_time_b = time.perf_counter_ns()

matrix = [{'Algorithm_Used': 'MatrixChainOrder_recursive', 
         'Matrix Dimensions': '[30, 35, 15, 5, 10, 20, 25]',
         'Minimum number of multiplications' : rm,
         'Optimal parenthesization': rs, 
         'Timer (ns)': end_time_r - start_time_r},

        {'Algorithm_Used': 'MatrixChainOrder_memo', 
         'Matrix Dimensions': '[30, 35, 15, 5, 10, 20, 25]',
         'Minimum number of multiplications' : mm,
         'Optimal parenthesization': ms, 
         'Timer (ns)': end_time_m - start_time_m},

        {'Algorithm_Used': 'MatrixChainOrder_bottom', 
         'Matrix Dimensions': '[30, 35, 15, 5, 10, 20, 25]',
         'Minimum number of multiplications' : bm,
         'Optimal parenthesization': bs, 
         'Timer (ns)': end_time_b - start_time_b}]

df_matrix = pd.concat([pd.DataFrame([d]) for d in matrix], ignore_index=True)
df_2 = pd.concat([df_2, df_matrix], ignore_index=True)


# calls and times for matrix dimensions of [5, 2, 3, 10, 5, 4, 20]
# inputs
p_2 = [2, 3, 10, 5, 4, 20]
n_2 = len(p_2) - 1
s_2 = [[0 for _ in range(n + 1)] for _ in range(n + 1)]


start_time_r2 = time.perf_counter_ns()
rm2 = MatrixChainOrder_recursive(p_2, 1, n_2, s_2)
rs2 = print_optimal_parens(s_2, 1, n_2)
end_time_r2 = time.perf_counter_ns()

start_time_m2 = time.perf_counter_ns()
mm2, s2_2 = MatrixChainOrder_memo(p_2,n_2)
ms2 = print_optimal_parens(s2_2, 1, n_2)
end_time_m2 = time.perf_counter_ns()

start_time_b2 = time.perf_counter_ns()
bm2, s3_2 = MatrixChainOrder_bottom(p_2, n_2)
bs2 = print_optimal_parens(s3_2, 1, n_2)
end_time_b2 = time.perf_counter_ns()

matrix_2 = [{'Algorithm_Used': 'MatrixChainOrder_recursive', 
         'Matrix Dimensions': '[2, 3, 10, 5, 4, 20]',
         'Minimum number of multiplications' : rm2,
         'Optimal parenthesization': rs2, 
         'Timer (ns)': end_time_r2 - start_time_r2},

        {'Algorithm_Used': 'MatrixChainOrder_memo', 
         'Matrix Dimensions': '[2, 3, 10, 5, 4, 20]',
         'Minimum number of multiplications' : mm2,
         'Optimal parenthesization': ms2, 
         'Timer (ns)': end_time_m2 - start_time_m2},

        {'Algorithm_Used': 'MatrixChainOrder_bottom', 
         'Matrix Dimensions': '[2, 3, 10, 5, 4, 20]',
         'Minimum number of multiplications' : bm2,
         'Optimal parenthesization': bs2, 
         'Timer (ns)': end_time_b2 - start_time_b2}]

df_matrix_2 = pd.concat([pd.DataFrame([d]) for d in matrix_2], ignore_index=True)
df_2 = pd.concat([df_2, df_matrix_2], ignore_index=True)
print(df_2)
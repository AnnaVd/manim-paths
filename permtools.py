"""
Python module.
Some utilities to extract data from permutations or other lists.
"""

def inverse(sigma):
    # Returns the inverse of the permutation sigma, given as a list of integers.

    tau = [0]*len(sigma)
    for i in range(len(sigma)):
        tau[sigma[i]-1] = i+1

    return tau

def des(sigma):
    # Returns the descents set of a list of integers.

    des = []
    for i in range(len(sigma)-1):
        if sigma[i] > sigma[i+1]:
            des += [i+1]

    return des

def asc(sigma):
    # Returns the ascents set of a list of integers.

    asc = []
    for i in range(len(sigma)-1):
        if sigma[i] < sigma[i+1]:
            asc += [i+1]

    return asc

def ides(sigma):
    return des(inverse(sigma))

def runs(sigma, increasing = True):
    # Returns the subdivision in maximal increasing (or decreasing) sequences of a list of integers.

    if increasing:
        jumps = des(sigma)
    else:
        jumps = asc(sigma)

    a = len(jumps)
    
    if a > 0:
        runs = [sigma[:jumps[0]]]
        for j in range(a-1):
            runs += [sigma[jumps[j]:jumps[j+1]]]
        runs += [sigma[jumps[a-1]:len(sigma)]]
    else:
        runs = [sigma]
        
    return runs

def maj(sigma):
    # Returns the sum of the descents of a list of integers.
    # If sigma is a permutation, this is the major index.

    return sum(des(sigma))

def set_to_composition(s, n):
    # Computes the composition of n corresponding to the sublist s of [n-1]

    s = [0] + s + [n]

    return [s[i+1]-s[i] for i in range(len(s)-1)]

def get_nth_index(list, item, n):
    # Returns the index of the n-th occurrence of item in list.

    from itertools import count
    c = count()

    return next(i for i, j in enumerate(list) if j == item and next(c) == n-1)
    
def prod(list):
    # Returns the product of the elements of the list.

    prod = 1
    for i in list:
        prod *= i

    return prod

def shuffle_two(a,b):
    # Given two lists, returns the shuffle of the two,
    # i.e. all the lists obtained by shuffling a and b.

    if len(a) == 0:
        yield b
    elif len(b) == 0:
        yield a
    else:
        for p in shuffle_two(a[1:],b):
            yield [a[0]] + p
        for p in shuffle_two(a,b[1:]):
            yield [b[0]] + p

def shuffle_list(l):
    # Given a list of lists, returns all the possible shuffles of the elements.

    if len(l) == 0:
        return []
    elif len(l) == 1:
        return [l[0]]
    else:
        return [y for x in shuffle_two(l[0], l[1]) for y in shuffle_list([x] + shuffle_list(l[2:]))]

def shuffle_munu(mu = [], nu = []):
    # Given two compositions mu, nu, it returns the list of all shuffles
    # of the lists [1, 2, ..., mu[1]], [mu[1]+1, mu[1]+2, ..., mu[1]+mu[2]], ...
    # of the lists [sum([mu])+nu[1], sum([mu])+nu[1]-1, ..., sum([mu])], ... 

    sh_mu = [[]]
    for i in range(0, len(mu)):
        auxsh = []
        for ww in sh_mu:
            auxsh = auxsh + list(shuffle_two(ww, [j + sum(mu[:i]) for j in range(1, mu[i]+1)]))
        sh_mu = auxsh

    sh_nu = [[]]
    for i in range(0, len(nu)):
        auxsh = []
        for ww in sh_nu:
            auxsh = auxsh + list(shuffle_two(ww, [sum(mu) + sum(nu[:i+1]) - j for j in range(0, nu[i])]))
        sh_nu = auxsh

    return [z for x in sh_mu for y in sh_nu for z in shuffle_two(x,y)]
    
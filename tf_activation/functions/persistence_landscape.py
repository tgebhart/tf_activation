import numpy as np
import sys


def diag_to_numpy(diag):
    ret = np.empty(shape=(len(diag),2))
    i = 0
    for p in diag:
        ret[i,0] = p.birth
        ret[i,1] = p.death
        i += 1
    return ret


def persistence_landscape(A, sorted=False, dionysus_diagram=False, k=float('Inf')):
    '''Compute persistence landscape of diagram A. Assume A 2D numpy array unless
    dionysus diagram.
    '''
    if dionysus_diagram:
        A = diag_to_numpy(A)
    if not sorted:
        A = A[A[:,0].argsort()]
        A = A[-A[:,1].argsort()]
    L = []
    while A.shape[0] > 0 and len(L) < k:
        l_k = []
        b,d = A[0,:]
        A = np.delete(A,0,0)
        l_k += [[-float('Inf'),0], [b,0], [(b+d)/2,(d-b)/2]]
        while l_k[len(l_k)-1] != [0,float('Inf')]:
            if d >= A[:,1].all():
                l_k.append([d,0])
                l_k.append([0,float('Inf')])
            else:
                print('in else')
                if A.shape[0] > 0:
                    bp,dp = A[0,:]
                else:
                    break
                A = np.delete(A,0,0)
                if bp > d:
                    l_k.append([d,0])
                else:
                    l_k.append([(bp+d)/2,(d-bp)/2])
                    isidx = np.searchsorted(A, [bp,d])
                    A = np.insert(A, isidx, [bp,d])
                l_k.append([(bp+dp)/2,(dp-bp)/2])
                b = bp
                d = dp
        if len(l_k) > 0:
            L.append(np.array(l_k))
    return L


def average_landscape(Ls):
    '''Computes average landscape between each landscape in `Ls`'''
    # find least max k:
    lmk = sys.maxsize
    for l in Ls:
        s = len(l)
        if s < lmk:
            lmk = s
    avg_Ls = []
    for k in range(lmk):
        avg_l_k = []
        for L in Ls:
            for i in range(len(L[k])):
                try:
                    avg_l_k[i] += L[k][i]
                except IndexError:
                    avg_l_k.append(L[k][i])
        for j in range(len(avg_l_k)):
            avg_l_k[j] = avg_l_k[j] / float(len(L))
        avg_Ls.append(np.array(avg_l_k))
    return avg_Ls

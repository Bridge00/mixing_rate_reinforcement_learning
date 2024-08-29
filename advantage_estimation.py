import numpy
import torch

#implemntation of advantage estimation fro bai et. al.
def advantage_estimation(trajectory, N, t1, t2, s, a, pi_a_s):

    i = 0 
    tau = t1
    taus = []
    ys = []
    while tau < t2 - N:
        if trajectory[tau][0] == s:
            i += 1

            taus.append(tau)
            ys.append(np.sum([trajectory[t][2] for t in range(tau, tau + N)]))
            tau += 2*N
        else:
            tau += 1

    if i > 0:
        
        V_est = np.mean(ys)

        cond_sum = 0
        for index, y_i in enumerate(ys):
            if trajectory[taus[index]][1] == a:
                cond_sum += y_i

        Q_est = 1/pi_a_s * 1/len(ys) * cond_sum

        return Q_est - V_est
    else:
        return 0
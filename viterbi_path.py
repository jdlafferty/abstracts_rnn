import numpy as np
# https://stackoverflow.com/a/43824753/684979


def viterbi_path(prior, transmat, obslik, scaled=True, ret_loglik=False):
    print('Viterbi path')
    '''Finds the most-probable (Viterbi) path through the HMM state trellis
    Notation:
        Z[t] := Observation at time t
        Q[t] := Hidden state at time t
    Inputs:
        prior: np.array(num_hid)
            prior[i] := Pr(Q[0] == i)
        transmat: np.ndarray((num_hid,num_hid))
            transmat[i,j] := Pr(Q[t+1] == j | Q[t] == i)
        obslik: np.ndarray((num_hid,num_obs))
            obslik[i,t] := Pr(Z[t] | Q[t] == i)
        scaled: bool
            whether or not to normalize the probability trellis along the way
            doing so prevents underflow by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(num_obs)
            path[t] := Q[t]
    '''
    num_hid = obslik.shape[0] # number of hidden states
    num_obs = obslik.shape[1] # number of observations (not observation *states*)

    # trellis_prob[i,t] := Pr((best sequence of length t-1 goes to state i), Z[1:(t+1)])
    trellis_prob = np.zeros((num_hid,num_obs))
    # trellis_state[i,t] := best predecessor state given that we ended up in state i at t
    trellis_state = np.zeros((num_hid,num_obs), dtype=int) # int because its elements will be used as indicies
    path = np.zeros(num_obs, dtype=int) # int because its elements will be used as indicies

    trellis_prob[:,0] = prior * obslik[:,0] # element-wise mult
    if scaled:
        scale = np.ones(num_obs) # only instantiated if necessary to save memory
        scale[0] = 1.0 / np.sum(trellis_prob[:,0])
        trellis_prob[:,0] *= scale[0]

    trellis_state[:,0] = 0 # arbitrary value since t == 0 has no predecessor
    for t in xrange(1, num_obs):
        for j in xrange(num_hid):
            trans_probs = trellis_prob[:,t-1] + transmat[:,j] # element-wise mult
            print(trans_probs)
            trellis_state[j,t] = trans_probs.argmax()
            trellis_prob[j,t] = trans_probs[trellis_state[j,t]] # max of trans_probs
            trellis_prob[j,t] += obslik[j,t]
        if scaled:
            scale[t] = 1.0 / np.sum(trellis_prob[:,t])
            trellis_prob[:,t] *= scale[t]

    path[-1] = trellis_prob[:,-1].argmax()
    for t in range(num_obs-2, -1, -1):
        path[t] = trellis_state[(path[t+1]), t+1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = trellis_prob[path[-1],-1]
            loglik = np.log(p)
        return path, loglik



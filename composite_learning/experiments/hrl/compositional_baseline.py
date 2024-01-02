import numpy as np
import solver.simple_hrl_solver as solver
import joblib as jl

Ks = [2, 3, 4, 5, 6, 7, 8]
nums_iter = [0, 1000000]
Ts_compositional = [2, 3, 4, 5, 6, 7, 8, 9, 10]
lr_w = 1.
lr_wc = 1.
lr_v = 1.
N = 1000
identical = False
v_norm = 0
lr_w = 1.
lr_wc = 1.
lr_v = 1.
update_frequency = 10


def gram_schmidt(N, K):
    """
    Given the dimension space dimension N, generate K random vectors and its orthogonal spans
    """

    def proj(u, v):
        """
        Return projection of v to u
        """
        return np.dot(v, u) / np.dot(u, u) * u

    V = np.random.normal(loc=0., scale=1.0, size=(K, N))
    U = np.zeros_like(V)

    ## Initialise u1 to v1
    U[0] = V[0]

    ## Gram-schomidt process
    for k in range(1, K):
        projection_terms = [proj(U[i], V[k]) for i in range(k)]
        U[k] = V[k] - np.sum(projection_terms, axis=0)

    return V, U


def control_VS(VT, angle):
    dim = len(VT)
    VT_norm = VT / np.linalg.norm(VT)
    a = np.random.normal(loc=0., scale=1, size=(dim))
    b = np.random.normal(loc=0., scale=1, size=(dim))
    h = (b - a) - np.dot((b - a), VT_norm) * VT_norm
    v = np.cos(angle) * VT_norm + np.sin(angle) * h / np.linalg.norm(h)

    return v


compositional_models = {'sim': {}, 'ode': {}}
for K in Ks:
    _, WT = gram_schmidt(N, K)
    compositional_models['sim'][K] = {i: None for i in Ts_compositional}
    compositional_models['ode'][K] = {i: None for i in Ts_compositional}
    WS = WT.copy()
    for i, w in enumerate(WT):
        w_ortho = np.array(control_VS(w, np.pi / 2) * np.sqrt(N),
                           dtype=np.float64)
        WS[i] = w_ortho
    VT = np.ones(K)
    VT = VT / np.linalg.norm(VT)
    VS = control_VS(VT, np.pi / 4)

    VS_sim = VS.copy()
    VS_ode = VS.copy()

    for T in Ts_compositional:

        WS_sim = WS.copy()
        WS_ode = WS_sim.copy()

        single_task_sim = solver.CurriculumCompositionalTaskSimulator(
            input_dim=N,
            seq_len=T,
            num_task=K,
            identical=identical,
            WT=WT,
            WS=WS_sim,
            VT=VT,
            VS=VS_sim,
            V_norm=v_norm)
        single_task_ode = solver.HRLODESolver(WT=WT,
                                              WS=WS_ode,
                                              VT=VT,
                                              VS=VS_ode,
                                              lr_ws=[lr_w, lr_wc],
                                              lr_v=lr_v,
                                              N=N,
                                              seq_length=T,
                                              V_norm=v_norm)
        single_task_sim.train(num_iter=nums_iter,
                              update_frequency=10,
                              lr={
                                  'lr_w': lr_w,
                                  'lr_wc': lr_wc,
                                  'lr_vc': lr_v
                              })
        single_task_ode.train(
            nums_iter=np.array(nums_iter),
            update_frequency=10,
        )
        compositional_models['sim'][K][T] = single_task_sim
        compositional_models['ode'][K][T] = single_task_ode
        print(f'T:{T}, K:{K} done')

jl.dump(compositional_models, 'compositional_baseline_models.jl')
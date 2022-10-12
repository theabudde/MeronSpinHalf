from MeronAlgorithmOneFixedCharge import MeronAlgorithmOneFixedCharge


def main():
    n = 12  # number of lattice points
    t = 12  # number of half time steps (#even + #odd)
    beta = 1  # beta
    mc_steps = 100  # number of mc steps
    initial_mc_steps = 5000
    w_a = 2 / 4  # np.exp(b/t)  # weight of a plaquettes U = t = 1
    w_b = 2 / 4  # np.sinh(b/t)  # weight of b plaquettes

    algorithm = MeronAlgorithmOneFixedCharge(n, t, w_a, w_b, beta, mc_steps)

    for mc in range(mc_steps):
        algorithm.mc_step()


if __name__ == "__main__":
    main()

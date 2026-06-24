def get_FH_copula_func(XR):
    """XR specifies the RNS structure. Its structure is:
        XR[0][i] is the set of variables (indices) that use RNS Ri
        XR[1][i] is the set of variables (indices) that use the inverted RNS Ri'
        This implements Eq. 21 in the paper
    """
    def FH_C(Px):
        n = len(Px)
        c = 1
        for i in range(n):
            PXR = [Px[j] for j in XR[0][i]]
            PXR_inv = [Px[j] for j in XR[1][i]]
            c *= max(min(PXR) + min(PXR_inv) - 1, 0)
        return c
    return FH_C
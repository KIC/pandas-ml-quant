# revised version of the LPPL without φ
# found on page 11 as equation (13)
import numpy as np


def lppl_fit_wrapper(t, prices, tc, m, w,
                     Cos=np.cos, Sin=np.sin, Log=np.log, Sum=np.sum, Stack=np.stack, LinSolver=np.linalg.lstsq,
                     anti_bubble=False):
    lppl = lppl_wrapper(Cos, Sin, Log, anti_bubble)
    abcc = abcc_solver_wrapper(Sum, Cos, Sin, Stack, LinSolver)

    N = len(t)
    dt = tc - t
    dtPm = dt ** m
    dtln = np.log(dt)
    abcc_ = abcc(prices, dtPm, dtln, w, N)
    a, b, c1, c2 = abcc_[0], abcc_[1], abcc_[2], abcc_[3]

    return lppl(t, tc, m, w, a, b, c1, c2)


def lppl_wrapper(Cos=np.cos, Sin=np.sin, Log=np.log, anti_bubble=False):

    # The distance to the critical time is τ = tc − t for bubbles and τ = t − tc for antibubbles.
    def lppl_bubble(t, tc, m, w, a, b, c1, c2):
        return a + (tc - t) ** m * (b + ((c1 * Cos(w * Log(tc - t))) + (c2 * Sin(w * Log(tc - t)))))

    # The distance to the critical time is τ = tc − t for bubbles and τ = t − tc for antibubbles.
    def lppl_antibubble(t, tc, m, w, a, b, c1, c2):
        return a + (t - tc) ** m * (b + ((c1 * Cos(w * Log(t - tc))) + (c2 * Sin(w * Log(t - tc)))))

    return lppl_antibubble if anti_bubble else lppl_bubble


def abcc_solver_wrapper(Sum=np.sum, Cos=np.cos, Sin=np.sin, Stack=np.stack, LinSolver=np.linalg.lstsq):
    def matrix_equation(x, dtPm, dtln, w, N):
        fi = dtPm
        gi = dtPm * Cos(w * dtln)
        hi = dtPm * Sin(w * dtln)

        fi_pow_2 = Sum(fi ** 2)
        gi_pow_2 = Sum(gi ** 2)
        hi_pow_2 = Sum(hi ** 2)

        figi = Sum(fi * gi)
        fihi = Sum(fi * hi)
        gihi = Sum(gi * hi)

        # note that our price is already a log price so we should not log it one more time
        yi = x  # K.log(x)
        yifi = Sum(yi * fi)
        yigi = Sum(yi * gi)
        yihi = Sum(yi * hi)

        fi = Sum(fi)
        gi = Sum(gi)
        hi = Sum(hi)
        yi = Sum(yi)

        A = Stack(
            [
                Stack([N, fi, gi, hi]),
                Stack([fi, fi_pow_2, figi, fihi]),
                Stack([gi, figi, gi_pow_2, gihi]),
                Stack([hi, fihi, gihi, hi_pow_2])
            ],
            axis=0
        )

        b = Stack([yi, yifi, yigi, yihi])

        # do a classic x = (A'A)⁻¹A' b
        return LinSolver(A, b)

    return matrix_equation


class LPPL(object):
    def __init__(self,
                 Sum=np.sum,
                 Cos=np.cos,
                 Sin=np.sin,
                 Log=np.log,
                 Stack=np.stack,
                 LinSolver=np.linalg.lstsq,
                 anti_bubble=False):
        self.Sum = Sum
        self.Cos = Cos
        self.Sin = Sin
        self.Log = Log
        self.Stack = Stack
        self.LinSolver = LinSolver
        self.anti_bubble = anti_bubble
        self.lppl = None

        # params
        self.tc = 1
        self.m = 0.9
        self.w = 0.6
        self.a = None
        self.b = None
        self.c1 = None
        self.c2 = None

    def fit(self, t, prices):
        lppl = lppl_wrapper(anti_bubble=self.anti_bubble)
        abcc = abcc_solver_wrapper()
        N = len(prices)

        # then we calculate the lppl with the given parameters
        dt = (self.tc - t)
        dtPm = dt ** self.m
        dtln = self.Log(dt)
        abcc = abcc_solver_wrapper(prices, dtPm, dtln, self.w, N)
        a, b, c1, c2 = (abcc[0], abcc[1], abcc[2], abcc[3])
        lppl(t, self.tc, self.m, self.w, a, b, c1, c2)
        pass

    def predict(self, t):
        if self.lppl is None:
            lppl = lppl_wrapper(anti_bubble=self.anti_bubble)
            lppl = np.vectorize(lppl)
            self.lppl = lppl

        return self.lppl(t, self.tc, self.m, self.w, self.a, self.b, self.c1, self.c2)
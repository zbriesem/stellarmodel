import numpy as np
import matplotlib.pyplot as plt

def plot_epsilons():
    from stellarmodel import nuc, opacity, density

    T = np.logspace(np.log10(4e6), 8)
    op = opacity.OpacityTable(.715, .270, .015)
    XCNO = op.XCNO

    epp = nuc.eppeff(T, 2.477e17, .715, .270)
    ecno = nuc.eCNOeff(T, 2.477e17, .715, .015 * XCNO)

    plt.loglog(T, epp + ecno, label='Total')
    plt.loglog(T, epp, '--', alpha=.5, label='pp-chain')
    plt.loglog(T, ecno, '--',alpha=.5, label='CNO')
    plt.title('ZAMS hydrogen burning')
    plt.xlim(4e6, 1e8)
    plt.ylim(1e9, 1e28)
    plt.legend()
    plt.show()
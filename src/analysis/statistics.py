import numpy as np
from scipy import stats


def paired_ttest(scores_a, scores_b):
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    t_stat, p_value = stats.ttest_rel(a, b)
    return float(p_value)


def wilcoxon_test(scores_a, scores_b):
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    stat, p_value = stats.wilcoxon(a, b)
    return float(p_value)


def cohens_d(scores_a, scores_b):
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    diff = np.mean(a) - np.mean(b)
    n1, n2 = len(a), len(b)
    s_pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    if s_pooled == 0:
        return 0.0
    return float(diff / s_pooled)


def confidence_interval_95(scores):
    arr = np.array(scores, dtype=float)
    n = len(arr)
    if n < 2:
        return (float(np.mean(arr)), float(np.mean(arr)))
    mean = np.mean(arr)
    sem = stats.sem(arr)
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem)
    return (float(ci[0]), float(ci[1]))


def full_comparison_report(methods_dict, save_path=None):
    methods = sorted(methods_dict.keys())
    n = len(methods)

    lines = ["method_a,method_b,ttest_p,ttest_sig,wilcoxon_p,wilcoxon_sig,cohens_d"]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = methods[i], methods[j]
            tt = paired_ttest(methods_dict[a], methods_dict[b])
            wt = wilcoxon_test(methods_dict[a], methods_dict[b])
            cd = cohens_d(methods_dict[a], methods_dict[b])
            lines.append(f"{a},{b},{tt:.6f},{tt<0.05},{wt:.6f},{wt<0.05},{cd:.4f}")

    report = "\n".join(lines)
    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
    return report

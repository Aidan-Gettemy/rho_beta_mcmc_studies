"""Run every report whose experiment has results, then the combined summary."""
import _common  # noqa: F401  (first: caps BLAS threads before NumPy loads)
import report_setup
import report_e1
import report_mpsrf
import report_grid
import report_curve
import report_prior_test
import report_sampler
import report_soft
import report_soft_prior_test
import report_combined

if __name__ == "__main__":
    for mod in (report_setup, report_e1, report_mpsrf, report_grid, report_curve, report_prior_test,
                report_sampler, report_soft, report_soft_prior_test, report_combined):
        print(f"== {mod.__name__}")
        mod.main([]) if mod in (report_e1, report_mpsrf, report_soft) else mod.main()

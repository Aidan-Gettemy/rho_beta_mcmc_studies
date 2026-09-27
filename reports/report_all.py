"""Run every report whose experiment has results, then the combined summary."""
import report_setup
import report_e1
import report_grid
import report_curve
import report_prior_test
import report_sampler
import report_combined

if __name__ == "__main__":
    for mod in (report_setup, report_e1, report_grid, report_curve, report_prior_test,
                report_sampler, report_combined):
        print(f"== {mod.__name__}")
        mod.main([]) if mod is report_e1 else mod.main()

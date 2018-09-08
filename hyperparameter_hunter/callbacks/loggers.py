##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseLoggerCallback
from hyperparameter_hunter.reporting import format_evaluation_results, format_fold_run
from hyperparameter_hunter.settings import G
from hyperparameter_hunter.utils.general_utils import sec_to_hms

##################################################
# Import Miscellaneous Assets
##################################################
import inspect


# TODO: Look into "from tabulate import tabulate" tables for temporary advanced logging
# TODO: See if there is a way of making a running table, where not all values are given and rows are printed at intervals
# FLAG: Will need to install tabulate. See "skorch==0.2.0" file "skorch/callbacks/logging.py"


class LoggerFitStatus(BaseLoggerCallback):
    float_format = "{:.5f}"
    log_separator = "  |  "
    # FLAG: Add means of updating float_format to "G.Env.reporting_handler_params['float_format']"

    def __init__(self):
        """Uncalled - See 'Notes' section of :class:`callbacks.bases.BaseCallback` for details"""
        self.stat_aggregates = None
        self.last_evaluation_results = None
        self.current_seed = None
        self._rep = None
        self._fold = None
        self._run = None
        super().__init__()

    def on_experiment_start(self):
        G.log("\n", previous_frame=inspect.currentframe().f_back)
        super().on_experiment_start()

    def on_repetition_start(self):
        G.log(f"Starting Repetition {self._rep}", previous_frame=inspect.currentframe().f_back)
        G.log("", previous_frame=inspect.currentframe().f_back)
        super().on_repetition_start()

    def on_fold_start(self):
        # fold_start_time = self.stat_aggregates['times']['folds'][-1]
        G.log("", previous_frame=inspect.currentframe().f_back)
        super().on_fold_start()

    def on_run_start(self):
        content = ""
        content += format_fold_run(fold=self._fold, run=self._run)
        content += format(self.log_separator if content != "" and self.current_seed else "")
        content += "Seed: {}".format(self.current_seed) if self.current_seed else ""
        G.log(content, previous_frame=inspect.currentframe().f_back, add_time=True)
        super().on_run_start()

    def on_run_end(self):
        content = [
            format_fold_run(fold=self._fold, run=self._run),
            format_evaluation_results(self.last_evaluation_results, float_format=self.float_format),
            f"Time Elapsed: {sec_to_hms(self.stat_aggregates['times']['runs'][-1], as_str=True)}",
        ]

        G.log(self.log_separator.join(content), previous_frame=inspect.currentframe().f_back)
        super().on_run_end()

    def on_fold_end(self):
        content = "F{}.{} AVG:   ".format(self._rep, self._fold)

        content += format_evaluation_results(
            self.last_evaluation_results, float_format=self.float_format
        )

        content += self.log_separator if not content.endswith(" ") else ""

        content += "Time Elapsed: {}".format(
            sec_to_hms(self.stat_aggregates["times"]["folds"][-1], as_str=True)
        )

        G.log(content, previous_frame=inspect.currentframe().f_back, add_time=False)
        super().on_fold_end()

    def on_repetition_end(self):
        content = ""
        content += "Repetition {} AVG:   ".format(self._rep)
        content += format_evaluation_results(
            self.last_evaluation_results, float_format=self.float_format
        )
        content += self.log_separator if not content.endswith(" ") else ""
        content += "Time Elapsed: {}".format(
            sec_to_hms(self.stat_aggregates["times"]["reps"][-1], as_str=True)
        )

        G.log("", previous_frame=inspect.currentframe().f_back)
        G.log(content, previous_frame=inspect.currentframe().f_back)
        super().on_repetition_end()

    def on_experiment_end(self):
        content = "FINAL:    "

        content += format_evaluation_results(
            self.last_evaluation_results, float_format=self.float_format
        )
        content += self.log_separator if not content.endswith(" ") else ""

        content += "Time Elapsed: {}".format(
            sec_to_hms(self.stat_aggregates["times"]["total_elapsed"], as_str=True)
        )

        G.log("")
        G.log(content, previous_frame=inspect.currentframe().f_back, add_time=False)
        super().on_experiment_end()


class LoggerOOF(BaseLoggerCallback):
    pass


class LoggerHoldout(BaseLoggerCallback):
    pass


class LoggerTest(BaseLoggerCallback):
    pass


class LoggerEvaluation(BaseLoggerCallback):
    pass


if __name__ == "__main__":
    pass

##################################################
# Import Own Assets
##################################################
from hyperparameter_hunter.callbacks.bases import BaseLoggerCallback
from hyperparameter_hunter.reporting import format_evaluation, format_fold_run
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
    stat_aggregates: dict
    last_evaluation_results: dict
    current_seed: int
    _rep: int
    _fold: int
    _run: int

    float_format = "{:.5f}"
    log_separator = "  |  "
    # FLAG: Add means of updating float_format to "G.Env.reporting_params['float_format']"

    def on_experiment_start(self):
        G.log("", previous_frame=inspect.currentframe().f_back)
        super().on_experiment_start()

    def on_repetition_start(self):
        if G.Env.verbose >= 3 and G.Env.cv_params.get("n_repeats", 1) > 1:
            G.log("", previous_frame=inspect.currentframe().f_back)
        super().on_repetition_start()

    def on_fold_start(self):
        if G.Env.verbose >= 4 and G.Env.runs > 1:
            G.log("", previous_frame=inspect.currentframe().f_back)
        super().on_fold_start()

    def on_run_start(self):
        content = format_fold_run(rep=self._rep, fold=self._fold, run=self._run)
        content += format(self.log_separator if content != "" and self.current_seed else "")
        content += "Seed: {}".format(self.current_seed) if self.current_seed else ""

        if G.Env.verbose >= 4 and G.Env.runs > 1:
            G.log(content, previous_frame=inspect.currentframe().f_back, add_time=True)
        else:
            G.debug(content, previous_frame=inspect.currentframe().f_back, add_time=True)
        super().on_run_start()

    def on_run_end(self):
        content = [
            format_fold_run(rep=self._rep, fold=self._fold, run=self._run),
            format_evaluation(self.last_evaluation_results, float_format=self.float_format),
            self.__elapsed_helper("runs"),
        ]

        if G.Env.verbose >= 3 and G.Env.runs > 1:
            G.log(self.log_separator.join(content), previous_frame=inspect.currentframe().f_back)
        else:
            G.debug(self.log_separator.join(content), previous_frame=inspect.currentframe().f_back)
        super().on_run_end()

    def on_fold_end(self):
        content = format_fold_run(rep=self._rep, fold=self._fold, run="-")
        content += self.log_separator if not content.endswith(" ") else ""
        content += format_evaluation(self.last_evaluation_results, float_format=self.float_format)
        content += self.log_separator if not content.endswith(" ") else ""
        content += self.__elapsed_helper("folds")

        if G.Env.verbose >= 2 and G.Env.cv_params["n_splits"] > 1:
            G.log(content, previous_frame=inspect.currentframe().f_back, add_time=False)
        else:
            G.debug(content, previous_frame=inspect.currentframe().f_back, add_time=False)
        super().on_fold_end()

    def on_repetition_end(self):
        content = format_fold_run(rep=self._rep, fold="-", run="-")
        content += self.log_separator if not content.endswith(" ") else ""
        content += format_evaluation(self.last_evaluation_results, float_format=self.float_format)
        content += self.log_separator if not content.endswith(" ") else ""
        content += self.__elapsed_helper("reps")

        if G.Env.verbose >= 2 and G.Env.cv_params.get("n_repeats", 1) > 1:
            G.log(content, previous_frame=inspect.currentframe().f_back)
        else:
            G.debug(content, previous_frame=inspect.currentframe().f_back)
        super().on_repetition_end()

    def on_experiment_end(self):
        content = "FINAL:    "

        content += format_evaluation(self.last_evaluation_results, float_format=self.float_format)
        content += self.log_separator if not content.endswith(" ") else ""
        content += self.__elapsed_helper("total_elapsed")

        G.log("")
        G.log(content, previous_frame=inspect.currentframe().f_back, add_time=False)
        super().on_experiment_end()

    def __elapsed_helper(self, period):
        times = self.stat_aggregates["times"]
        if period == "total_elapsed":
            return "Time Elapsed: {}".format(sec_to_hms(times[period], as_str=True))
        else:
            return "Time Elapsed: {}".format(sec_to_hms(times[period][-1], as_str=True))


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

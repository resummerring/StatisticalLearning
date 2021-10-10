import time
import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import Union, List, Callable, Tuple
from StatisticalLearning.Toolbox.Logger import Logger


logger = Logger.get_logger(level='info')


class ParallelEngine:

    """
    A multiprocessing engine with pd.Series / pd.DataFrame as output
    """

    def __init__(self,
                 num_thread: int = 2,
                 batch: int = 1,
                 linear_partition: bool = True):
        """
        :param num_thread: int, number of processors to run
        :param batch: int, number of jobs per processor
        :param linear_partition: bool, if True, use linear partition else double-nested
        """

        self._batch = batch
        self._num_thread = num_thread
        self._use_linear_partition = linear_partition

    # ====================
    #  Private
    # ====================

    @staticmethod
    def _report_progress(index: int, num_jobs: int, start: float, task: str):
        """
        Report the running status of jobs

        :param index: int, index of running job
        :param num_jobs: int, total number of jobs to run
        :param start: float, timestamp of start run time
        :param task: str, name of running task
        """

        progress, minutes = float(index) / num_jobs, (time.time() - start) / 60.
        message = [progress, minutes, minutes * (1 / progress - 1)]

        time_stamp = str(datetime.datetime.fromtimestamp(time.time()))
        message = time_stamp + ' ' + str(round(message[0] * 100, 2)) + '% ' + task + ' done after' + \
            str(round(message[1], 2)) + ' minutes. Remaining ' + str(round(message[2], 2)) + ' minutes.'

        logger.info(message)

    @staticmethod
    def _expand_call_back(kwargs: dict) -> Union[pd.Series, pd.DataFrame]:
        """
        Transform a dictionary into a task
        """

        func = kwargs['func']
        del kwargs['func']

        return func(**kwargs)

    def _linear_partition(self, num_atoms: int) -> np.ndarray:
        """
        Atoms are indivisible tasks. When running jobs in parallel, we want to group atoms into
        molecules, which can be processed in parallel using multiple processors. Each molecule is
        a subset of atoms that will be processed sequentially, by a callback function.

        Linear partition: partition a list of atoms in subsets of equal size

        :param num_atoms: int, number of individual tasks
        """

        parts = np.linspace(0, num_atoms, min(self._num_thread * self._batch, num_atoms) + 1)
        return np.ceil(parts).astype(int)

    def _nested_partition(self, num_atoms: int, upperTriDiag: bool = False) -> np.ndarray:
        """
        When we loop over {(i, j) | 1 <= j <= i, i = 1, 2, ..., N}, we have 0.5 * N * (N + 1)
        atoms in total. If we want to partition it into M processors, then each processor should
        take approximately 0.5 * N * (N + 1) / M atoms.
        Processor k will take care of the r(k - 1) + 1 to r(k) rows of atoms:
                    {(i, j) | 1 <= j <= i, i = r(k - 1) + 1, ..., r(k)}

        S(k) = {r(k - 1) + 1, r(k - 1) + 2, ... , r(k)}
         -> #S(k) = 0.5 * (r(k) + r(k - 1) + 1) * (r(k) - r(k - 1)) = 0.5 * N * (N + 1) / M
         -> r(k) = (-1 + sqrt{1 + 4 * [r(k - 1)^2 + r(k - 1) + N * (N + 1) / M] }) / 2

        :param num_atoms: int, number of individual tasks
        :param upperTriDiag: bool, if True, the heaviest processor will be the first
        """

        parts, num_threads = [0], min(self._num_thread * self._batch, num_atoms)
        for num in range(num_threads):
            part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.) / num_threads)
            parts.append(0.5 * (-1 + part ** 0.5))

        parts = np.round(parts).astype(int)

        if upperTriDiag:
            parts = np.cumsum(np.diff(parts)[::-1])
            parts = np.append(np.array([0]), parts)

        return parts

    def _process_jobs_without_parallel(self, jobs: List[dict]) -> List[Union[pd.Series, pd.DataFrame]]:
        """
        Turn off multiprocessing and run jobs sequentially for debugging

        :param jobs: List[dict], a collection of jobs to run
        """

        output = []

        for job in jobs:
            out = self._expand_call_back(job)
            output.append(out)

        return output

    def _process_jobs_with_parallel(self, jobs: List[dict]) -> List[Union[pd.Series, pd.DataFrame]]:

        """
        Turn on multiprocessing and run jobs in parallel

        :param jobs: List[dict], a collection of jobs to run
        """

        task = jobs[0]['func'].__name__

        pool = mp.Pool(processes=self._num_thread)
        output, start = pool.imap_unordered(self._expand_call_back, jobs), time.time()

        result = []

        for index, out in enumerate(output):
            self._report_progress(index, len(jobs), start, task)

        pool.close()
        pool.join()

        return result

    # ====================
    #  Public
    # ====================

    def run(self, func: Callable, objects: Tuple[str, List], **kwargs: dict) -> List[Union[pd.Series, pd.DataFrame]]:
        """
        Main function to run tasks in parallel

        :param func: Callable, a call back function to be executed in parallel
        :param objects: Tuple, objects[0]: name of argument passed to call back function,
                               objects[1]: list of individual tasks - atoms
        :param kwargs: dict, arguments passed to func
        """

        if self._use_linear_partition:
            parts = self._linear_partition(len(objects[1]))
        else:
            parts = self._nested_partition(len(objects[1]))

        jobs = []

        for i in range(1, len(parts)):
            job = {objects[0]: objects[1][parts[i - 1]: parts[i]], 'func': func}
            job.update(kwargs)
            jobs.append(job)

        if self._num_thread == 1:
            result = self._process_jobs_without_parallel(jobs)
        else:
            result = self._process_jobs_with_parallel(jobs)

        return result

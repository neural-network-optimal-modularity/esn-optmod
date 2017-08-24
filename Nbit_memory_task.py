import numpy as np

class BoundaryViolationError(LookupError):
    """Raise when some boundary condition fails"""

class Nbit_memory_task():

    def __init__(self, **kwargs):
        """
        """
        task_defaults = {'distractor_value': 1,
            'sequence_dimension': 1,
            'start_time': 0,
            'sequence_length': 1,
            'distraction_duration': 0,
            'cue_value': 1,
            'loop_unique_input': False}

        for key in kwargs.keys():
            if key not in task_defaults.keys():
                raise KeyError(key + " not a valid key")

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.recall_time = self.start_time + self.sequence_length + self.distraction_duration + 1
        self.total_duration = self.start_time + self.sequence_length + self.distraction_duration + 1 + self.sequence_length
        self.input_dimension = self.sequence_dimension + 2

        if self.loop_unique_input:
            self.generate_all_configurations()

        # if self.sequence_length > self.sequence_dimension:
        #     raise BoundaryViolationError("Sequence length can not be greater than sequence dimension.")

    def generate_all_configurations(self):
        """only use for small dimensions"""
        import itertools
        import random

        full_index_set = list(itertools.product(range(self.sequence_dimension), repeat=self.sequence_length))
        random.shuffle(full_index_set)

        self.arr_signal_set = np.zeros((len(full_index_set), self.sequence_length, self.sequence_dimension))
        for i, index_set in enumerate(full_index_set):
            for j in range(self.sequence_length):
                self.arr_signal_set[i, j, index_set[j]] = 1

        self.looping_index = 0

    def generate_signal(self):
        """
        Generates a time-series and a target time-series
        The time-series is of dimensions (T+2Sl+1) x (Sd + 2)
        The target is of dimensions (Sl) X (Sd)
        Sl is the sequence length
        Sd is the sequence dimension
        T is the time duration of the distractor (recall time = distractor duration + Sl + start time)
        """

        if self.loop_unique_input:
            target_sequence = self.arr_signal_set[self.looping_index]
            self.looping_index = (self.looping_index + 1) % self.arr_signal_set.shape[0]

        else:
            # Create sequence to memorize
            activation_indices = np.random.choice(range(self.sequence_dimension), size=self.sequence_length, replace=True)
            target_sequence = np.zeros(shape=(self.sequence_length, self.sequence_dimension))
            for i in xrange(self.sequence_length):
                target_sequence[i, activation_indices[i]] = 1

        # Create 0 padding following memory sequence (or before if start is later)
        memory_series = np.concatenate((np.zeros(shape=(self.start_time, self.sequence_dimension)), 
            target_sequence, np.zeros(shape=(self.distraction_duration + 1 + self.sequence_length,
                self.sequence_dimension))), axis=0)

        # Create distractor series
        distractor_series = np.concatenate((self.distractor_value * np.ones(shape=(self.start_time, 1)),
            np.zeros(shape=(self.sequence_length, 1)),
            self.distractor_value * np.ones(shape=(self.distraction_duration, 1)),
            np.zeros(shape=(1, 1)),
            self.distractor_value * np.ones(shape=(self.sequence_length, 1))) , axis=0)

        # Create cue series
        cue_series = np.concatenate((np.zeros(shape=(self.start_time + self.sequence_length + self.distraction_duration, 1)),
            self.cue_value * np.ones(shape=(1,1)), np.zeros(shape=(self.sequence_length,1))), axis=0)

        return np.concatenate((memory_series, distractor_series, cue_series), axis=1), target_sequence

if __name__ == '__main__':
    testtask = Nbit_memory_task(distractor_value=1, sequence_dimension=5, sequence_length=5, start_time=1, distraction_duration=3)
    test_seq = testtask.generate_signal()
    print test_seq
    print testtask.recall_time
    print test_seq[0][testtask.recall_time:]
import numpy as np

class base_task(object):

    def __init__(self):

        self.input_dimensions = (0,0)
        self.output_dimensions = (0,0) 
        self.recall_time = 0

    def get_input_dimensions(self):

        return self.input_dimensions

    def get_output_dimensions(self):

        return self.output_dimensions

    def generate_signal(self):

        return None

class null_task(base_task):

    def __init__(self, **kwargs):

        task_defaults = {
            'sequence_dimension': 2,
            'sequence_length': 3,
            'normalized_input': True,
            'duration': 100
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.total_duration = self.sequence_length + self.duration
        self.input_dimension = self.sequence_dimension
        self.input_dimensions = (self.total_duration, self.input_dimension)
        self.output_dimensions = (1,1)

    def generate_signal(self):

        activation_sequence = np.random.random((self.sequence_length, self.sequence_dimension))

        if self.normalized_input:
            for i in range(self.sequence_length):
                activation_sequence[i] = activation_sequence[i] / np.sum(activation_sequence[i])

        return np.concatenate((activation_sequence, np.zeros((self.duration, self.sequence_dimension))), axis=0), np.zeros((1,1))

class binary_memory_task(base_task):
    """
    Memorizes the sequence of largest input values from a normalized input sequence
    """

    def __init__(self, **kwargs):

        task_defaults = {
            'sequence_dimension' : 2,
            'sequence_length': 3,
            'normalized_input': True,
            'distraction_duration': 1,
            'distractor_value': 0
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.recall_time = self.sequence_length + self.distraction_duration + 1
        self.total_duration = self.sequence_length + self.distraction_duration + 1 + self.sequence_length
        self.input_dimension = self.sequence_dimension + 2

        self.input_dimensions = (self.total_duration, self.input_dimension)
        self.output_dimensions = (self.sequence_length, self.sequence_dimension)

    def generate_signal(self):

        # Create sequence to memorize
        activation_sequence = np.random.random((self.sequence_length, self.sequence_dimension))

        if self.normalized_input:
            for i in range(self.sequence_length):
                activation_sequence[i] = activation_sequence[i] / np.sum(activation_sequence[i])

        target_sequence = np.zeros(shape=(self.sequence_length, self.sequence_dimension))
        for i in xrange(self.sequence_length):
            target_sequence[i, np.argmax(activation_sequence[i])] = 1

        # Create 0 padding following memory sequence (or before if start is later)
        memory_series = np.concatenate((np.zeros(shape=(0, self.sequence_dimension)), 
            activation_sequence, np.zeros(shape=(self.distraction_duration + 1 + self.sequence_length,
                self.sequence_dimension))), axis=0)

        # Create distractor series
        distractor_series = np.concatenate((self.distractor_value * np.ones(shape=(0, 1)),
            np.zeros(shape=(self.sequence_length, 1)),
            self.distractor_value * np.ones(shape=(self.distraction_duration, 1)),
            np.zeros(shape=(1, 1)),
            self.distractor_value * np.ones(shape=(self.sequence_length, 1))) , axis=0)

        # Create cue series
        cue_series = np.concatenate((np.zeros(shape=(self.sequence_length + self.distraction_duration, 1)),
            np.ones(shape=(1,1)), np.zeros(shape=(self.sequence_length,1))), axis=0)

        return np.concatenate((memory_series, distractor_series, cue_series), axis=1), target_sequence

class poly_task(base_task):
    """
    """

    def __init__(self, **kwargs):

        super(poly_task, self).__init__()

        task_defaults = {
            'sequence_dimension': 2,
            'sequence_length': 3,
            'exponent_sequence': (1,1),
            'normalized_input': True
        }

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.input_dimensions = (self.sequence_length, self.sequence_dimension)
        self.output_dimensions = (self.sequence_length, 1)

    def generate_signal(self):
        """
        Generates a time-series of random values and a target time-series
        Input time-series are of dimensions (sequence_length) X (sequence_dimension)
        target time-series are of dimensions (sequence_length) X (1)
        The input time-series is normalized so that the inputs along the dimensions
        sum to one (if the normalized_input option is active)
        """

        input_sequence = np.random.random((self.sequence_length, self.sequence_dimension))

        if self.normalized_input:
            for i in range(self.sequence_length):
                input_sequence[i] = input_sequence[i] / np.sum(input_sequence[i])

        tar_sequence = np.ones((self.sequence_length, 1))
        for i in range(self.sequence_length):
            for j in range(self.sequence_dimension):
                tar_sequence[i] *= np.power(input_sequence[i][j], self.exponent_sequence[j])

        return input_sequence, tar_sequence

class find_largest_task(poly_task):
    """
    Apply poly task and return largest of the set of polys evalutated
    """

    def __init__(self, distraction_duration=0, **kwargs):

        super(find_largest_task, self).__init__(**kwargs)

        self.distraction_duration = distraction_duration
        self.recall_time = self.sequence_length + self.distraction_duration
        self.input_dimensions = (self.recall_time + 1, self.sequence_dimension)
        self.output_dimensions = (1,1)

    def generate_signal(self):
        """
        Generates a time-series of random values and a target time-series
        Input time-series are of dimensions (sequence_length) X (sequence_dimension)
        target time-series are of dimensions (1)X(1)
        """

        # Create initial input sequence
        input_sequence = np.random.random((self.sequence_length, self.sequence_dimension))

        if self.normalized_input:
            for i in range(self.sequence_length):
                input_sequence[i] = input_sequence[i] / np.sum(input_sequence[i])

        # Find target value for output
        eval_sequence = np.ones((self.sequence_length, 1))
        for i in range(self.sequence_length):
            for j in range(self.sequence_dimension):
                eval_sequence[i] *= np.power(input_sequence[i][j], self.exponent_sequence[j])

        tar_output = np.zeros((1,1))
        tar_output[0,0] = np.max(eval_sequence)

        # Append distractor period to input_squence until recall time
        return np.concatenate((input_sequence, np.zeros((self.distraction_duration+1, self.sequence_dimension))), axis=0), tar_output

if __name__ == '__main__':
    """
    for testing
    """
    # test_task = poly_task(sequence_dimension=3, sequence_length=2, exponent_sequence=(2,1,1))
    # print(test_task.generate_signal())

    # test_task = binary_memory_task()
    # print(test_task.generate_signal())

    # test_task = find_largest_task(10)
    # print(test_task.generate_signal())

    # test_task = null_task(sequence_dimension=10, sequence_length=10, duration=0)
    # print(test_task.generate_signal())
    pass
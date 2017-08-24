import numpy as np
import operator
import re

class BMP_task(object):
    """
    The BMP task is used to generate linear memory and non-linear processing tasks and
    combinations of the two. The BMP generates a n-dimensional sequence of random inputs
    and it demands recall at some later time. For linear memory, it demands recall only
    given some deltaT. For non-linear processing it constructs a non-linear boolean table
    that must be learned. Given the input sequence a single output value is required.
    When combined, a deltaT time difference is needed between the ouput and the computation.
    """
    def __init__(self, **kwarg):
        """
        distractor_value - what value the distractor will have (put 0 for no distractor)
        sequence_dimension - the dimension of the input sequence of 0/1's
        distraction_duration - interum between signal input and expected response
        start_time - time when task begins and input is given
        sequence_length - the duration of the sequence input
        proc - Some processing of the input seq
                will occur. proc is the bmp_function obj that will be called to act
                upon the input seq. proc must take as input an 
                (sequence_dimension X sequence_length) array and return a
                target sequence which must be matched following recall cue
                proc should have an attribute .dim that specifies the output dimensionality
                as a tuple.
        """

        task_defaults = {'distractor_value': 1,
            'sequence_dimension': 1,
            'start_time': 0,
            'sequence_length': 1,
            'distraction_duration': 0,
            'proc': None,
            'proc_args': None}

        for key in kwargs.keys():
            if key not in task_defaults.keys():
                raise KeyError(key + " not a valid key")

        for key, default in task_defaults.iteritems():
            setattr(self, key, kwargs.get(key, default))

        self.recall_time = self.start_time + self.sequence_length + self.distraction_duration + 1
        self.input_dimension = self.sequence_dimension + 2


        self.total_duration = self.start_time + self.sequence_length + self.distraction_duration + 1 + self.sequence_length

class bmp_function(object):

    def __init__(self, time_axis_len, sig_dim):

        self.dim = (time_axis_len, sig_dim)

    def __call__(self, input):

        pass

class memory_bmp_function(bmp_function):

    def __init__(self, sequence_length, sequence_dimension):

        super(memory_function, self).__init__(sequence_length, sequence_dimension)

    def __call__(self, input):

        return input

class boolean_bmp_function(bmp_function):
    """
    """

    def __init__(self):

        super(bmp_function, self).__init__((1,1))

class nonlinear_boolean_bmp_function(boolean_bmp_function):
    """
    """

    base_1 = 

    def __init__(self, ??):

    def __call__(self, input):

class linear_boolean_bmp_function(boolean_bmp_function):

    def __init__(self):

class boolean_function(object):

    def __init__(self, raw_expression):

        tokens = self.tokenize(raw_expression)
        self.dict_variables = { token : bf_state() for token in tokens if type(token) == int }

    def tokenize(self, raw_expression):

        tokens = []
        digit = ''
        valid_nondigit_characters = set([')','(','+','*'])
        for char in raw_expression:

            if char.isdigit():
                digit += char                

            elif char in valid_nondigit_characters:

                if digit != '':
                    token.append(int(digit))
                    digit = ''

                tokens.append(char)

        return tokens

    def infix_to_postfix(self, tokenized_expression):
        """
        Using the shunting_yard algorithm
        """

        operator = set(['+', '*'])
        output = []
        operator_stack = []
        for token in tokenized_expression:

            if type(token) == int:
                output.append(token)

            elif token in operator:
                while (len(operator_stack) != 0) and \
                    (operator_stack[-1] in operator) and \
                    ((operator_stack[-1] != '+') and (token != '*')):

                    output.append(operator_stack.pop())

                operator_stack.append(token)

            elif token == '(':
                operator_stack.append(token)

            elif token == ')':
                while (len(operator_stack) != 0) and (operator_stack[-1] != '('):
                    output.append(operator_stack.pop())
                output.pop()

        while len(operator_stack) != 0:
            output.append(operator_stack.pop())

        return output

    def infix_to_prefix(self, tokenized_expression):

        operator = set(['+', '*'])
        output = []
        operator_stack = []

        for token in reversed(tokenized_expression):

            if type(token) == int:
                output.append(token)

            elif token in operator:
                while (len(operator_stack) != 0) and \
                    (operator_stack[-1] in operator) and \
                    ((operator_stack[-1] != '+') and (token != '*')):

                    output.append(operator_stack.pop())

                operator_stack.append(token)

            elif token == ')':
                operator_stack.append(token)

            elif token == '(':
                while (len(operator_stack) != 0) and (operator_stack[-1] != ')'):
                    output.append(operator_stack.pop())
                output.pop()

        while len(operator_stack) != 0:
            output.append(operator_stack.pop())

        # Turn stack back into queue for iteration
        output.reverse()

        return output

    def parse_expression(self, tokenized_expression):

        token = next(tokenized_expression)

class bf_state(object):
    """
    A leaf component for boolean functions. It acts as a 
    state variable that can be changed. It is a terminal
    in a sense during activation, but can be modified.
    """

    def __init__(self, state=0):
        """
        State must be a boolean value (True/False)
        Defaults to 0
        """

        self.state = state

    def __str__(self):

        return str(self.state)

    def __and__(self, b):

        return self.state & b

    def __or__(self, b):

        return self.state | b

    def __xor__(self, b):

        return self.state ^ b

    def __invert__(self):

        return ~self.state

class bf_unit(object):
    """
    A binary functional unit that has three elements, an operator and
    two others units or terminals that the operator acts on.

    bf_units can be constructed in a tree-like structure where the bottom
    is evaluated first and the evaluation going up the tree is saved so
    it doesn't have to be repeated at each function call.

    The eval method can override the saved evaluations and run it again.
    This maybe desired if one of the units has changed somewhere in the tree.
    """

    def __init__(self, op, unit1, unit2):

        self.op = op
        self.unit1 = unit1
        self.unit2 = unit2
        self.value = None

    def __and__(self, b):

        return self.eval() & b

    def __or__(self, b):

        return self.eval() | b

    def __xor__(self, b):

        return self.eval() ^ b

    def __invert__(self):

        return ~self.eval()

    def __call__(self, force=False):

        # This automatically forces re-evalutation
        if (type(self.unit1) is bf_unit) and force:
            self.unit1(force)
        if (type(self.unit2) is bf_unit) and force:
            self.unit2(force)

        self.value = self.op(self.unit1, self.unit2)
        return self.value

    def eval(self, force=False):
        """
        Force acts differently then when self.value is None.
        When self.value is None, only the topmost unevaluated
        branches are evaluated in order to get this unit's
        value. If Force is true, then the whole tree is
        re-evaluated.

        If self.value is alread defined and force is False,
        then the saved value is returned.
        """

        if force:
            return self.__call__(force)

        # Check if unit has already been evalutated
        if self.value == None:
            return self.__call__()
        elif:
            return self.value

if __name__ == '__main__':
    """
    """

    pass
class Parameters:
    def __init__(self, function, num_args, arguments, num_optional_args = 0, optional_arguments=[]):
        self.function = function
        self.num_args = num_args
        self.arguments = arguments
        self.num_optional_args = num_optional_args
        self.optional_arguments = optional_arguments
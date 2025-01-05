class AllTrue(str):
    def __init__(self, representation=None) -> None:
        self.repr = representation
        pass
    def __ne__(self, __value: object) -> bool:
        return False
    # isinstance, jsonserializable hijack
    def __instancecheck__(self, instance):
        return True
    def __subclasscheck__(self, subclass):
        return True
    def __bool__(self):
        return True
    def __str__(self):
        return self.repr
    # jsonserializable hijack
    def __jsonencode__(self):
        return self.repr
    def __repr__(self) -> str:
        return self.repr
    def __eq__(self, __value: object) -> bool:
        return True
anytype = AllTrue("*") # when a != b is called, it will always return False

class AnyToStringNode:
    def __init__(self):
        # Initialize any necessary parameters for the node
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (anytype, {"default": 0.0}),
            }
        }

    RETURN_TYPES = ("STRING",)  # Output type is string
    CATEGORY = "RunningHub"      # Category name is RunningHub
    FUNCTION = "process"         # The processing function is the 'process' method

    def process(self, anything):
        """
        Converts any input type to a string.
        If the input is a string that can be converted to an integer, it performs the conversion.
        Otherwise, it directly converts the input to a string.
        """
        if isinstance(anything, str):
            try:
                # Attempt to convert the string to an integer and then back to string
                return (str(int(anything)),)
            except ValueError:
                # If conversion fails, return the original string
                return (anything,)
        else:
            # For non-string types, directly convert to string
            return (str(anything),)

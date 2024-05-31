"""The type parameter refers to the fact that SingletonMeta is a metaclass. 
A metaclass in Python is a subclass of the built-in type class.
The type class itself is a metaclass, responsible for creating classes."""
class SingletonMeta(type):
    """
    This is a metaclass that creates a Singleton base class when called.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # If an instance of the class doesn't exist, create one and store it in the _instances dictionary
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        # Return the stored instance
        return cls._instances[cls]
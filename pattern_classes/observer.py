
class Subject:
    def __init__(self):
        """
        Initializes the Subject with an empty list of observers.
        """
        self._observers = []

    def attach(self, observer):
        """
        Attaches an observer to the list of observers.

        Parameters:
            observer: The observer instance to be attached.

        Returns:
            None
        """
        self._observers.append(observer)

    def detach(self, observer):
        """
        Removes the specified observer from the list of observers.

        Parameters:
            observer: The observer instance to be removed.

        Returns:
            None
        """
        self._observers.remove(observer)

    def notify(self, message={'type': 'fetch'}):
        """
        Notifies all observers in the list by calling their update method with a message.
        
        Parameters:
            message (dict): A dictionary specifying the type of message to be sent. Default is {'type': 'fetch'}.
        
        Returns:
            None
        """
        for observer in self._observers:
            observer.update(self, message)

class Observer:
    def update(self, subject, message={'type': 'fetch'}):
        """
        Updates the observer with new information from the subject.
        
        Parameters:
            self: The Observer instance.
            subject: The subject providing the new information.
            message (dict): A dictionary specifying the type of message to be processed. Default is {'type': 'fetch'}.
        
        Returns:
            None
        """
        pass
    
class AccuracyLogger(Observer):
    def update(self, subject, message={'type': 'fetch'}):
        """
        Updates the observer with the accuracy information if the message type is 'accuracy'.

        Parameters:
            self: The AccuracyLogger instance.
            subject: The subject instance being observed.
            message (dict): A dictionary specifying the type of message to be processed. Default is {'type': 'fetch'}.

        Returns:
            None
        """
        if message['type'] == 'accuracy':
            print(f"{subject._strategy.__class__.__name__} accuracy: {subject.accuracy*100:.1f}%")

class ProgressMonitor(Observer):
    def update(self, subject, message={'type': 'fetch'}):
        """
        Updates the observer with the training progress if the message type is not 'accuracy'.

        Parameters:
            self: The ProgressMonitor instance.
            subject: The subject instance being observed.
            message (dict): A dictionary specifying the type of message to be processed. Default is {'type': 'fetch'}.

        Returns:
            None
        """
        if message['type'] != 'accuracy':
            print(f"Training progress: {subject.progress}% complete")

class ParameterChanger(Observer):
    def update(self, subject, message={'type': 'fetch'}):
        """
        Updates the observer with new information from the subject.
        
        Parameters:
            self: The Observer instance.
            subject: The subject providing the new information.
            message (dict): A dictionary specifying the type of message to be processed. Default is {'type': 'fetch'}.
        
        Returns:
            None
        """
        # Example: if accuracy is below 90%, increase `max_iter` for LogisticRegression
        if subject.accuracy < 0.9:
            print("Accuracy below 90%, increasing max_iter")
            subject.model.max_iter += 100
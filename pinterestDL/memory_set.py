class MemorySet(list):

    def __init__(self, *args, **kwargs):
        """
        A ordered set that keeps track of the last iteration trough it and starts at the end in the next iteration.
        """
        self.last_len = 0
        super(MemorySet, self).__init__(*args, **kwargs)

    def __iter__(self):
        """
        Start iteration at the last index that was not iterated trough last time.
        """
        iterator = iter(self[self.last_len:])
        self.last_len = len(self)
        return iterator

    def update(self, more_entries):
        """
        Inserts new elements into the MemorySet. Values that are already present in the MemorySet will not be
        added again.

        :param more_entries: Iterable that can be converted to a set. All elements not already present will be added.
        :returns True, if any elements were added, False otherwise.
        """
        temp_entries = set(more_entries)
        additions = [x for x in temp_entries if x not in self]

        if len(additions) == 0:
            return False
        else:
            self.extend(additions)
            return True

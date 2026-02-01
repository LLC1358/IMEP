import numpy as np

###################################################################################################################################################

class Data(object):
    def __init__(self, data, shuffle):
        self.data = data
        self.histories = []
        for clicked_news_sequence in data[2]:
            history = [clicked_news for clicked_news in clicked_news_sequence if clicked_news != 0]
            self.histories.append(history)
        self.histories = np.asarray(self.histories)
        self.candidates = np.asarray(data[0])
        self.labels = np.asarray(data[1])
        self.length = len(self.histories)
        self.shuffle = shuffle

    ###############################################################

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.histories = self.histories[shuffled_arg]
            self.candidates = self.candidates[shuffled_arg]
            self.labels = self.labels[shuffled_arg]

        slices = []
        for start in range(0, self.length, batch_size):
            end = min(start + batch_size, self.length)
            slices.append(np.arange(start, end))

        return slices
    
    ###############################################################

    def get_padding(self, index):
        histories = self.histories[index]
        candidates = self.candidates[index]
        labels = self.labels[index]

        history_lengths = []
        for history in histories:
            history_lengths.append(len(history))
        max_history_length = max(history_lengths)
        history_padding = self.zero_padding(histories, max_history_length)

        candidate_lengths = []
        for candidate in candidates:
            candidate_lengths.append(len(candidate))
        max_candidate_length = max(candidate_lengths)
        candidate_padding = self.zero_padding(candidates, max_candidate_length)

        label_lengths = []
        for label in labels:
            label_lengths.append(len(label))
        max_label_length = max(label_lengths)
        label_padding = self.zero_padding(labels, max_label_length)

        return history_padding, candidate_padding, label_padding

    ###############################################################

    def zero_padding(self, data, max_length):
        zero_padding_data = np.zeros((len(data), max_length), dtype=np.int)
        for i in range(len(data)):
            zero_padding_data[i, :len(data[i])] = data[i]
        '''
        histories  = [[12, 8, 5], [3]]    -> [[12, 8, 5], [3, 0, 0]]
        candidates = [[9, 7], [4, 1, 6]]  -> [[9, 7, 0], [4, 1, 6]]
        labels     = [[0, 1],  [1, 0, 0]] -> [[0, 1, 0], [1, 0, 0]]
        '''

        return zero_padding_data
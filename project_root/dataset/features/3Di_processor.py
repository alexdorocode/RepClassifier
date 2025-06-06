class ThreeDiProcessor(FeatureProcessor):
    def __init__(self, method='onehot', raw_data=None, processed_data=None):
        super().__init__(raw_data, processed_data)
        self.method = method

    def process(self):
        if self.processed_data is not None:
            return self.processed_data
        if self.method == 'onehot':
            self.processed_data = self.one_hot_encode(self.raw_data)
        elif self.method == 'embed':
            self.processed_data = self.embed(self.raw_data)

    def one_hot_encode(self, tokens):
        # Implement fixed-length one-hot encoding
        pass

    def embed(self, tokens):
        # Implement fixed-length learned embedding
        pass

    def reduce_dimensionality(self, method='autoencoder'):
        # Apply AE to fixed-size vector
        pass

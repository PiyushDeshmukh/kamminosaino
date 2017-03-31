output_nodes = 25

class Gesture():
    def __init__(self, label, pixels):
        self.label = label
        self.pixels = pixels

    def get_pixels(self):
        return self.pixels

    def get_label(self):
        return self.label

    def generate_output(self):
        self.outputs = [0] * output_nodes
        self.outputs[ord(self.label) - ord('A')] += 1

    def get_outputs(self):
        return self.outputs

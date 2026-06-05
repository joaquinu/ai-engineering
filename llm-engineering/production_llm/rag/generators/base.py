class Generator:
    def __init__(self, generator=None):
        self.generator = generator

    def generate(self, prompt):
        if self.generator is not None:
            return self.generator(prompt)
        return ""

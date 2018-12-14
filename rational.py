class Rational:

    def __init__(self, numerator, denominator):
        self._numerator = numerator
        self._denominator = denominator

    def __float__(self):
        return self._numerator / self._denominator

    def __int__(self):
        return int(self.__float__())

    @property
    def numerator(self):
        return self._numerator

    @property
    def nu(self):
        return self._numerator

    @property
    def denominator(self):
        return self._denominator

    @property
    def de(self):
        return self._denominator

    def normalize(self, n):
        val = (self.de*n)/self.nu
        if int(val) == val:
            return int(val)
        else:
            raise ValueError(f"Couldn't normalize {n}")

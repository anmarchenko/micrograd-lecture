import math

class Value:
    def __init__(self, data, _children = (), _op='', label='') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data},label={self.label})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * res.grad
            other.grad += 1.0 * res.grad

        res._backward = _backward
        return res

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * res.grad
            other.grad += self.data * res.grad

        res._backward = _backward
        return res

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supports int/float powers for now"
        res = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * res.grad

        res._backward = _backward
        return res

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return self - other

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)

        res = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * res.grad

        res._backward = _backward
        return res

    def exp(self):
        x = self.data
        res = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += res.data * res.grad

        res._backward = _backward
        return res

    def backward(self):
        topo = []
        visited = set()

        def topological_sort(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    topological_sort(child)
                topo.append(v)

        topological_sort(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

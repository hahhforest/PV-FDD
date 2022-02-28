class Computer:
    def __init__(self):
        self.name = "计算器"
        self.color = "blue"

    def plus(self, a, b):
        return a + b

    def multi(self, a, b):
        return a * b

    def substract(self, a, b):
        return a - b

    def divide(self, a, b):
        return a / b

    def compute(self, a, b, op):
        c: float

        if op == '+':
            c = self.plus(a, b)
        elif op == '*':
            c = self.multi(a, b)
        elif op == '-':
            c = self.substract(a, b)
        else:
            c = self.divide(a, b)
        return c

    def exe(self):
        while True:
            print("请输入操作数1: ")
            a = input()
            print("请输入操作数2: ")
            b = input()
            print("请输入运算符：")
            op = input()

            a = float(a)
            b = float(b)
            # 执行
            result = self.compute(a, b, op)

            print("答案是：" + str(result))


if __name__ == '__main__':
    computer1 = Computer()
    computer1.exe()

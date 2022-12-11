from Pyro4 import expose
import array

class Solver:
    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name
        self.workers = workers

    def solve(self):
        (A, B) = self.read_input()
        step = int(len(A) / len(self.workers))

        # Mapping: workerIdx => workerResult
        mapped = {}
        for workerIdx in range(0, len(self.workers)):
            resultMatrixSlice = []
            if workerIdx == len(self.workers) - 1:
                resultMatrixSlice = self.workers[workerIdx].mymap(workerIdx * step, len(A), A, B)
            else:
                resultMatrixSlice = self.workers[workerIdx].mymap(workerIdx * step, min((workerIdx + 1) * step, len(A) - 1), A, B)
            mapped[workerIdx] = resultMatrixSlice

        self.write_output(self.myreduce(mapped))

    @staticmethod
    @expose
    def myreduce(mapped):
        output = {}
        for idx, resultRows in mapped.items():
            output[idx] = resultRows.value
        return output

    @staticmethod
    @expose
    def mymap(rowFrom, rowTo, matrix_A, matrix_B):
        resultMatrixSlice = []
        if len(matrix_B) == 0 or\
           len(matrix_A) == 0 or\
           len(matrix_B[0]) == 0 or\
           len(matrix_A[0]) != len(matrix_B):
               return []

        matrix_B_columns_len = len(matrix_B[0])
        
        for rowIdx in range(rowFrom, rowTo):
            resultRow = []
            for columnIdx in range(0, matrix_B_columns_len):
                column = [row[columnIdx] for row in matrix_B]
                dot_product = sum([i*j for (i, j) in zip(matrix_A[rowIdx], column)])
                resultRow.append(dot_product)
            resultMatrixSlice.append(resultRow)

        return resultMatrixSlice


    def read_input(self):
        with open(self.input_file_name, 'r') as f:
            A = []
            B = []
            procede_A = True
            for line in f:
                if line == '\n':
                    # A reading end
                    procede_A = False
                    continue

                splited_line = list(map(int, line.split(' ')))
                if procede_A:
                    A.append(splited_line)
                else:
                    B.append(splited_line)

            return A, B

    def write_output(self, resultDict):
        result = []
        for key, val in sorted(resultDict.items()):
            result.append(val)

        with open(self.output_file_name, 'w') as f:
            for row in result:
                f.write('\n'.join(map(str,row)) + '\n')






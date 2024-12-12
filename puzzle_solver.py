from z3 import Solver, Bool, And, Or, Not, Implies, sat, unsat


class SudokuSolver:
    def __init__(self, puzzle):
        self.puzzle = puzzle
        self.solver = None
        self.variables = None

    def create_variables(self):
        """
        Set self.variables as a 3D list containing the Z3 variables.
        self.variables[i][j][k] is true if cell i,j contains the value k+1.
        """
        # Your code here

        # Initialize self.variables with trivial 3D list
        self.variables = [[[Bool(0) for k in range(9)]
                           for i in range(9)]for j in range(9)]

        # Fill self.variables with the appropriate Z3 variables
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    self.variables[i][j][k] = Bool(
                        f"x_{i}_{j}_{k}")
        pass

    def encode_rules(self):
        """
        Encode the rules of Sudoku into the solver.
        The rules are:
        1. Each cell must contain a value between 1 and 9.
        2. Each row must contain each value exactly once.
        3. Each column must contain each value exactly once.
        4. Each 3x3 subgrid must contain each value exactly once.
        """

        # Encoding for rule 1
        and_clause = []
        for i in range(9):
            for j in range(9):
                or_clause = []
                for k in range(9):
                    or_clause.append(self.variables[i][j][k])
                and_clause.append(Or(or_clause))
        self.rule1 = And(and_clause)

        # Encoding for rule 2: If cell has X then no other cell in same row has X.
        and_clause = []
        for j in range(9):
            for i in range(9):
                for k in range(9):
                    not_or_clause = []
                    for j1 in range(j+1, 9):
                        not_or_clause.append(self.variables[i][j1][k])
                    and_clause.append(
                        Implies(self.variables[i][j][k], Not(Or(not_or_clause))))
        self.rule2 = And(and_clause)

        # Encoding for rule 3: If cell has X then no other cell in same column has X.
        and_clause = []
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    not_or_clause = []
                    for i1 in range(i+1, 9):
                        not_or_clause.append(self.variables[i1][j][k])
                    and_clause.append(
                        Implies(self.variables[i][j][k], Not(Or(not_or_clause))))
        self.rule3 = And(and_clause)

        # Encoding for rule 4: If cell has X then no other cell in same 3x3 subgrid has X
        and_clause = []
        for box_row in range(3):
            for box_col in range(3):
                for i in range(3):
                    for j in range(3):
                        for k in range(9):
                            not_or_clause = []
                            for x in range(3):
                                for y in range(3):
                                    if x != i or y != j:
                                        not_or_clause.append(
                                            self.variables[box_row * 3 + x][box_col * 3 + y][k])
                            and_clause.append(
                                Implies(self.variables[box_row * 3 + i][box_col * 3 + j][k], Not(Or(not_or_clause))))
        self.rule4 = And(and_clause)
        self.solver = Solver()
        self.rules = And([self.rule1, self.rule2, self.rule3, self.rule4])
        self.solver.add(self.rules)
        pass

    def encode_puzzle(self):
        """
        Encode the initial puzzle into the solver.
        """
        # Your code here

        # Iterate over each cell in the puzzle and add the appropriate value to the solver under its corresponding variable
        for i in range(9):
            for j in range(9):
                if self.puzzle[i][j]:
                    # adjusting to match 0-indexing
                    self.solver.add(
                        self.variables[i][j][self.puzzle[i][j] - 1])

    def extract_solution(self, model):
        """
        Extract the satisfying assignment from the given model and return it as a
        9x9 list of lists.
        Args:
            model: The Z3 model containing the satisfying assignment.
        Returns:
            A 9x9 list of lists of integers representing the Sudoku solution.
        Hint:
            To access the value of a variable in the model, you can use:
            value = model.evaluate(var)
            where `var` is the Z3 variable whose value you want to retrieve.
        """
        # Your code here

        # Initiate a 9x9 list to hold the solution
        sol = [[0 for j in range(9)]for i in range(9)]

        # Extract the solution from the model assignment
        for i in range(9):
            for j in range(9):
                for k in range(9):
                    if model.evaluate(self.variables[i][j][k]):
                        sol[i][j] = k + 1

        return sol

    def solve(self):
        """
        Solve the Sudoku puzzle.

        :return: A 9x9 list of lists representing the solved Sudoku puzzle, or None if no solution exists.
        """
        self.solver = Solver()
        self.create_variables()
        self.encode_rules()
        self.encode_puzzle()

        if self.solver.check() == sat:
            model = self.solver.model()
            solution = self.extract_solution(model)
            return solution
        else:
            return None

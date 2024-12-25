import copy
import time
from typing import List, Tuple

class State:
    '''
    Un Etat décrit une grille de Sudoku NxN peuplée de nombres entiers (0 signifie une case vide).
    '''
    def __init__(self, board: List[List[int]]) -> None:
        self.board = board

    def get_empty_position(self):
        '''Retourne la position de la première case vide (0).'''
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    return i, j
        return None
    
    # Fonction heuristique h(n)
    def get_empty_position_heuristique(self, problem):
        '''Retourne la position de la case vide ayant le plus de contraintes (la moins flexible).'''
        i_max, j_max = None, None  # Initialisation à None
        min_possible_values = float('inf')  # Commencer avec un nombre très élevé
        
        for i in range(len(self.board)):
            for j in range(len(self.board[i])):
                if self.board[i][j] == 0:
                    possible_values = problem.get_possible_values(self, i, j)  # Utilise 'problem' pour obtenir les valeurs possibles
                    num_possible_values = len(possible_values)  # Nombre de valeurs possibles
                    
                    # Si cette case a moins de valeurs possibles, elle devient la meilleure option
                    if num_possible_values < min_possible_values:
                        min_possible_values = num_possible_values
                        i_max = i
                        j_max = j

        if i_max is not None or j_max is not None:
            if(i_max is None): return 0, j_max 
            elif(j_max is None): return i_max, 0 
            else: return i_max, j_max
            
        else:
            return None  # Aucune position vide

    def get_nb_empty(self) -> int:
        '''Retourne le nombre de cases vides.'''
        return sum(row.count(0) for row in self.board)

class SudokuProblem:
    '''Cette classe définit la structure du problème Sudoku'''
    
    def __init__(self, board: List[List[int]]):
        self.size = len(board)  # Grille NxN (généralement 9x9)
        self.initial_state = State(board)

    def initial_state(self) -> State:
        '''Retourne l'état initial de la grille.'''
        return self.initial_state

    def actions(self, state: State) -> List[Tuple[int, int, int]]:
        '''Retourne les actions possibles pour remplir une case vide.'''
        empty_pos = state.get_empty_position()
        if empty_pos is None:
            return []  # Aucune case vide restante
        row, col = empty_pos
        possible_values = self.get_possible_values(state, row, col)
        return [(row, col, value) for value in possible_values]
    
    def actions_heuristique(self, state: State) -> List[Tuple[int, int, int]]:
        '''Retourne les actions possibles pour remplir une case vide.'''
        empty_pos = state.get_empty_position_heuristique(self)
        if empty_pos is None:
            return []  # Aucune case vide restante
        row, col = empty_pos
        possible_values = self.get_possible_values(state, row, col)
        return [(row, col, value) for value in possible_values]

    def succ(self, state: State, action: Tuple[int, int, int]) -> State:
        '''Retourne l'état résultant de l'application d'une action.'''
        row, col, value = action
        new_board = [row.copy() for row in state.board]  # Copie profonde de la grille
        new_board[row][col] = value
        return State(new_board)

    def goal_test(self, state: State) -> bool:
        '''Vérifie si la grille est une solution complète et valide.'''
        return state.get_nb_empty() == 0 and self.is_valid(state)

    def action_cost(self, action: Tuple[int, int, int]) -> int:
        '''Retourne le coût d'une action (1 dans ce cas).'''
        return 1

    def is_valid(self, state: State) -> bool:
        '''Vérifie si l'état actuel est valide (pas de conflits de chiffres).'''
        return (self.is_valid_rows(state) and
                self.is_valid_cols(state) and
                self.is_valid_subgrids(state))

    def is_valid_rows(self, state: State) -> bool:
        '''Vérifie la validité des lignes (pas de doublons).'''
        for row in state.board:
            if len(set(row)) != len(row):  # Contient des doublons
                return False
        return True

    def is_valid_cols(self, state: State) -> bool:
        '''Vérifie la validité des colonnes (pas de doublons).'''
        for col in range(self.size):
            column = [state.board[row][col] for row in range(self.size)]
            if len(set(column)) != len(column):  # Contient des doublons
                return False
        return True

    def is_valid_subgrids(self, state: State) -> bool:
        '''Vérifie la validité des sous-grilles 3x3.'''
        subgrid_size = int(self.size ** 0.5)
        for row in range(0, self.size, subgrid_size):
            for col in range(0, self.size, subgrid_size):
                subgrid = []
                for r in range(row, row + subgrid_size):
                    for c in range(col, col + subgrid_size):
                        subgrid.append(state.board[r][c])
                if len(set(subgrid)) != len(subgrid):  # Contient des doublons
                    return False
        return True

    def get_possible_values(self, state: State, row: int, col: int) -> List[int]:
        '''Retourne une liste de numéros valides pouvant être placés à (row, col).'''
        used = set(state.board[row])  # Nombres déjà dans la ligne
        used.update(state.board[i][col] for i in range(self.size))  # Nombres déjà dans la colonne

        # Nombres déjà dans la sous-grille 3x3
        subgrid_size = int(self.size ** 0.5)
        subgrid_row = (row // subgrid_size) * subgrid_size
        subgrid_col = (col // subgrid_size) * subgrid_size
        for r in range(subgrid_row, subgrid_row + subgrid_size):
            for c in range(subgrid_col, subgrid_col + subgrid_size):
                used.add(state.board[r][c])

        return [i for i in range(1, self.size + 1) if i not in used]


class Node:
    '''Un noeud dans l'arbre de recherche, représentant un état et son parent.'''

    def __init__(self, state: State, parent=None, action=None, path_cost=0):
        '''Crée un noeud dans l'arbre de recherche, dérivé d'un parent par une action.'''
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        '''Représentation sous forme de chaîne de caractères du noeud'''
        return "<Node {}>\n".format(self.state)

    def child_node(self, problem: SudokuProblem, action: Tuple[int, int, int]) -> "Node":
        '''Retourne un noeud enfant obtenu en appliquant une action donnée.'''
        return Node(problem.succ(self.state, action), self, action)

    def expand(self, problem: SudokuProblem) -> List["Node"]:
        '''Retourne la liste des noeuds accessibles en un seul coup depuis ce noeud.'''
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    # Appel a la fonction heuristique h(n)
    def expand_heuristique(self, problem: SudokuProblem) -> List["Node"]:
        '''Retourne la liste des noeuds accessibles en un seul coup depuis ce noeud.'''
        return [self.child_node(problem, action) for action in problem.actions_heuristique(self.state)]

    def solution(self) -> List[Tuple[int, int, int]]:
        '''Retourne la séquence d'actions pour aller du racine à ce noeud.'''
        return [node.action for node in self.path()[1:]]

    def path(self) -> List["Node"]:
        '''Retourne le chemin du racine à ce noeud.'''
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


# Algorithmes de recherche non informée (DFS et BFS)
def depth_first_tree_search(problem: SudokuProblem):
    '''Effectue une recherche en profondeur pour résoudre le Sudoku.'''
    start = Node(problem.initial_state)  # Utilise l'état, pas la méthode
    if problem.goal_test(start.state):
        return start
    
    visited_count = 0  # Compteur de noeuds visités

    frontiere = [start]
    while frontiere:
        node = frontiere.pop()
        visited_count += 1
        if problem.goal_test(node.state):
            return node,visited_count
        frontiere.extend(node.expand(problem))
    return None,visited_count

def greedy_best_first_search(problem: SudokuProblem):
    '''f(n)=GBFS est une fonction de recherche qui cherche les noeuds ayant le plus petit
    nombre de candidats afin d'explorer le moins de noeuds possible pour arriver à la solution'''
    start = Node(problem.initial_state)  # Utilise l'état, pas la méthode
    if problem.goal_test(start.state):
        return start
    
    visited_count = 0  # Compteur de noeuds visités

    frontiere = [start]
    while frontiere:
        node = frontiere.pop()
        
        if problem.goal_test(node.state):
            return node,visited_count

        visited_count += 1

        # Ajouter le noeud fils à la frontière qui a moins de fils en appelant la fonction heuristique
        frontiere.extend(node.expand_heuristique(problem))

    return None,visited_count

# Résolution

N = 9  # Sudoku standard 9x9
sudoku_board = [
    [5, 0, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]



sudoku_problem = SudokuProblem(sudoku_board)
solution_node1,visited_nodes1 = depth_first_tree_search(sudoku_problem)
solution_node2,visited_nodes2 = greedy_best_first_search(sudoku_problem)

if solution_node1:
    print("Solution de la méthode parcours en profondeur :")
    for row in solution_node1.state.board:
        print(row)
else:
    print("Aucune solution trouvée par la méthode parcours en profondeur")
print(f"Nombre de nœuds visités : {visited_nodes1}")

if solution_node2:
    print("Solution de la méthode greedy_best_first_search :")
    for row in solution_node2.state.board:
        print(row)
else:
    print("Aucune solution trouvée par la méthode de greedy_best_first_search")
print(f"Nombre de nœuds visités : {visited_nodes2}")

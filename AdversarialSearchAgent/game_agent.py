"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from operator import itemgetter

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    if len(opp_moves) == 1 and (opp_moves[0] in my_moves):
        return float(1000.)
    if len(my_moves) == 1 and (my_moves[0] in opp_moves):
        return float(-1000.)
    if len(my_moves) <= len(opp_moves):
        return (-100.+(10.*len(my_moves)))*float(len(opp_moves)-len(my_moves))
    elif len(my_moves) > len(opp_moves):
        return (100.-(10.*len(opp_moves)))*float(len(my_moves)-len(opp_moves))

def custom_score_2(game, player):
    """Calcu	late the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    my_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    if len(opp_moves) == 1 and (opp_moves[0] in my_moves):
        return float(1000.)
    if len(my_moves) == 1 and (my_moves[0] in opp_moves):
        return float(-1000.)
    if game.move_count > game.height*2:
        return float(10.*(len(my_moves)-len(opp_moves)))
    loc = game.get_player_location(player)
    low = 2
    x_high = game.height - 3
    y_high = game.width - 3
    if low <= loc[0] <= x_high:
        if low <= loc[1] <= y_high:
            move_score = float(7)
        elif loc[1] == low-1 or loc[1] == y_high+1:
            move_score = float(5)
        elif loc[1] == low-2 or loc[1] == y_high+2:
            move_score = float(3)
    elif loc[0] == low-1 or loc[0] == x_high+1:
        if low <= loc[1] <= y_high:
            move_score = float(5)
        elif loc[1] == low-1 or loc[1] == y_high+1:
            move_score = float(3)
        elif loc[1] == low-2 or loc[1] == y_high+2:
            move_score = float(2)
    elif loc[0] == low-2 or loc[0] == x_high+2:
        if low <= loc[1] <= y_high:
            move_score = float(3)
        elif loc[1] == low-1 or loc[1] == y_high+1:
            move_score = float(2)
        elif loc[1] == low-2 or loc[1] == y_high+2:
            move_score = float(1)
    return float(move_score*10.)

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # TODO: finish this function!
    mCount = game.move_count
    my_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(game.get_opponent(player))
    if len(opp_moves) == 1 and (opp_moves[0] in my_moves):
        return float(1000.)
    if len(my_moves) == 1 and (my_moves[0] in opp_moves):
        return float(-1000.)
    if mCount <= game.height:
        return float(10.*len(my_moves))
    else:       
        if len(opp_moves) < len(my_moves) and len(opp_moves) == 2:
            float(100.*(len(my_moves) - len(opp_moves))/len(opp_moves))
        if bool(set(my_moves).intersection(opp_moves)):
            return float(10.*len(my_moves))
        return float(100.-(10.*len(my_moves)))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        moveList = game.get_legal_moves()

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            if len(moveList)>0:
                return moveList[0]  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1,-1)
        try:
            move_val_list = [(move,self.min_value(game.forecast_move(move),depth)) for move in legal_moves]
            return max(move_val_list,key=itemgetter(1))[0]
        except:
            return legal_moves[0]
        return best_move

    def min_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moveList = game.get_legal_moves()
        if not moveList:
            return float("inf")
        min_val = float("inf")
        if depth > 1:
            for move in moveList:
                min_val = min(min_val,self.max_value(game.forecast_move(move),depth-1))
            return min_val
        return self.score(game,self)

    def max_value(self,game,depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        moveList = game.get_legal_moves()
        if not moveList:
            return float("-inf")
        max_val = float("-inf")
        if depth > 1:
            for move in moveList:
                max_val = max(max_val,self.min_value(game.forecast_move(move),depth-1))
            return max_val
        return self.score(game,self)

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        moveList = game.get_legal_moves()
        if not moveList:
            return (-1,-1)
        depth = 1
        best_move = moveList[0]
        while True:
            try:
                move = self.alphabeta(game,depth)
                depth += 1
                if move != 'None':
                    best_move = move
            except:
                return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moveList = game.get_legal_moves()
        max_val_move = moveList[0]
        for move in moveList:
            val = self.min_value(game.forecast_move(move),depth,alpha,beta)
            if alpha < val:
                alpha = val
                max_val_move = move
        return max_val_move

    def min_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moveList = game.get_legal_moves()
        if not moveList:
            return float("inf")
        min_val = float("inf")
        if depth > 1:
            for move in moveList:
                min_val = min(min_val,self.max_value(game.forecast_move(move),depth-1,alpha,beta))
                if min_val <= alpha:
                    return min_val
                beta = min(beta,min_val)
            return min_val
        return self.score(game,self)

    def max_value(self,game,depth,alpha,beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        moveList = game.get_legal_moves()
        if not moveList:
            return float("-inf")
        max_val = float("-inf")
        if depth > 1:
            for move in moveList:
                max_val = max(max_val,self.min_value(game.forecast_move(move),depth-1,alpha,beta))
                if beta <= max_val:
                    return max_val
                alpha = max(alpha,max_val)
            return max_val
        return self.score(game,self)
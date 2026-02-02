from agents.simple import RandomAgent, FirstLegalActionAgent
import env.connect4 as c4

class CliAgent:
    def get_action(self, state: c4.GameState):
        move = None
        while move is None:
            try:
                number = int(input("move? "))
                if c4.is_valid_move(state, number):
                    move = number
                else:
                    print("invalid move")
            except Exception as e:
                if e is KeyboardInterrupt:
                    break
                else:
                    print("what?")
        return move


def main():
    human = CliAgent()
    opp = FirstLegalActionAgent()
    state = c4.new_game()
    turn = state.current_player
    while True:
        print("----")
        print(c4.to_string(state))
        player = human if turn == c4.PLAYER1 else opp
        move = player.get_action(state)
        state = c4.make_move(state, move)
        turn = c4.other_player(turn)
        if state.done:
            print("----")
            print(c4.to_string(state))
            print("game over. winner = ", "draw" if state.winner is None else "X" if state.winner == c4.PLAYER1 else "O")
            state = c4.new_game()
            turn = state.current_player


if __name__ == "__main__":
    main()

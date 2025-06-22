#!/usr/bin/env python3

import torch
from lib.env import ChessGame, WHITE, BLACK
from lib.agent import RandomAgent
from lib.greedy_agent import GreedyChessAgent


def test_greedy_agent():
    """Test basic functionality of the GreedyChessAgent"""
    print("Testing GreedyChessAgent...")
    
    # Create agents
    greedy_agent = GreedyChessAgent(WHITE)
    random_agent = RandomAgent(BLACK)
    
    # Create game
    game = ChessGame()
    
    # Test basic functionality
    print(f"Initial state shape: {game.state_np().shape}")
    print(f"Initial turn: {game.turn}")
    
    # Test that agent can make a move
    if game.turn == greedy_agent.player:
        move = greedy_agent.get_action(game)
        print(f"Greedy agent chose move: {move}")
        game.step(move)
    
    # Test training step (should not crash)
    state = game.state_np()
    greedy_agent.add_experience(state, 0.1)
    
    # Add more experiences for training
    for _ in range(50):
        greedy_agent.add_experience(state, 0.5)
    
    loss = greedy_agent.train_step()
    if loss is not None:
        print(f"Training loss: {loss:.4f}")
    else:
        print("Not enough experiences for training yet")
    
    print("✓ GreedyChessAgent basic test passed!")


def test_short_game():
    """Test a short game between greedy and random agent"""
    print("\nTesting short game...")
    
    greedy_agent = GreedyChessAgent(WHITE)
    random_agent = RandomAgent(BLACK)
    
    game = ChessGame()
    moves_played = 0
    max_moves = 10
    
    while not game.is_game_over() and moves_played < max_moves:
        if game.turn == WHITE:
            move = greedy_agent.get_action(game)
            agent_name = "Greedy (WHITE)"
        else:
            move = random_agent.get_action(game)
            agent_name = "Random (BLACK)"
        
        print(f"Move {moves_played + 1}: {agent_name} plays {move}")
        game.step(move)
        moves_played += 1
    
    print(f"Game ended after {moves_played} moves")
    if game.is_game_over():
        winner = game.winner()
        if winner:
            winner_name = "WHITE" if winner == WHITE else "BLACK"
            print(f"Winner: {winner_name}")
        else:
            print("Game ended in draw")
    
    print("✓ Short game test completed!")


if __name__ == "__main__":
    test_greedy_agent()
    test_short_game()
    print("\nAll tests completed successfully!")
# Easy21
Monte Carlo Agent for the card game Easy21 from David Silver's DeepMind lectures

This script uses the NumPy library to work with Q-value data about the environment, as well as Matplotlib to visualize performance.

Performance seems to level off at a ~50% win rate after about 1 million games of training, although the win rate is only 1.2% lower after 100,000 games. Performance increases the most over the first 1000 games, reaching ~42.4% as the agent determines the majority of Q-values through experience.

Rules (from the Assignment handout):

The game is played with an infinite deck of cards (i.e. cards are sampled with replacement)

• Each draw from the deck results in a value between 1 and 10 (uniformly distributed). The player has a 2/3 chance of picking a black card.

• There are no aces or picture (face) cards in this game

• At the start of the game both the player and the dealer draw one black card (fully observed)

• Each turn the player may either stick or hit

• If the player hits then she draws another card from the deck

• If the player sticks she receives no further cards

• The values of the player’s cards are added (black cards) or subtracted (red cards)

• If the player’s sum exceeds 21, or becomes less than 1, then she “goes bust” and loses the game (reward -1)

• If the player sticks then the dealer starts taking turns. The dealer always sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome – win (reward +1), lose (reward -1), or draw (reward 0) – is the player with the largest total wins.

# Easy21
Monte Carlo Agent for the card game Easy21 from David Silver's DeepMind lectures

This script uses the NumPy library to work with Q-value data about the environment, as well as Matplotlib to visualize performance.

Performance seems to level off at a ~50% win rate after about 1 million games of training, although the win rate is only 1.2% lower after 100,000 games. Performance increases the most over the first 1000 games, reaching ~42.4% as the agent determines the majority of Q-values through experience.

## Surface Plots:

![Q-function after 100 games](https://github.com/dhruv-sirohi/Easy21/blob/main/Surface%20Plots/Q_surface_100.png)

After 100 games, the Q value for a few states has been initialized to 1, because the agent happened to win in the few games it played there. This value is often not representative of the Q value after further training, as the agent plays more games from each state and determines a more accurate Q-value.

![Alt text](https://github.com/dhruv-sirohi/Easy21/blob/main/Surface%20Plots/Q_surface_1000.png)

After the agent has played 1000 games, we see that most states have now been visited, but since most of them are initialized to 1 or -1, they likely have not been visited many times.

10,000 games:

![Alt text](https://github.com/dhruv-sirohi/Easy21/blob/main/Surface%20Plots/Q_surface_10000.png)

After 10,000 games we begin to see a pattern emerge, as the Q value for favorable states tends towards 1 as the player's total reaches 21. There's a large spike for the states in which the player has a total of ~16+. This is likely because the dealer only hits until their total is 17. There also seems to be a trend downwards as the dealer's initial card value increases. This may be representative of them having a lesser chance of going bust by getting a large red card, while also being able to get close to blackjack with only 1-2 cards.

![Alt text](https://github.com/dhruv-sirohi/Easy21/blob/main/Surface%20Plots/Q_surface_1000000.png)

This pattern is further consolidated after 1 million games, with a smooth trend emerging both across the states in which the player has different totals, as well as the states in which the dealer has different totals. 


## Rules 

These rules can be found in the Easy21 Assignment Handout available at: https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf

The game is played with an infinite deck of cards (i.e. cards are sampled with replacement)

• Each draw from the deck results in a value between 1 and 10 (uniformly distributed). The player has a 2/3 chance of picking a black card.

• There are no aces or picture (face) cards in this game

• At the start of the game both the player and the dealer draw one black card (fully observed)

• Each turn the player may either stick or hit

• If the player hits then they draw another card from the deck

• If the player sticks they receive no further cards

• The values of the player’s cards are added (black cards) or subtracted (red cards)

• If the player’s sum exceeds 21, or becomes less than 1, then they “go bust” and loses the game (reward -1)

• If the player sticks then the dealer starts taking turns. The dealer always sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome – win (reward +1), lose (reward -1), or draw (reward 0) – is the player with the largest total wins.

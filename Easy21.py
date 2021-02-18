from mpl_toolkits.mplot3d import Axes3D

import random
import numpy as np
import matplotlib.pyplot as plt
#Some debug/gameplay statements have been made comments, but left in
#This code can be modified to be interactive (user plays against Dealer), in which case the print statements are useful

#Defines dealer class
#The only dealer action is updating their total after picking up a card
class Dealer:
    
    def __init__(self,total):
        self.total = total

    def update(self, new_total):
        self.total = self.total + new_total
        #print("Dealer total is now: ", self.total)

#Defines Player class
#The player can either stick or hit
#Past states and rewards are stored for training
#A Monte Carlo algorithm is used for agent training
        
class Player:
    def __init__(self,total, Q_matrix, past_states):
        self.total = total
        self.Q_matrix = Q_matrix
        self.past_states = past_states

    def update(self, new_total):
        self.total = self.total + new_total
        #print("Player total is now: ", self.total)

    def add_state(self, new_state):
        self.past_states.append(new_state)
        

#card dealing function
def deal_card():
    one_in_three = random.randint(0,2)
    
    if(one_in_three > 1):
        total = random.randint(1,10)
        
        return 0 - total
    elif (one_in_three < 2):
        total = random.randint(1,10)
        
        return total

#custom black card dealing function for initial move
def deal_black():
    total = random.randint(1,10)
    
    return total

#function where dealer and agent make their moves
#returns: (change_turn,terminal_state)
def state_update(dealer,player,who_acts):
    #who_acts = 0 if player, 1 if dealer
    
    #hard coded logic for dealer
    if (who_acts):
        #dealer acts
        if(dealer.total < 0):
            ##print("Dealer is Bust!")
            return (True,True)
            
        if dealer.total < 17:
            #print("Dealer Hits")
            dealer.update(deal_card())
            #print("dealer has: ",dealer.total)
            return (False,False)
            
        else:
            return (True,True)            
    else:
        #the player acts after the dealer has been dealt an initial black card, but nothing else
        #this means the dealer's values range from 1 to 10
        #as such, it is possible to index all possible states by:
        #1. representing the player's total as the integer division of the state index by 10
        #2. representing the dealer's total as the modulo of the state index by 10, minus 1
        #This isn't aliased because of the bounds on the dealer's values
        #Example: if s = 21, then the dealer's total is 2 and the player's total is also 2
        
        state_index = int(dealer.total - 1 + 10 * (player.total))

        q = player.Q_matrix
        
        #Determine the q values for sticking/hitting
        q_stick = q[state_index][0][2]
        q_hit = q[state_index][1][2]

        #Run epsilon greedy algorithm
        epsilon = 100 / (100 + q[state_index][0][3])
        q[state_index][0][3] += 1
        prob = random.randint(0,100)

        #Performs action based on q-values, or randomly, because of the epsilon greedy strategy 
        if (prob < 100 * epsilon):
            if(random.randint(0,1) == 0):
                    #print("stick")
                    player.add_state([state_index,0])
                    return (True,False)

            else:
                    #print("hit")
                    player.add_state([state_index,1])
                    player.update(deal_card())
                    
        else:
            if (q_hit > q_stick):
                #print("hit")
                player.add_state([state_index,1])
                player.update(deal_card())
                
                
            elif (q_stick > q_hit):
                #print("stick")
                player.add_state([state_index,0])
                return (True,False)
                
            elif (q_hit == q_stick):
                if(random.randint(0,1) == 0):
                    #print("stick")
                    player.add_state([state_index,0])
                    return (True,False)

                else:
                    #print("hit")
                    player.add_state([state_index,1])
                    player.update(deal_card())
            
        if(player.total > 21 or player.total < 0 or player.total == 21):
            return (True,True)

        else:
            return (False,False)

#function which determines whether a state is terminal
#updates Q-values by running the Monte Carlo update on the full episode
def terminal_state(dealer,player):
    #r is reward
    r = 0
    #determines whether player has bust
    #reward of +1 for winning, -1 for busting, 0 for a draw
    if(player.total > 21 or player.total < 0):
        #print("Player Bust!")
        for i in player.past_states:
            #s is state index
            s = i[0]
            #a is action index (0 = stick, 1 = hit)
            a = i[1]
            player.Q_matrix[s][a][4] += 1
            #in the Easy21 document, alpha is chosen to be 1/N
            alpha = 1/(player.Q_matrix[s][a][4])
            q_val = player.Q_matrix[s][a][2]
            #update step
            player.Q_matrix[s][a][2] += float((-1-q_val)*alpha)
        r = -1
    #determines whether dealer has bust
    elif(dealer.total > 21 or dealer.total < 0):
        #print("Dealer Bust!")
        for i in player.past_states:
            s = i[0]
            a = i[1]
            
            player.Q_matrix[s][a][4] += 1
            alpha = player.Q_matrix[s][a][4]
            q_val = player.Q_matrix[s][a][2]

            #update step
            player.Q_matrix[s][a][2] += float((1-q_val)/alpha)
        r = 1
        
    else:
        #if neither player has bust, then determines which player has highest total
        if dealer.total == 21:
            #print("Dealer Wins!")
            for i in player.past_states:
                s = i[0]
                a = i[1]

                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]

                #update step
                player.Q_matrix[s][a][2] += float((-1-q_val)/alpha)
            r = -1
                
        if player.total == 21:
            #print("Player Wins")
            for i in player.past_states:
                s = i[0]
                a = i[1]
                
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]

                #update step
                player.Q_matrix[s][a][2] += float((1-q_val)/alpha)
            r = 1

        if dealer.total > player.total:
            #print("Dealer Wins!")
            for i in player.past_states:
                s = i[0]
                a = i[1]

                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]

                #update step
                player.Q_matrix[s][a][2] += float((-1-q_val)/alpha)
            r = -1
                
        elif dealer.total < player.total:
            #print("Player Wins")
            for i in player.past_states:
                s = i[0]
                a = i[1]
                
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]

                #update step
                player.Q_matrix[s][a][2] += float((1-q_val)/alpha)
            r = 1
                
        else:
            #if neither the agent or the dealer have bust or have 21, and both have the same total, the result is a draw
            for i in player.past_states:
                s = i[0]
                a = i[1]
                
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]
                
                player.Q_matrix[s][a][2] += float((0-q_val)/alpha)
            r = 0

    #resets array of past states and card totals
    player.past_states = []
    player.total = 0
    dealer.total = 0
    return r
    #print("---------------Game Over-----------------")

def start():
    dealer = Dealer(0)
    q_mat = []
    #each element has: (state, action, q val, number of time's the state has been visited, number of times a given action has been taken at that state)
    #action = 0 : stick
    #action = 1 : hit
    for i in range(21*10):
        q_mat.append([[i,0,0,0,0],[i,1,0,0,0]])
    #initializes array in which past states will be stored for monte carlo update
    past_states = []
    #initializes player
    player = Player(0,q_mat,past_states)


    #initializes text file in which results will be stored
    f = open("easy21_results.txt", "a")
    f.write("------new game------")
    f.write("\n")
                
    avg = 0
    avg_since_plot = 0

    #array in which score values will be stored
    score_array = []
    
    n_played = 1
    n_since_last_plot = 0
    q_val = []

    n_plots = 1
    #training loop
    while(n_played <= 1000000):
        
        #game start
        dealer.update(deal_black())
        player.update(deal_black())

        change_turns = False
        terminate = False
        while(change_turns == False):
            (change_turns,terminate) = state_update(dealer,player,0)
        while(terminate == False):
            (change_turns,terminate) = state_update(dealer,player,1)

        #r is reward    
        r = terminal_state(dealer,player)

        #updates overall average score, and average score since last plot
        avg = float((avg * n_played + r) / (n_played+1))
        avg_since_plot = float((avg_since_plot * n_since_last_plot + r) / (n_since_last_plot+1))
        
        n_since_last_plot = n_since_last_plot+1

        score_array.append(r)

        #plots q values after increaing intervals
        if(n_played == 10**n_plots):
                q_val = []
                X = []
                #x represents player's total
                Y = []
                #y represents dealer's total
                Z = []
                #z array contains  maximum q-value for a given state
                for x in range(21):
                    q_val.append([])
                    for y in range(10):
                        #s represents state index
                        s = y + x*10
                        
                        q_stick = q_mat[s][0][2]
                        q_hit = q_mat[s][1][2]

                        q_max = max(q_stick,q_hit)
                        q_val[x].append((format(q_max, '.2f')))
                        
                        X.append(x+1)
                        Y.append(y+1)
                        Z.append(q_max)

                #sets up 3D figure
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(np.array(Y),np.array(X),np.array(Z))
                

                #updating text file
                text_to_write = "average score between " + str(10**(n_plots-1)) + " and " + str(10**(n_plots)) + " games: "
                f.write(text_to_write)
                f.write(str(avg_since_plot))
                f.write("\n")
                
                n_since_last_plot = 0
                avg_since_plot = 0
                n_plots += 1
                
        n_played = n_played+1
    
    #updates text file with overall stats after training has ended
    f.write("average score over full training: ")
    f.write(str(avg))
    f.write("\n")
    f.write("list of scores:")
    f.write("\n")
    for score in score_array:
        f.write(str(score))
        f.write("\n")
    f.close()

    plt.show()
    
if __name__ == "__main__":       
    start()


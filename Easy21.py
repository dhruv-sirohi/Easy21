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
        #print("Player total is now: ", self.total)

#card dealing function
def deal_card():
    one_in_three = random.randint(0,2)
    ##print("random chance: ", one_in_three)
    if(one_in_three > 1):
        total = random.randint(1,10)
        #print("picked up: -", total)
        return 0 - total
    elif (one_in_three < 2):
        total = random.randint(1,10)
        #print("picked up: ", total)
        return total

#custom black card dealing function for initial move
def deal_black():
    total = random.randint(1,10)
    ##print("picked up: ", total)
    return total

#updates state based on new cards dealt
def state_update(dealer,player,who_acts):
    #who_acts = 0 if player, 1 if dealer
    #state = [player total, dealer card]
    
    #hard coded logic for dealer
    if (who_acts):
        #dealer acts
        if(dealer.total < 0):
            ##print("Bust!")
            return (True,True)
            
        if dealer.total < 17:
            #dealer hits
            #print("dealer hits")
            dealer.update(deal_card())
            ##print("dealer has: ",dealer.total)
            return (False,False)
            
        else:
            return (True,True)
            
            
    else:
        #player acts
        
        #10's place digit indexes palyer 
        s = dealer.total - 1 + 10 * (player.total)
        q = player.Q_matrix
        #Determine the q values for sticking/hitting
        #print("state: ",s)
        #print("dealer: ",dealer.total)
        #print("player: ",player.total)
        q_stick = q[s][0][2]
        #print("state: ", s)
        #print('q_stick = ', q_stick)
        q_hit = q[s][1][2]
        #print('q_hit = ', q_hit)
        #Run epsilon greedy
        epsilon = 100 / (100 + q[s][0][3])
        q[s][0][3] += 1
        prob = random.randint(0,100)
        #act
        if (prob < 100 * epsilon):
            if(random.randint(0,1) == 0):
                    #print("stick")
                    player.add_state([s,0])
                    return (True,False)

            else:
                    #print("hit")
                    player.add_state([s,1])
                    player.update(deal_card())
                    
        else:
            if (q_hit > q_stick):
                #print("hit")
                player.add_state([s,1])
                player.update(deal_card())
                
                
            elif (q_stick > q_hit):
                #print("stick")
                player.add_state([s,0])
                return (True,False)
                
            elif (q_hit == q_stick):
                if(random.randint(0,1) == 0):
                    #print("stick")
                    player.add_state([s,0])
                    return (True,False)

                else:
                    #print("hit")
                    player.add_state([s,1])
                    player.update(deal_card())
            
        if(player.total > 21 or player.total < 0 or player.total == 21):
            return (True,True)

        return (False,False)
        
        
            

def terminal_state(dealer,player):
    #do MC update
    r = 0
    if(player.total > 21 or player.total < 0):
        #print("Player Bust!")
        for i in player.past_states:
            s = i[0]
            a = i[1]
            player.Q_matrix[s][a][4] += 1
            alpha = player.Q_matrix[s][a][4]
            q_val = player.Q_matrix[s][a][2]
            player.Q_matrix[s][a][2] += float((-1-q_val)/alpha)
    elif(dealer.total > 21 or dealer.total < 0):
        #print("Dealer Bust!")
        for i in player.past_states:
            s = i[0]
            a = i[1]
            player.Q_matrix[s][a][4] += 1
            alpha = player.Q_matrix[s][a][4]
            q_val = player.Q_matrix[s][a][2]
            player.Q_matrix[s][a][2] += float((1-q_val)/alpha)
            r = 1
        
    else:
        if dealer.total > player.total:
            #print("Dealer Wins!")
            for i in player.past_states:
                s = i[0]
                a = i[1]
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]
                player.Q_matrix[s][a][2] += float((-1-q_val)/alpha)
        if dealer.total < player.total:
            #print("Player Wins")
            for i in player.past_states:
                s = i[0]
                a = i[1]
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]
                player.Q_matrix[s][a][2] += float((1-q_val)/alpha)
                r = 1
        else:
            for i in player.past_states:
                s = i[0]
                a = i[1]
                player.Q_matrix[s][a][4] += 1
                alpha = player.Q_matrix[s][a][4]
                q_val = player.Q_matrix[s][a][2]
                player.Q_matrix[s][a][2] += float((0-q_val)/alpha)
                r = 0
            
    player.past_states = []
    player.total = 0
    dealer.total = 0
    return r
    #print("---------------Game Over-----------------")

def start():

    
    dealer = Dealer(0)
    q_mat = []
    #q has: state, action, q val, n_s, n_a
    for i in range(21*10):
        q_mat.append([[i,0,0,0,0],[i,1,0,0,0]])
    past_states = []
    player = Player(0,q_mat,past_states)
    i = 1
    avg = 0
    j = 0
    q_val = []

    pwr = 0
    while(i <= 1000000):
        
        
        dealer.update(deal_black())
        player.update(deal_black())
        change_turns = False
        terminate = False
        while(change_turns != True):
            (change_turns,terminate) = state_update(dealer,player,0)
        while(terminate != True):
            (change_turns,terminate) = state_update(dealer,player,1)
        r = terminal_state(dealer,player)
        
        avg = float((avg * j + r) / (j+1))
        j = j+1
        
        if(i % 10**pwr == 0):
                
                q_val = []
                X = []
                Y = []
                Z = []
                for x in range(21):
                    q_val.append([])
                    for y in range(10):
                        s = y + x*10
                        q_stick = q_mat[s][0][2]
                        q_hit = q_mat[s][1][2]
                        q_max = max(q_stick,q_hit)
                        q_val[x].append((format(q_max, '.2f')))
                        X.append(x+1)
                        Y.append(y+1)
                        Z.append(q_max)
                        #y is equal to shown card
                
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(np.array(Y),np.array(X),np.array(Z))
                pwr += 1
                f = open("easy21_results.txt", "a")
                f.write(str(avg))
                f.write("\n")
                f.close()
                j = 0
                avg = 0
                
        i = i+1
    
    
    plt.show()
    
if __name__ == "__main__":       
    start()


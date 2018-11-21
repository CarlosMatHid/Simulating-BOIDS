from visual import *

import time
import sys
import numpy as np
import math

#PARAMETERS
N_boids = 50
MaxVel_boids = 1

N_predators = 0 #No predators 0
MaxVel_predators = 1

Size_box = 100  #Box side from 0 to size box
Center_box = np.ones(3)*Size_box/2
Vel_distribution = 0.001

dt = 1
framespersecond = 24
timesteps = 10

plot_order_parameter = np.array([])



#INITIAL CONDITIONS BOIDS:
####SQUARE LATTICE
##Pos_boids = np.zeros((N_boids,3))
##sqrtN = np.ceil(np.sqrt(N_boids))
##for i in range(N_boids): 
##    Pos_boids[i][0] = (i%N_boids**(1/3) + 0.5)/N_boids**(1/3)*Size_box
##    Pos_boids[i][1] = (i//N_boids**(1/3) + 0.5)/N_boids**(1/3)*Size_box - i//N_boids**(2/3)*Size_box
##    Pos_boids[i][2] = (i//N_boids**(2/3) + 0.5)/N_boids**(1/3)*Size_box
##Vel_boids = np.zeros((N_boids,3))
Pos_boids = (np.random.rand(N_boids,3))*Size_box #Around the center
Vel_boids = (np.random.randn(N_boids,3))

#INITIAL CONDITIONS PREDATORS:
Pos_predators = np.random.rand(N_predators,3)*Size_box+Size_box
Vel_predators = np.array([1,0,0])+(np.random.randn(N_predators,3)*2-1)*Vel_distribution



#DRAWING

# Scene
scene = display(title="boids", width=1000, height=1000,
                range=2*Size_box, forward=(-1,-1,-1))
xaxis = curve(pos=[(0,0,0), (Size_box*10,0,0)], color=(0.5,0.5,0.5))
yaxis = curve(pos=[(0,0,0), (0,Size_box*10,0)], color=(0.5,0.5,0.5))
zaxis = curve(pos=[(0,0,0), (0,0,Size_box*10)], color=(0.5,0.5,0.5))
# Boids
boids = []
for i in range(N_boids-1):
    boids = boids+[sphere(pos=(Pos_boids[i][0],Pos_boids[i][1],Pos_boids[i][2]), radius=1, make_trail=False, interval=1)]
boids = boids+[sphere(pos=(Pos_boids[N_boids-1][0],Pos_boids[N_boids-1][1],Pos_boids[N_boids-1][2]), radius=1, color=color.green, make_trail=True, interval=1)]
# predators
predators = []
for i in range(N_predators):
    predators = predators+[sphere(pos=(Pos_predators[i][0],Pos_predators[i][1],Pos_predators[i][2]), radius=1,color=color.red, make_trail=False, interval=1)]


##Vel_boids = np.zeros((N_boids,3))
    
#STEP FOR THE BOIDS    
def boids_step(N_predators, N_boids, Pos_boids, Vel_boids, Pos_predators, MaxVel_boids, Size_box):
    
    #PARAMETERS
    radius_obstacle = 5
    
    alignment_radius = 5
    repulsion_radius = 10 
    cohesion_radius = 60
    predator_radius = 25
    
    #WITHOUT OBSTACLE:
    #ROTATION AND MIGRATION: 0.1, 0.1,  2, 10, 3, 0   + RADIUS 5,  10, 60, 25 
    #MIGRATION:              0.1, 0.1,  2, 10, 3, 0   + RADIUS 20, 10, 60, 25
    #JAMMED:                 0.1, 0.1,  2,  5, 3, 0   + RADIUS 5,  10, 60, 25   PONER VELOCIDADES A CERO!


    
    Weight_random = 0.1
    Weight_cohesion  = 0.1
    Weight_repulsion = 2
    Weight_alignment = 5
    Weight_scape = 3
    Weight_center = 0
    Weight_avoid = 10
    
    #FOR EACH BOID
    for i in range(N_boids):  
        vel_alignment = np.zeros(3)   
        vel_repulsion = np.zeros(3)
        vel_scape = np.zeros(3)
        alignment_neighbours = 0
        vel_cohesion = np.zeros(3)
        vel_avoid = 0
        
        #FOR EACH OTHER BOID
        for j in range(N_boids):
            distance = Pos_boids[j]-Pos_boids[i]         
            #REPULSION
            if i != j:
                if np.linalg.norm(distance) < repulsion_radius:
                    vel_repulsion += -distance/np.linalg.norm(distance)  
            #COHESION VELOCITY
                if np.linalg.norm(distance) < cohesion_radius:
                    vel_cohesion += distance/np.linalg.norm(distance) 
            #ALIGNMENT
            if np.linalg.norm(distance) < alignment_radius:
                alignment_neighbours += 1
                vel_alignment += Vel_boids[j]
        vel_alignment = vel_alignment/alignment_neighbours
        #RANDOM VELOCITY
        vel_random = (np.random.randn(3))
        #SCAPE PREDATORS
        if N_predators != 0:
            for j in range(N_predators):
                distance = Pos_predators[j]-Pos_boids[i]
                if np.linalg.norm(distance) < predator_radius:
                    vel_scape += -distance/np.linalg.norm(distance) 
        #CENTER OF THE GRID
        center_grid = np.ones(3)*Size_box/2
        vel_center = (center_grid-Pos_boids[i])
#        #OBSTACLE
#        obstacle_distance = Pos_boids[i,0:2] - np.array([50, 0])
#        if np.linalg.norm(obstacle_distance) < radius_obstacle:
#            vel_avoid = obstacle_distance/np.linalg.norm(obstacle_distance)
#            vel_avoid = np.append(vel_avoid, 0)

        Vel_boids[i] = vel_alignment*Weight_alignment + vel_cohesion*Weight_cohesion + vel_repulsion*Weight_repulsion + vel_scape*Weight_scape +  vel_center*Weight_center + vel_random*Weight_random + vel_avoid*Weight_avoid
        if np.linalg.norm(Vel_boids[i]) > MaxVel_boids:
            Vel_boids[i] = Vel_boids[i]/np.linalg.norm(Vel_boids[i])*MaxVel_boids
        
    Pos_boids += Vel_boids*dt
    
    return Pos_boids, Vel_boids
    
    
#STEP FOR THE PREDATORS
def predators_step(N_predators, N_boids, Pos_boids, Vel_predators, Pos_predators, MaxVel_predators, timestep, timesteps):
    
    #PARAMETERS
    attack_radius = 50
    Weight_cohesion = 0.2
    Weight_attack = 20
    Weight_around = 0
    
    #FOR EACH PREDATORS
    for i in range(N_predators):
        vel_attack = np.zeros(3)
        vel_around = np.zeros(3)
            
        #ATTACK FLOCK
        for j in range(N_boids):
            distance = Pos_boids[j]-Pos_predators[i]         
            if np.linalg.norm(distance) < attack_radius:
                vel_attack += distance/np.linalg.norm(distance)
        #COHESION VELOCITY (CENTER OF THE FLOCK)
        center_flock = np.mean(Pos_boids, axis=0)
        vel_cohesion = (center_flock-Pos_predators[i])
        #MOVE AROUND THE CENTER OF THE FLOCK
        t = (timestep/float(timesteps))*4*math.pi + i*(math.pi/4.0) #WHY?Can't we put just i?
##        t = i
        vel_around[0] = (2.0+math.cos(3.0*t))*math.cos(2.0*t)
        vel_around[1] = (2.0+math.cos(3.0*t))*math.sin(2.0*t)
        vel_around[2] = math.sin(3*t)
        
        Vel_predators[i] += vel_cohesion*Weight_cohesion + vel_attack*Weight_attack + vel_around*Weight_around
        if np.linalg.norm(Vel_predators[i]) > MaxVel_predators:
            Vel_predators[i] = Vel_predators[i]/np.linalg.norm(Vel_predators[i])*MaxVel_predators       
        
    Pos_predators += Vel_predators*dt
    
    return Pos_predators, Vel_predators
   

while True:
    rate(100)
    
    #update boids
    Pos_boids, Vel_boids = boids_step(N_predators, N_boids, Pos_boids, Vel_boids, Pos_predators, MaxVel_boids, Size_box)
    for i in range(N_boids):
        boids[i].pos = Pos_boids[i]
        
    #update predators
    if N_predators != 0:
        Pos_predators, Vel_predators = predators_step(N_predators, N_boids, Pos_boids, Vel_predators, Pos_predators, MaxVel_predators, i, timesteps)
    for i in range(N_predators):
        predators[i].pos = Pos_predators[i]

    #order parameter
    order_parameter_vector = np.zeros(3) 
    order_parameter = 0    
    for i in range(N_boids):
        order_parameter_vector += Vel_boids[i]/N_boids/np.linalg.norm(Vel_boids[i])
    order_parameter = np.linalg.norm(order_parameter_vector)
    plot_order_parameter = np.append(plot_order_parameter, order_parameter)
    print(order_parameter)

        
    

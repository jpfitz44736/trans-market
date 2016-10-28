#-------------------------------------------------------------------------------
# Name:        module1
# Purpose: Simulates a transactional economy of N agents transacting in pairs and following
#           a specified rule
#
# Author:      J P Fitzsimmons
#
# Created:     10/27/2016
# Copyright:   (c) J P Fitzsimmons 2016
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

N  = 1000 # Default size of population
mu = 100. # Default mean of population's wealth
J=1

def sample(distribution, N=N, mu=mu):
    "Sample from the distribution N times, then normalize results to have mean mu."
    return normalize([distribution() for J in range(N)], mu * N)

def sample_cust(distribution,N=N,mu=mu,J=1):
    """
    Use this function for the custom distribution of wealth only
    """
    print N
    numbers=[]
    for J in range(N):
        numbers.append(distribution(J=J,N=N ))
    print "No of Agents=",len(numbers)
    return normalize(numbers,mu*N)

def constant(mu=mu):          return mu
def uniform(mu=mu, width=mu): return random.uniform(mu-width/2, mu+width/2)
def gauss(mu=mu, sigma=mu/5): return random.gauss(mu, sigma)
def beta(alpha=2, beta=3):    return random.betavariate(alpha, beta)
def pareto(alpha=4):          return random.paretovariate(alpha)
def custom(J=J,N=N,mu=mu):
        P1=N*.5
        if J < P1:
            return mu *.1
        else:
            return mu*1.9


def normalize(numbers, total):
    "Scale the numbers so that they add up to total."
    factor = total / float(sum(numbers))
    return [x * factor for x in numbers]

def random_split(X, Y):
    "Take all the money in the pot and divide it randomly between X and Y."
    pot = X + Y
    m = random.uniform(0, pot)
    return m, pot - m

def winner_take_most(X, Y, most=3/4.):
    "Give most of the money in the pot to one of the parties."
    pot = X + Y
    m = random.choice((most * pot, (1 - most) * pot))
    return m, pot - m

def winner_take_all(X, Y):
    "Give all the money in the pot to one of the actors."
    return winner_take_most(X, Y, 1.0)

def redistribute(X, Y):
    "Give 55% of the pot to the winner; 45% to the loser."
    return winner_take_most(X, Y, 0.55)

def split_half_min(X, Y):
    """The poorer actor only wants to risk half his wealth;
    the other actor matches this; then we randomly split the pot."""
    pot = min(X, Y)
    m = random.uniform(0, pot)
    return X - pot/2. + m, Y + pot/2. - m

def split_neutral_growth(X, Y):
    """The size of the pot grows for each transactionThe poorer actor only wants to risk 1/xx of his wealth;
    the other actor matches this; then we randomly split the pot."""
    xx=5.0
    gr=.0
    X=X*(1+gr)
    Y=Y*(1+gr)
    pot = 2.0*min(X, Y)/xx
    m = random.uniform(0, pot)
    return X -(pot/2.) + m, Y + (pot/2.) - m





def more_to_the_poor(X, Y):
    """The poorer actor only wants to risk 1/xx of his wealth;
    the other actor matches this; then we randomly split the pot but the poorer agent
    gets the larger amount."""
    xx=5.0
    pot = 2.0*min(X,Y)/xx
    pot2=pot/2.
    m = random.uniform(0, pot)
    mm=max(pot-m,m)
    if X <= Y :
        wx=mm
        wy=pot-mm
    else:
        wy=mm
        wx=pot-mm

    return X -(pot2) + wx, Y - (pot2)  + wy


def favor_the_poor(X, Y):
    """The poorer actor only wants to risk 1/xx of his wealth;
    the other actor matches this; then we randomly split the pot but the poorer agent
    gets the larger amount."""
    xx=5.0
    pot = 2.0*min(X,Y)/xx
    pot2=pot/2.
    m = random.uniform(0, pot)
    mm=max(pot-m,m)
    luck=random.uniform(0,1.0)
    if X <= Y and luck >.4 :
        wx=mm
        wy=pot-mm
    else:
        wy=mm
        wx=pot-mm

    return X -(pot2) + wx, Y - (pot2)  + wy

def more_to_the_rich(X, Y):
    """The poorer actor only wants to risk 1/xx of his wealth;
    the other actor matches this; then we randomly split the pot but the richer agent gets
    the bigger amount."""
    xx=5.0
    pot = 2.0*min(X,Y)/xx
    pot2=pot/2.
    m = random.uniform(0, pot)
    mm=max(pot-m,m)
    if X >= Y :
        wx=mm
        wy=pot-mm
    else:
        wy=mm
        wx=pot-mm

    return X -(pot2) + wx, Y - (pot2) + wy




def anyone(pop): return random.sample(range(len(pop)), 2)

def nearby(pop, k=5):
    i = random.randrange(len(pop))
    j = i + random.choice((1, -1)) * random.randint(1, k)
    return i, (j % len(pop))

def nearby1(pop): return nearby(pop, 1)


def simulate(population, transaction_fn, interaction_fn, T, percentiles, record_every):
    "Run simulation for T steps; collect percentiles every 'record_every' time steps."
    results = []
    for t in range(T):
        i, j = interaction_fn(population)
        population[i], population[j] = transaction_fn(population[i], population[j])
        if t % record_every == 0:
            results.append(record_percentiles(population, percentiles))
    return results


def simulate_e(population, transaction_fn, interaction_fn, T, percentiles, record_every):
    "Run simulation for T steps; collect percentiles every 'record_every' time steps."
    res = []
    pop_new=[]
    for t in range(T):
        i, j = interaction_fn(population)
        population[i], population[j] = transaction_fn(population[i], population[j])
##        if t % record_every == 0:
##            print population
##            results.append(population)
##            print results


        res.append(population[:]) # you must use the colon python lists very mysterious

    return res


def population_status():
    population = sample(distribution=gauss, N=100, mu=100)
    #population = sorted(population, reverse=True)
    return population

def report(distribution=gauss, transaction_fn=random_split, interaction_fn=anyone, N=N, mu=mu, T=5*N,
           percentiles=(1, 10, 25, 33.3, 50, -33.3, -25, -10, -1), record_every=25):
    "Print and plot the results of the simulation running T steps."
    # Run simulation
    population = sample(distribution, N, mu)
    pop_start=population
    results = simulate(population, transaction_fn, interaction_fn, T, percentiles, record_every)
    # Print summary
    print('Simulation: {} * {}(mu={}) for T={} steps with {} doing {}:\n'.format(
          N, name(distribution), mu, T, name(interaction_fn), name(transaction_fn)))
    fmt = '{:6}' + '{:10.2f} ' * len(percentiles)
    print(('{:6}' + '{:>10} ' * len(percentiles)).format('', *map(percentile_name, percentiles)))
    for (label, nums) in [('start', results[0]), ('mid', results[len(results)//2]), ('final', results[-1])]:
        print fmt.format(label, *nums)
    # Plot results
    for line in zip(*results):
        plt.plot(line)
    plt.show()



def report_e(distribution=gauss, transaction_fn=more_to_the_poor, interaction_fn=anyone,N=N, mu=mu, T=5*N,
           percentiles=(1, 10, 25, 33.3, 50, -33.3, -25, -10, -1), record_every=25):
    "Print and plot the results of the simulation running T steps for N agents."
    # Run simulation
    print
    print "Number of Agents N= ",N
    print "Number of transactions T= ",T
    print
    population = sample_cust(distribution,N, mu) # se the sample_cust for custom distribution only
    pop_start=list(population)
    #print "pop_t0= ", pop_start
    results = simulate_e(population, transaction_fn, interaction_fn, T, percentiles, record_every)
##    # Print summary
##    print('Simulation: {} * {}(mu={}) for T={} steps with {} doing {}:\n'.format(
##          N, name(distribution), mu, T, name(interaction_fn), name(transaction_fn)))
##    fmt = '{:6}' + '{:10.2f} ' * len(percentiles)
##    print(('{:6}' + '{:>10} ' * len(percentiles)).format('', *map(percentile_name, percentiles)))
##    for (label, nums) in [('start', results[0]), ('mid', results[len(results)//2]), ('final', results[-1])]:
##        print fmt.format(label, *nums)
    # Plot results
##    for line in zip(*results):
##        plt.plot(line)
##    plt.show()

    proc_results_f(results,pop_start)



def record_percentiles(population, percentiles):
    "Pick out the percentiles from population."
    population = sorted(population, reverse=True)
    N = len(population)
    return [population[int(p*N/100.)] for p in percentiles]

def percentile_name(p):
    return ('median' if p == 50 else
            '{} {}%'.format(('top' if p > 0 else 'bot'), abs(p)))

def name(obj):
    return getattr(obj, '__name__', str(obj))

def proc_results(results,pop_start):
    """ Process the results by summing all the outcomes of each transaction
    for each agent
    """
    popu=[]
    print
    #print pop_start
    c=zip(*results)  # converts results to N items each with T elements

    for K in c:
        popu.append(sum(K))   #Now sum all T elements for each N
    P=sorted(popu,reverse=True)
    print "minimum=", min(popu)
    print "maximum=", max(popu)
    print "median=",np.median(popu)
    print "ratioMax/min = ",max(popu)/min(popu)
    print "std deviation=",np.std(popu)
    plt.figure(1)
    plt.title("Initial Wealth of Agents")
    plt.xlabel("AGENTS")
    plt.ylabel("Initial Wealth of Each Agent")
    plt.plot(pop_start)
    plt.figure(2)
    plt.title("Sum of Winnings for Each Agent")
    plt.xlabel("AGENTS")
    plt.ylabel("Sum of Winnings for all Transaction")
    plt.plot(popu)
    plt.figure(3)
    plt.xlabel("AGENTS SORTED Descending")
    plt.plot(P)
    plt.show()

def proc_results_f(results,pop_start):
    """ Process the results by getting the statistics for the final
    state of each of N agents after T transactions
    """
    res=results # a list of lists, the sub-list N contains the results for each agent at the end of the Nth transaction


    len_res=len(res)
    final=res[len_res-1]
    print "final=",len(final)

    print "The Statistics for the final state for all agents"
    print "minimum=", min(res[len_res-1])
    print "maximum=", max(res[len_res-1])
    print "median=",np.median(res[len_res-1])
    print "mean= ",np.mean(res[len_res-1])
    print "ratioMax/min = ",max(res[len_res-1])/min(res[len_res-1])
    print "std deviation=",np.std(res[len_res-1])
    print "Total Wealth of all agents=",np.sum(res[len_res-1])
    P=sorted(res[len_res-1],reverse=True)


    bottom=(N/2)-1
    upper=N-1
    print "The Statistics for the final state for poorest agents"
    finPoor=final[0:bottom]
    print "minimum=", min(finPoor)
    print "maximum=", max(finPoor)
    print "median=",np.median(finPoor)
    print "mean= ",np.mean(finPoor)
    print "ratioMax/min = ",max(finPoor)/min(finPoor)
    print "std deviation=",np.std(finPoor)
    print "Total Wealth of poorests agents=",np.sum(finPoor)

    print "The Statistics for the final state for richest agents"
    finRich=final[(bottom+1):upper]
    print "minimum=", min(finRich)
    print "maximum=", max(finRich)
    print "median=",np.median(finRich)
    print "mean= ",np.mean(finRich)
    print "ratioMax/min = ",max(finRich)/min(finRich)
    print "std deviation=",np.std(finRich)
    print "Total Wealth of richests agents=",np.sum(finRich)


    plt.figure(4)
    plt.xlabel("AGENTS")
    plt.ylabel("$ Final Wealth of Each Agent")
    plt.title("Final Balance")
    plt.plot(res[len_res-1])

    plt.figure(1)
    plt.title("Initial Wealth of Agents")
    plt.xlabel("AGENTS")
    plt.ylabel("Initial Wealth of Each Agent")
    plt.plot(pop_start)

##    plt.figure(3)
##    plt.xlabel("AGENTS SORTED Descending Poorest")
##    plt.plot(Q)
##

    plt.figure(2)
    plt.xlabel("AGENTS SORTED Descending")
    plt.plot(P)
    plt.show()
def main():
    #report(constant,favor_the_poor,record_every=1,N=1000,T=100000)
     report_e(custom,split_neutral_growth,record_every=1,N=1000,T=100000)


if __name__ == '__main__':
    main()

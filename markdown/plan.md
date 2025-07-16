# Current Plan

We start by identifying all possible valid plcaement configurations. For each card in our hand, we need to identify a spot where it can be placed, with each location having 4 possible rotations per tile.

Once we have this list, we want to obtain information about the value each placement will add to us. Our metrics is total points, so we want to find a way to evaluate each placement position for the expected amount of points we will get, constrasted against the 'cost' of using a meeple. Although maximum points wins, given that games are a variable length, our target metric will be to increase the average points we get per turn.

With that in mind, these are the variables we need to calculate/identify for each placed tile:

1. Points earned from adding the structures with existing meeples.
2. Points earned by adding a meeple to this specific structure on this specific tile
3. Cost of 

5. Find the probability that a given structure will complete, with or without our input.


For 5. Probability, We need to do the same calculations as above, in the following manner:

1. For a given structure we want to evaluate/complete, Find the list of all missing tiles



# TO DO:


1. Add progressive weights for freeing meeples, rather than all at once when freeing.
2. Prioritise reducing number of open ends on structures we own
2. Incorporate probability of completion into the formula
3. Add separate computing for Tri-road

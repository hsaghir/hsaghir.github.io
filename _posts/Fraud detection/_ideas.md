

# message passing

- We want to calculate marginal distributions for every node in a probabilistic graphical model. If we write down the marginalizations on paper we see that there are two fundamental operations on for variables and factors in the factor graph. When we go from variables to a factor we sum over variables and when we go from factors to variables we do a product. 

- We have to sum all variables out once for each node when we want to calculate the marginal for that node. So to calculate the marginals for all nodes means that we have to do this time the number of nodes.  However, a lot of these calculations will be duplicate since once a marginal for a part of the factorized graph is computed, we can save it and reuse it for marginalization in the case of another node. 

- We can calculate the marginals for all nodes by choosing a node and setting it as the root of the tree, then two passes from leaves to root and root to leaves will give us the marginals for all nodes in the graph. If the graph is a tree, the marginals will be exact but if there are loops in the graph, the marginals will be an approximation.  